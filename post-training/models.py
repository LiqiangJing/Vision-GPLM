from transformers.modeling_outputs import ModelOutput
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import timm
import math
import torch.nn as nn
tokenizer = BartTokenizer.from_pretrained('./bart-base')

def shift_tokens_right(input_ids, pad_token_id):
  """ Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
      This is taken directly from modeling_bart.py
  """
  prev_output_tokens = input_ids.clone()
  index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
  prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
  prev_output_tokens[:, 1:] = input_ids[:, :-1]
  return prev_output_tokens

def mask_tokens(inputs, tgt_NN_mask, tokenizer, mlm_probability=0.05):
    """
    Prepare masked tokens inputs/labels  for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    probability_matrix = probability_matrix+0.15*tgt_NN_mask
    special_tokens_mask = [
        get_special_tokens_mask(val, tokenizer) for val in labels.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

def get_special_tokens_mask(token_ids_0, tokenzier):
    """
    Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
    special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.
    """
    all_special_ids = tokenzier.all_special_ids  # cache the property
    special_tokens_mask = [1 if token in all_special_ids else 0 for token in token_ids_0]
    return special_tokens_mask

class MultiModal_BART(torch.nn.Module):
    def __init__(self, config):
        super(MultiModal_BART, self).__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(config.pretrained_model)
        self.config = config

        self.map = torch.nn.Linear(1024, 768)
        self.img_encoder = timm.create_model("swin_base_patch4_window7_224", pretrained=True)
        self.kqv_attention = Multihead_Attention(768, 6, 0.1, True)
        self.kqv_norm = nn.LayerNorm(768)

        for name, p in self.bart.named_parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask, decoder_ids=None, decoder_attention_mask=None, img=None):

        decoder_input_ids = shift_tokens_right(decoder_ids, tokenizer.bos_token_id)
        encoder_outputs = self.bart.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = encoder_outputs[0]

        if img is not None:
            img_emb = self.get_img_emb(img)
            last_hidden = last_hidden + self.kqv_attention(last_hidden, self.map(img_emb))
            last_hidden = self.kqv_norm(last_hidden)
            # last_hidden = torch.cat((self.map(img_emb), last_hidden), dim=1)

        encoder_outputs = ModelOutput(
            last_hidden_state=last_hidden,
            hidden_states=None,
            attentions=None,
        )

        outputs = self.bart(decoder_input_ids=decoder_input_ids,
                            decoder_attention_mask=decoder_attention_mask, return_dict=True, encoder_outputs=encoder_outputs,
                            use_cache=False)
        return outputs.logits

    def get_img_emb(self, img):
        # x = self.swin(img)
        x = self.img_encoder.patch_embed(img)
        if self.img_encoder.absolute_pos_embed is not None:
            x = x + self.img_encoder.absolute_pos_embed
        x = self.img_encoder.pos_drop(x)
        x = self.img_encoder.layers(x)
        x = self.img_encoder.norm(x)
        return x
        
class Multihead_Attention(nn.Module):
    """
    Multi-head Attention
    """

    def __init__(self,
        model_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True,):
        """
        initialization for variables and functions
        :param model_dim: hidden size
        :param num_heads: head number, default 8
        :param dropout: dropout probability
        """
        super(Multihead_Attention, self).__init__()

        self.head_dim = model_dim // num_heads
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.linear_keys = nn.Linear(model_dim, num_heads * self.head_dim, bias=bias)
        self.linear_values = nn.Linear(model_dim, num_heads * self.head_dim, bias=bias)
        self.linear_query = nn.Linear(model_dim, num_heads * self.head_dim, bias=bias)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)
        self.sigmoid = nn.Hardtanh(min_val=0)

    def forward(self, hidden_states, key_value_states):
        """
        run multi-head attention
        :param key: key, [batch, len, size]
        :param value: value, [batch, len, size]
        :param query: query, [batch, len, size]
        :param mask: mask
        :param layer_cache: layer cache for transformer decoder
        :param type: "self" or "context"
        :param tau: temperature, will be deprecated
        :param Bernoulli: use Bernoulli selection or not
        :return: attention output and attention weights
        """
        query = hidden_states
        key = value = key_value_states
        
        batch_size = key.size(0)
        head_dim = self.head_dim
        head_count = self.num_heads
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, head_dim) \
                .transpose(1, 2)    # [batch, head, len, head_dim]

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * head_dim)    # [batch, len, size]

        # For transformer decoder.
        # denote the device for multi-gpus
        query, key, value = self.linear_query(query),\
            self.linear_keys(key),\
            self.linear_values(value)   # [batch, len, size]
        key = shape(key)    # [batch, head, k_len, head_dim]
        value = shape(value)    # [batch, head, v_len, head_dim]

        query = shape(query)    # [batch, head, q_len, head_dim]

        key_len = key.size(2)
        query_len = query.size(2)

        query = query / math.sqrt(head_dim)

        scores = torch.matmul(query, key.transpose(2, 3))   # [batch, head, q_len, k_len]


        # use Bernoulli selection or not
        attn = self.softmax(scores)  # [batch, head, q_len, k_len]

        drop_attn = self.dropout(attn)  # [batch, head, q_len, k_len]
        context = unshape(torch.matmul(drop_attn, value))   # [batch, q_len, size]

        output = self.final_linear(context)  # [batch, q_len, size]

        top_attn = attn \
            .view(batch_size, head_count,
                  query_len, key_len)[:, 0, :, :] \
            .contiguous()   # [batch, q_len, k_len]

        return output
