from transformers import BartTokenizer, BartForConditionalGeneration
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import AutoConfig
import torch
from args import get_args
from tqdm import tqdm
# from tensorboard_logger import configure, log_value
import timm
import os
from data import read_data
from metric import Evaluator
import time
import numpy as np
from models import MultiModal_BART
import warnings
warnings.filterwarnings('ignore')

def shift_tokens_right(input_ids, pad_token_id):
  """ Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
      This is taken directly from modeling_bart.py
  """
  prev_output_tokens = input_ids.clone()
  index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
  prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
  prev_output_tokens[:, 1:] = input_ids[:, :-1]
  return prev_output_tokens

def train_one_epoch(config, train_dataloader, model, optim, criterion, epoch):
    model.train()
    pbar = tqdm(total=len(train_dataloader))

    batch_idx = 0
    total_loss = 0

    for src_ids, src_mask, tgt_ids, tgt_mask, img in train_dataloader:
        pbar.update(1)
        if config.use_gpu:
            src_ids, src_mask, tgt_ids, tgt_mask = src_ids.cuda(), src_mask.cuda(), tgt_ids.cuda(), tgt_mask.cuda()
            img = img.cuda()

        optim.zero_grad()

        outputs = model(src_ids, src_mask, tgt_ids, tgt_mask, img)
        # print(outputs[0])
        logits = outputs
        labels = tgt_ids
        loss = criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))

        pbar.set_description(
            "epoch {}, step {}, loss {:.4f}".format(
                epoch, batch_idx, loss.item())
        )
        total_loss += loss.item()
        batch_idx += 1

        loss.backward()
        optim.step()
        
    pbar.close()

def tensor2texts(input_ids):
    return [tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) for w in input_ids]

def evaluation(config, model, data_dataloader, evaluator, epoch):
    model.eval()
    pbar = tqdm(total=len(data_dataloader))
    batch_idx = 0
    src = []
    ref = []
    gen = []
    for src_ids, src_mask, tgt_ids, tgt_mask, img in data_dataloader:
        pbar.update(1)
        if config.use_gpu:
            src_ids, src_mask, tgt_ids, tgt_mask = src_ids.cuda(), src_mask.cuda(), tgt_ids.cuda(), tgt_mask.cuda()
            img = img.cuda()

        with torch.no_grad():
            output = model.generate_text(
                src_ids, src_mask, img
            )
  
        gen += output
        src += tensor2texts(src_ids)
        ref += tensor2texts(tgt_ids)
        batch_idx += 1
    pbar.close()
    # print(gen)
    # print(ref)
    rouge = evaluator.rouge(refs=ref, hyps=gen)
    print(('[auto_eval] rouge 1: {:.4f} rouge 2: {:.4f} rouge l: {:.4f} \n').format(rouge[0], rouge[1], rouge[2])) 



    # with open(os.path.join(config.model_name, 'eval_log.txt'), 'a') as fl:
    #     print(('iter{:5d}: [auto_eval] rouge 1: {:.4f} rouge 2: {:.4f} rouge l: {:.4f}').format(rouge[0], rouge[1], rouge[2]), file=fl))

    # with open(os.path.join(config.model_name,'test'+ str(epoch)+'.txt'), 'w') as fw:
    #     print(('[auto_eval] rouge 1: {:.4f} rouge 2: {:.4f} rouge l: {:.4f} \n').format(rouge[0], rouge[1], rouge[2]), file=fw)）
    #     for idx in range(len(gen)):
    #         print('[raw_dii]', gen[idx], file=fw)
    return rouge

def save_model(epoch, model, score, config):
    torch.save(model.state_dict(), config.save_folder+'/'+str(epoch)+'_'+str(score)+'.pkl')
    return

if __name__ == '__main__':
    results = {
            'rouge1': [], 
            'rouge2': [],
            'rougel': [],
            'test_rouge1': [], 
            'test_rouge2': [],
            'test_rougel': [],
            'best': 0,
            'best_epoch': []
            }

    config = get_args()
    tokenizer = BartTokenizer.from_pretrained(config.pretrained_model)

    train_dataloader, valid_dataloader, test_dataloader = read_data(config, tokenizer)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    config.save_folder = os.path.join(
        os.path.join(config.save_path, config.model_name),
        str(time.strftime('%b%d%H%M%S', time.localtime()))
        )
    if not os.path.exists(config.save_folder):
        os.makedirs(config.save_folder)
    model = MultiModal_BART(config)

    if config.load_model is not None:
        print('Loading model from: ', config.load_model)
        model.load_state_dict(torch.load(config.load_model))

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    if config.use_gpu:
        model = model.cuda()
    evaluator = Evaluator()
    eval_log_file = config.save_folder + '/eval_log.txt'

    for epoch in range(config.Epoch):
        train_one_epoch(config, train_dataloader, model, optimizer, criterion, epoch+1)
        print('validing ... ...')
        val_score = evaluation(config, model, valid_dataloader, evaluator,epoch)
        results['rouge1'].append(val_score[0])
        results['rouge2'].append(val_score[1])
        results['rougel'].append(val_score[2])

        # print('epoch {:4d}  rouge {:.5f} {:.5f} {:.5f}'.format(epoch, results['rouge1']),max(results['rouge2']), max(results['rougel'])))
        print('valid epoch:'+str(epoch)+str(results))
        with open(eval_log_file, 'a') as fl:
            print('epoch {:4d} val rouge {:.5f} {:.5f} {:.5f}'.format(epoch, val_score[0], val_score[1], val_score[2]), file=fl)

        score = evaluation(config, model, test_dataloader, evaluator, epoch)

        print('test rouge {:.5f} {:.5f} {:.5f}'.format(score[0], score[1], score[2]))

        with open(eval_log_file, 'a') as fl:
            print('epoch {:4d} test rouge {:.5f} {:.5f} {:.5f}'.format(epoch, score[0], score[1], score[2]), file=fl)
        if val_score[1] > results['best']:
            results['best'] = val_score[1]
            results['best_epoch'].append(epoch)
            print('saving ... ...')
            save_model(epoch, model, results['best'], config)

