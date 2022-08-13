from transformers import BartTokenizer, BartForConditionalGeneration, BartTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import AutoConfig
import torch
from args import get_args
from tqdm import tqdm
# from tensorboard_logger import configure, log_value
import os
from data import read_data
from metric import Evaluator
import time
import numpy as np
from models import MultiModal_BART
import warnings
warnings.filterwarnings('ignore')
from models import mask_tokens

def train_one_epoch(config, train_dataloader, model, optim, criterion, epoch, tokenizer):
    model.train()
    pbar = tqdm(total=len(train_dataloader))

    batch_idx = 0
    total_loss = 0

    for src_ids, src_mask, src_NN_mask, tgt_ids, tgt_mask, tgt_NN_mask, img in train_dataloader:
        input_ids, labels = mask_tokens(tgt_ids, tgt_NN_mask, tokenizer)
        pbar.update(1)
        if config.use_gpu:
            input_ids, labels, tgt_mask = input_ids.cuda(), labels.cuda(), tgt_mask.cuda()
            img = img.cuda()
            tgt_ids = tgt_ids.cuda()

        optim.zero_grad()
        # print(input_ids)
        outputs = model(input_ids=input_ids, attention_mask=tgt_mask, decoder_ids=tgt_ids, img=img)
        # print(outputs[0])
        logits = outputs

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

def evaluation(config, model, data_dataloader, evaluator, epoch, tokenizer):
    model.eval()
    pbar = tqdm(total=len(data_dataloader))
    batch_idx = 0
    loss_total = 0
    for src_ids, src_mask, src_NN_mask, tgt_ids, tgt_mask, tgt_NN_mask, img in data_dataloader:
        input_ids, labels = mask_tokens(tgt_ids, tgt_NN_mask, tokenizer)
        pbar.update(1)
        if config.use_gpu:
            input_ids, labels, tgt_mask = input_ids.cuda(), labels.cuda(), tgt_mask.cuda()
            img = img.cuda()
            tgt_ids = tgt_ids.cuda()

        with torch.no_grad():
            output = model(input_ids, tgt_mask, tgt_ids, img=img)
        loss_total += criterion(output.view(-1, output.shape[-1]), labels.view(-1)).item()
        batch_idx += 1
    pbar.close()
    return loss_total / len(data_dataloader)

def save_model(epoch, model, score, config):
    torch.save(model.state_dict(), config.save_folder+'/'+str(epoch)+'_'+str(score)+'.pkl')
    return

if __name__ == '__main__':
    results = {
            'val_loss': [], 
            'test_rouge1': [], 
            'test_rouge2': [],
            'test_rougel': [],
            'best': 999999,
            'best_epoch': []
            }

    config = get_args()
    tokenizer = BartTokenizerFast.from_pretrained(config.pretrained_model)

    train_dataloader, valid_dataloader, test_dataloader = read_data(config, tokenizer)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    config.save_folder = os.path.join(
        os.path.join(config.save_path, config.model_name),
        str(time.strftime('%b%d%H%M%S', time.localtime()))
        )
    if not os.path.exists(config.save_folder):
        os.makedirs(config.save_folder)
    model = MultiModal_BART(config)

    
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad is not False ,model.parameters()),lr=config.learning_rate)
    if config.use_gpu:
        model = model.cuda()
    evaluator = Evaluator()
    eval_log_file = config.save_folder + '/eval_log.txt'

    for epoch in range(config.Epoch):
        train_one_epoch(config, train_dataloader, model, optimizer, criterion, epoch+1, tokenizer)
        print('validing ... ...')
        val_loss = evaluation(config, model, valid_dataloader, evaluator, epoch, tokenizer)
        results['val_loss'].append(val_loss)

        with open(eval_log_file, 'a') as fl:
            print('epoch {:4d} val loss {:.5f}'.format(epoch, val_loss), file=fl)

        if val_loss < results['best']:
            results['best'] = val_loss
            results['best_epoch'].append(epoch)
            save_model(epoch, model, val_loss, config)
        print('epoch: '+ str(epoch) + 'val_loss: ' + str(results['val_loss']))
        print('best epoch' + str(results['best_epoch']))


            