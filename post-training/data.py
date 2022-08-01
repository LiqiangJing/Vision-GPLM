
import os
import json
import random
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
from PIL import Image
import torchvision
import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize, WhitespaceTokenizer
# random.seed(0)
from args import get_args


def read_txt(data_path):
    f = open(data_path, 'r')
    results = []
    for line in f:
        results.append(line.strip())
    f.close()
    return results


class NewsDataset(Dataset):
    def __init__(self, src_path, tgt_path, tokenizer, config, img_path):  ## 每句最多 20 length
        self.src = read_txt(src_path)
        self.tgt = read_txt(tgt_path)

        self.config = config
        self.tokenizer = tokenizer

        self.img_path = img_path
        # self.imgs = torch.load(img_path).cpu()
        # print(self.imgs.shape)
        # print(self.imgs.device)
        # print(self.imgs[9].shape)

        self.transform = torchvision.transforms.Compose([
                                torchvision.transforms.Resize(256),
                                torchvision.transforms.CenterCrop(224),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])
                            ])


    def __getitem__(self, index):
        src_ids, src_mask, src_NN_mask = self.get_tokenized(self.src[index], self.config.max_length_src)
        dii_tgt, dii_tgt_mask, dii_tgt_NN_mask = self.get_tokenized(self.tgt[index], self.config.max_length_tgt)
        img = Image.open(
            os.path.join(self.img_path, f"{index+1}.jpg")
        ).convert("RGB")
        simg = self.transform(img)
        # simg = self.imgs[index, :, :]
        return src_ids, src_mask, src_NN_mask, dii_tgt, dii_tgt_mask, dii_tgt_NN_mask, simg

    def get_tokenized(self, text, max_length):
        tokens = self.tokenizer.encode_plus(text, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt',return_offsets_mapping=True)
        span_generator = WhitespaceTokenizer().span_tokenize(text)
        spans = [span for span in span_generator]
        text2 = [text[span[0]: span[1]] for span in spans]
        pos = pos_tag(text2)
        NN_span = []
        for index,term in enumerate(pos):
            if term[1] in ['NN', 'NNS', 'NNP', 'NNPS']:
                NN_span.append(spans[index])
        NN_mask = torch.zeros_like(tokens['attention_mask'].squeeze())
        offsets = tokens['offset_mapping'].squeeze()
        cur_1 = 0
        cur_2 = 1 # 起始有<EOS>
        end_1 = len(NN_span)
        end_2 = offsets.shape[0]

        # 双指针扫描
        while cur_1!=end_1 and cur_2!=end_2:
            # 如果当前单词超过了
            if offsets[cur_2][0] > NN_span[cur_1][1]:
                cur_1+=1
            elif offsets[cur_2][1] < NN_span[cur_1][0]:
                cur_2 += 1
            else:
                NN_mask[cur_2] = 1
                cur_2 += 1
        ### debug***********************************************
        # temp_list = tokens.tokens()
        # temp = []
        # for ind, val in enumerate(NN_mask):
        #     if val:
        #         temp.append(temp_list[ind])
        return tokens['input_ids'].squeeze(), tokens['attention_mask'].squeeze(), NN_mask
    
    def __len__(self):
        return len(self.src)

def read_data(config, tokenizer):
    train_dataset = NewsDataset('/home/share/xujunhao/data/corpus/train_sent.txt', '/home/share/xujunhao/data/corpus/train_title.txt', tokenizer, config, img_path='/home/share/xujunhao/data/corpus/images_train')
    valid_dataset = NewsDataset('/home/share/xujunhao/data/corpus/dev_sent.txt','/home/share/xujunhao/data/corpus/dev_title.txt',tokenizer, config, img_path='/home/share/xujunhao/data/corpus/images_dev')
    test_dataset = NewsDataset('/home/share/xujunhao/data/corpus/test_sent.txt', '/home/share/xujunhao/data/corpus/test_title.txt', tokenizer, config, img_path='/home/share/xujunhao/data/corpus/images_test')

    # train_dataset = NewsDataset('/home/share/xujunhao/data/corpus/train_sent.txt', '/home/share/xujunhao/data/corpus/train_title.txt', tokenizer, config, img_path='/home/admin/workspace/project/mmss/visualer/pretraining/train_img.pt')
    # valid_dataset = NewsDataset('/home/share/xujunhao/data/corpus/dev_sent.txt','/home/share/xujunhao/data/corpus/valid_title.txt',tokenizer, config, img_path='/home/admin/workspace/project/mmss/visualer/pretraining/valid_img.pt')
    # test_dataset = NewsDataset('/home/share/xujunhao/data/corpus/test_sent.txt', '/home/share/xujunhao/data/corpus/test_title.txt', tokenizer, config, img_path='/home/admin/workspace/project/mmss/visualer/pretraining/test_img.pt')


    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size_test, num_workers=config.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size_test, num_workers=config.num_workers)
    print('train {} valid {} test {}'.format(len(train_dataset), len(valid_dataset), len(test_dataset)))
    return train_dataloader, valid_dataloader, test_dataloader



