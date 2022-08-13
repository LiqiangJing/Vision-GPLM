

import os
import json
import random
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
from PIL import Image
import torchvision
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
        src_ids, src_mask = self.get_tokenized(self.src[index], self.config.max_length_src)
        dii_tgt, dii_tgt_mask = self.get_tokenized(self.tgt[index], self.config.max_length_tgt)
        img = Image.open(
            os.path.join(self.img_path, f"{index+1}.jpg")
        ).convert("RGB")
        simg = self.transform(img)
        # simg = self.imgs[index, :, :]
        return src_ids, src_mask, dii_tgt, dii_tgt_mask, simg

    def get_tokenized(self, text, max_length):
        tokens = self.tokenizer.encode_plus(text, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
        # tokens = self.tokenizer.encode_plus(text, max_length=max_length, pad_to_max_length=True, truncation=True, return_tensors='pt')
        return tokens['input_ids'].squeeze(), tokens['attention_mask'].squeeze()
    
    def __len__(self):
        return len(self.src)

def read_data(config, tokenizer):
    train_dataset = NewsDataset('/home/limw/xyy/data/corpus/train_sent.txt', '/home/limw/xyy/data/corpus/train_title.txt', tokenizer, config, img_path='/home/limw/xyy/data/corpus/images_train')
    valid_dataset = NewsDataset('/home/limw/xyy/data/corpus/dev_sent.txt','/home/limw/xyy/data/corpus/dev_title.txt',tokenizer, config, img_path='/home/limw/xyy/data/corpus/images_dev')
    test_dataset = NewsDataset('/home/limw/xyy/data/corpus/test_sent.txt', '/home/limw/xyy/data/corpus/test_title.txt', tokenizer, config, img_path='/home/limw/xyy/data/corpus/images_test')

    # train_dataset = NewsDataset('/home/admin/workspace/project/mmss/data/corpus/train_sent.txt', '/home/admin/workspace/project/mmss/data/corpus/train_title.txt', tokenizer, config, img_path='/home/admin/workspace/project/mmss/visualer/pretraining/train_img.pt')
    # valid_dataset = NewsDataset('/home/admin/workspace/project/mmss/data/corpus/valid_sent.txt','/home/admin/workspace/project/mmss/data/corpus/valid_title.txt',tokenizer, config, img_path='/home/admin/workspace/project/mmss/visualer/pretraining/valid_img.pt')
    # test_dataset = NewsDataset('/home/admin/workspace/project/mmss/data/corpus/test_sent.txt', '/home/admin/workspace/project/mmss/data/corpus/test_title.txt', tokenizer, config, img_path='/home/admin/workspace/project/mmss/visualer/pretraining/test_img.pt')

    
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size_test, num_workers=config.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size_test, num_workers=config.num_workers)
    print('train {} valid {} test {}'.format(len(train_dataset), len(valid_dataset), len(test_dataset)))
    return train_dataloader, valid_dataloader, test_dataloader



