from args import get_args
from transformers import BartTokenizer
import timm
import torch
from tqdm import tqdm
from pyossfs.oss_bucket_manager import OSSFileManager
import os
import json
import random
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import nltk
from PIL import Image
import torchvision
# random.seed(0)

import warnings
warnings.filterwarnings('ignore')

from args import get_args
def read_txt(data_path):
    f = open(data_path, 'r')
    results = []
    for line in f:
        results.append(line.strip())
    f.close()
    return results


class NewsDataset(Dataset):
    def __init__(self, src_path, tgt_path, tokenizer, img_path):  ## 每句最多 20 length
        self.src = read_txt(src_path)
        self.img_path = img_path
        self.transform = torchvision.transforms.Compose([
                                torchvision.transforms.Resize(256),
                                torchvision.transforms.CenterCrop(224),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])
                            ])

    def __getitem__(self, index):
        img = Image.open(
            os.path.join(self.img_path, f"{index+1}.jpg")
        ).convert("RGB")
        simg = self.transform(img)
        return simg

    def get_tokenized(self, text, max_length):
        tokens = self.tokenizer.encode_plus(text, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
        # tokens = self.tokenizer.encode_plus(text, max_length=max_length, pad_to_max_length=True, truncation=True, return_tensors='pt')
        return tokens['input_ids'].squeeze(), tokens['attention_mask'].squeeze()
    
    def __len__(self):
        return len(self.src)

def read_data(tokenizer):
    train_dataset = NewsDataset('/home/admin/workspace/project/mmss/data/corpus/train_sent.txt', '/home/admin/workspace/project/mmss/data/corpus/train_title.txt', tokenizer, img_path='/home/admin/workspace/project/mmss/data/corpus/images_train')
    valid_dataset = NewsDataset('/home/admin/workspace/project/mmss/data/corpus/valid_sent.txt','/home/admin/workspace/project/mmss/data/corpus/valid_title.txt',tokenizer, img_path='/home/admin/workspace/project/mmss/data/corpus/images_valid')
    test_dataset = NewsDataset('/home/admin/workspace/project/mmss/data/corpus/test_sent.txt', '/home/admin/workspace/project/mmss/data/corpus/test_title.txt', tokenizer, img_path='/home/admin/workspace/project/mmss/data/corpus/images_test')

    train_dataloader = DataLoader(train_dataset, batch_size=128)
    valid_dataloader = DataLoader(valid_dataset, batch_size=128)
    test_dataloader = DataLoader(test_dataset, batch_size=128)
    print('train {} valid {} test {}'.format(len(train_dataset), len(valid_dataset), len(test_dataset)))
    return train_dataloader, valid_dataloader, test_dataloader



class SWIN(torch.nn.Module):
    def __init__(self):
        super(SWIN, self).__init__()
        self.img_encoder = timm.create_model("swin_base_patch4_window7_224", pretrained=True)

    def get_emb(self, img):
        x = self.img_encoder.patch_embed(img)
        if self.img_encoder.absolute_pos_embed is not None:
            x = x + self.img_encoder.absolute_pos_embed
        x = self.img_encoder.pos_drop(x)
        x = self.img_encoder.layers(x)
        x = self.img_encoder.norm(x)

        x = self.img_encoder.avgpool(x.transpose(1, 2)) # B C 1
        x = x.transpose(1,2) # B,1,C
        # print(x.shape)
        return x

def get_fea(data_dataloader, model):
    img_feas = None
    pbar = tqdm(total=len(data_dataloader))
    for img in data_dataloader:
        img = img.cuda()
        with torch.no_grad():
            img_fea = model.get_emb(img)
        if img_feas is None:
            img_feas = img_fea
        else:
            img_feas = torch.cat((img_feas, img_fea), 0)
        pbar.update(1)
    pbar.close()
    print(img_feas.shape)
    return img_feas

config = get_args()
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
train_dataloader, valid_dataloader, test_dataloader = read_data(tokenizer)
model = SWIN()
model = model.cuda()
model.eval()


torch.save(get_fea(test_dataloader, model), "./test_img.pt")
torch.save(get_fea(train_dataloader, model), "./train_img.pt")
torch.save(get_fea(valid_dataloader, model), "./valied_img.pt")
