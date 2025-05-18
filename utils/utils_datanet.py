

# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta

# FastTraffic
def tranHex2Dec(content):
    new = [int(i.strip("\n"),16) for i in content]

    return new

#IoTJ
def Dec(content):
    new = [int(i.strip("\n")) for i in content]
    return new



def build_dataset(config):
   
    tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
   
    def load_dataset(path):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                #content,label,_,_ = lin.split('\t')
                content,label = lin.split('\t')

                token = tokenizer(content)
                token = token[:1480]
                contents.append((tranHex2Dec(token), int(label)))
       
        return contents  # [([...], 0), ([...], 1), ...]

    train = load_dataset(config.train_path)
   
    dev = load_dataset(config.dev_path)
    test = load_dataset(config.test_path)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.FloatTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        #print("Original x shape:", x.shape)
        x = torch.reshape(x,(x.shape[0],1480))
        return x,y
        
    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time, test = 5, data=None):
    end_time = time.time()
    time_dif = end_time - start_time
    
    # preprocess time
    if test == 0:

        if data == 'MALAYAGT':
            average_time = time_dif / 1046795
        
        elif data == 'ISCXVPN2016':
            average_time = time_dif / 1090302

        return time_dif, average_time
    
    # Testing Time
    elif test == 1:

        if data == 'MALAYAGT':
            average_time = time_dif / 104679
        
        elif data == 'ISCXVPN2016':
            average_time = time_dif / 109031

        return time_dif, average_time
    
    # Training time
    elif test == 2:

        if data == 'MALAYAGT':
            average_time = time_dif / 837436
        
        elif data == 'ISCXVPN2016':
            average_time = time_dif / 872241

        return time_dif, average_time
    
    else:
        return timedelta(seconds=int(round(time_dif)))

