#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
from tokenization_kobert import KoBertTokenizer
from transformers import BertModel
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import pytorch_lightning as pl
from naver_data import SentimentalData
import pdb
import torch.nn.functional as F
class BertSentimental(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('monologg/kobert')
        self.linear = nn.Linear(self.bert.config.hidden_size, 2)
        self.tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')

    def forward(self, input_ids, attention_mask,token_type_ids):
        outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        h_cls = outputs[0][:, 0]
        logits = self.linear(h_cls)
        return logits


    def training_step(self, batch, batch_idx):
        label, input_ids, attention_mask, token_type_ids = batch
        y_hat= self.forward(input_ids, attention_mask, token_type_ids)
        loss = F.cross_entropy(y_hat, label)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}


    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)

    def train_dataloader(self):
        
        train_data = SentimentalData('data/ratings_train.json')
        return DataLoader(train_data, batch_size = 8)

    def test_dataloader(self):

        test_data = SentimentalData('data/ratings_test.json')
        return DataLoader(test_data, batch_size = 8)






if __name__ == '__main__':
    news_classifier = BertSentimental()
    trainer = pl.Trainer(gpus=-1, accelerator = 'ddp', precision = 16)
    trainer.fit(news_classifier)
