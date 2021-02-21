#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
from transformers import BertModel
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.loggers import TensorBoardLogger
from tokenization_kobert import KoBertTokenizer
import torch
import torch.nn as nn
import pytorch_lightning as pl
from naver_data import SentimentalData
import pdb
import torch.nn.functional as F
import warnings

class BertSentimental(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('monologg/kobert')
        self.linear = nn.Linear(self.bert.config.hidden_size, 2)
        self.tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
        self.batch_size = 64
        self.lr = 3e-05

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        h_cls = outputs[0][:, 0]
        logits = self.linear(h_cls)
        return logits

    def training_step(self, batch, batch_idx):
        label, input_ids, attention_mask, token_type_ids = batch
        y_hat = self.forward(input_ids, attention_mask, token_type_ids)
        loss = F.cross_entropy(y_hat, label)
        acc = accuracy(y_hat, label)
        self.log('train_loss',loss, on_epoch = True, on_step=True, prog_bar = True, logger = True)
        self.log('train_acc',acc, on_step=True, on_epoch=True,logger=True)
        #return {'loss': loss, 'progress_bar': pbar}

    def validation_step(self, batch, batch_idx):
        label, input_ids, attention_mask,token_type_ids = batch
        y_hat = self.forward(input_ids, attention_mask, token_type_ids)
        loss = F.cross_entropy(y_hat, label)
        acc = accuracy(y_hat, label)
        self.log('val_acc',acc)
        self.log('val_loss',loss)



    def test_step(self, batch, batch_idx):
        label, input_ids, attention_mask, token_type_ids = batch
        y_hat = self.forward(input_ids, attention_mask, token_type_ids)
        loss = F.cross_entropy(y_hat, label)
        acc = accuracy(y_hat, label)
        self.log('test_loss', loss, on_epoch=True, on_step=True, logger=True)
        self.log('test_acc',acc, on_epoch=True, on_step=True, logger = True)
        pbar = {'test_acc': acc}
        #return {'test_loss': loss, 'progress_bar':pbar}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.lr)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
        #                                            step_size=1000,
        #                                            gamma=0.9)

        return {'optimizer': optimizer}

    def train_dataloader(self):

        train_data = SentimentalData('data/ratings_train.json')
        return DataLoader(train_data,
                          batch_size=self.batch_size, num_workers=0)

    def val_dataloader(self):
        val_data = SentimentalData('data/ratings_val.json')
        return DataLoader(val_data,
                          batch_size = self.batch_size, num_workers=0)
    def test_dataloader(self):

        test_data = SentimentalData('data/ratings_test.json')
        return DataLoader(test_data, batch_size=self.batch_size, num_workers=0)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    news_classifier = BertSentimental()
    logger = TensorBoardLogger('tb_logs', name='BertSentimental')
    trainer = pl.Trainer(callbacks= [EarlyStopping(monitor='val_loss')],logger = logger,
                         gpus=-1,
                         accelerator='ddp',
                         precision=16,
                         auto_select_gpus=True,
                         max_epochs=10)
    #trainer = pl.Trainer(auto_lr_find = True)
    trainer.tune(news_classifier)
    trainer.test(news_classfier)
