#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
from tokenization_kobert import KoBertTokenizer
from transformers import BertModel
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


import torch
import torch.nn as nn
import pytorch_lightning as pl
from naver_data import SentimentalData
import pdb
import torch.nn.functional as F
class BertSentimental(pl.LightningModule):
    def __init__(self,model_save_path):
        super().__init__()
        self.bert = BertModel.from_pretrained('monologg/kobert')
        self.linear = nn.Linear(self.bert.config.hidden_size, 2)
        self.tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
        self.test_acc = pl.metrics.Accuracy()
        self.batch_size = 104
        self.lr = 8e-05
        self.model_save_path = model_save_path

    def forward(self, input_ids, attention_mask,token_type_ids):
        outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
<<<<<<< HEAD
        h_cls = outputs['last_hidden_state'][:, 0]
=======
        h_cls = outputs[0][:, 0]
>>>>>>> d1f99f0fdd14ce4671c4fb3ea45b8837f82fa382
        logits = self.linear(h_cls)
        return logits


    def training_step(self, batch, batch_idx):
        label, input_ids, attention_mask, token_type_ids = batch
        y_hat= self.forward(input_ids, attention_mask, token_type_ids)
        loss = F.cross_entropy(y_hat, label)
        tensorboard_logs = {'train_loss': loss}
        self.log('train_loss', loss)
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        label, input_ids, attention_mask, token_type_ids = batch
        y_hat = self.forward(input_ids, attention_mask, token_type_ids)
        loss = F.cross_entropy(y_hat, label)
        self.log('val_loss', loss)
        self.save_model()
        return {'loss':loss}


    def test_step(self, batch, batch_idx):
        label, input_ids, attention_mask, token_type_ids = batch
        y_hat = self.forward(input_ids, attention_mask, token_type_ids)
        self.log('test_acc', self.test_acc, on_step = True, on_epoch = True)

    def configure_optimizers(self):
        optimizer =  torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=1e-08)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.1) 
        
        return {'optimizer' :optimizer, 
                'scheduler' : scheduler}
    def train_dataloader(self):
        
        train_data = SentimentalData('data/ratings_train.json')
        return DataLoader(train_data, batch_size = self.batch_size, num_workers = 8)

    def test_dataloader(self):

        test_data = SentimentalData('data/ratings_test.json')
        return DataLoader(test_data, batch_size = self.batch_size, num_workers = 8)

    def save_model(self):
        if (
                self.trainer.global_rank == 0
                and self.global_step % self.save_step_interval ==0
        ):
            torch.save(
                self.model.state_dict()
                self.model_save_path +"."+ str(self.global_step)
            )




if __name__ == '__main__':
<<<<<<< HEAD
    news_classifier = BertSentimental()
    trainer = pl.Trainer(gpus=-1, accelerator = 'ddp', precision = 16)
=======
    news_classifier = BertSentimental(model_save_path = "saved/model.ckpt")
    trainer = pl.Trainer(callbacks = [EarlyStopping(monitor='val_loss')],gpus=-1, accelerator = 'ddp', precision = 16, auto_select_gpus = True , max_epochs = 10)
>>>>>>> d1f99f0fdd14ce4671c4fb3ea45b8837f82fa382
    trainer.fit(news_classifier)
