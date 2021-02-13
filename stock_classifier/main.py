#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
from tokenization_kobert import KoBertTokenizer
from transformers import BertModel



import torch
import torch.nn as nn
import pytorch-lightning as pl
import naver_data import SentimentalData


class BertSentimental(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.bert = bert
        self.linear = nn.Linear(bert.config.hidden_size, 2)

    def prepare_data(self):
        tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')

        def _tokenize(x):
            x['input_ids'] = tokenizer.encode(
                    x[],
                    max_length = 128,
                    pad_to_length = True
                    )
            return x


    def forward(self, input_ids, attention_mask, token_type_ids):

        h, _, attn = self.bert(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        h_cls = h[:, 0]
        logits = self.W(h_cls)
        return logits, attn


    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, label = batch

        y_hat, attn = self(input_ids, attention_mask, token_type_ids)

        loss = F.cross_entropy(y_hat, label)

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}


    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)

    def train_dataloader(self):







if __name__ == '__main__':
    news_classifier = BertSentimental()
    trainer = pl.Trainer(gpus=-1)
    trainer.fit(news_classifier)
    token = tokenizer.tokenize("[CLS] 한국어 모델을 공유합니다.[SEP] 여러분 모두 반가워요.")
    ids = kotokenizer.convert_tokens_to_ids(token)
    ids = torch.tensor(ids)
    ids = torch.unsqueeze(ids, 0)
    print(ids.size())
    model = BertModel.from_pretrained("monologg/kobert")
    output = model(ids)
    print(f"last hidden : {output[0].size()}")
    print(f"pooler : {output[1].size()}")
    print(model.embeddings.word_embeddings)
