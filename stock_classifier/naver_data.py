#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from tqdm import tqdm
import sys
sys.path.append('..')
import json
from tokenization_kobert import KoBertTokenizer
from typing import List
class SentimentalData(torch.utils.data.Dataset):
    def __init__(self, infile):
        """
        데이터셋 전처리 해주는 부분
        """
        self.tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
        self.sentence = [] 
        self.label =  []
        self.sentence_dict = {}
        line_cnt = 0
        self.token2idx = self.tokenizer.token2idx
        self.idx2token = {v: k for k, v in self.token2idx.items()}
        with open(infile, "r") as f:
            for line in f:
                line_cnt += 1
                data = json.loads(line)
                self.label.append(data["label"])
                self.sentence.append(data["doc"])
       
        self.sentence_dict=self.encode_batch(self.sentence, 128)     

    def __len__(self):
        assert len(self.label) == len(self.sentence) 
        return len(self.label)
        """
        데이터셋의 길이. 즉 ,총 샘플의 수를 적어주는 부분
        """

    def __getitem__(self, idx):
        return (
            torch.tensor(self.label[idx]),
            torch.tensor(self.sentence_dict['input_ids'][idx]),
            torch.tensor(self.sentence_dict['attention_mask'][idx]),
            torch.tensor(self.sentence_dict['token_type_ids'][idx])
        )


        """
        데이터셋에서 특정 1개의 샘플을 가져오는 함수
        """

    def encode_batch(self, x: List[str], max_length):
        max_len = 0
        result_tokenization = []

        for i in x:
            tokens = self.tokenizer.encode(i, max_length=max_length, truncation=True)
            result_tokenization.append(tokens)

            if len(tokens) > max_len:
                max_len = len(tokens)

        padded_tokens = []
        for tokens in result_tokenization:
            padding = (torch.ones(max_len) * self.token2idx["[PAD]"]).long()
            padding[: len(tokens)] = torch.tensor(tokens).long()
            padded_tokens.append(padding.unsqueeze(0))

        padded_tokens = torch.cat(padded_tokens, dim=0).long()
        mask_tensor = torch.ones(padded_tokens.size()).long()

        attention_mask = torch.where(
            padded_tokens == self.token2idx["[PAD]"], padded_tokens, mask_tensor * -1
        ).long()
        attention_mask = torch.where(
            attention_mask == -1, attention_mask, mask_tensor * 0
        ).long()
        attention_mask = torch.where(
            attention_mask != -1, attention_mask, mask_tensor
        ).long()

        return {
            "input_ids": padded_tokens.long(),
            "attention_mask": attention_mask.long(),
            "token_type_ids" : torch.tensor(([[0]*128] * len(result_tokenization))).long()  
        }
