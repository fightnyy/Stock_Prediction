#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import tqdm
from tokenization_kobert import KoBertTokenizer


class SetimentalData(torch.utils.data.Datset):

    def __init__(self, infile):
        """
        데이터셋 전처리 해주는 부분
        """
        self.tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
        self.sentence = [] 
        self.label =  []

        line_cnt = 0
        with open(infile, "r") as f:
            for line in f:
                line_cnt += 1


        with open(infile, "r") as f:
            for i, line in enumerate(tqdm(f, total = line_cnt, desc = f"Loading {infile}", unit ="lines")):
                data = json.loads(line)
                self.label.append(data["label"])
                self.sentence.append([self.tokenizer.encode(p) for p in data["doc"]])

    def __len__(self):
        assert len(self.label) == len(self.senctence) 
        return len(self.label)
        """
        데이터셋의 길이. 즉 ,총 샘플의 수를 적어주는 부분
        """

    def __getitem__(self, idx):
        return (
            torch.tensor(self.label[idx]),
            torch.tensor(self.sentence[idx])
        )


        """
        데이터셋에서 특정 1개의 샘플을 가져오는 함수
        """
