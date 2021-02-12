#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
from tokenization_kobert import KoBertTokenizer

kotokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
print(kotokenizer.tokenize("[CLS] 한국어 모델을 공유합니다.[SEP] 여러분 모두 반가워요."))
