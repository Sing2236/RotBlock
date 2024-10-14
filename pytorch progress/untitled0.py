#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:49:39 2024

@author: mant
"""
import spacy
from torch import nn
from torchtext.data import Field, TabularDataset, BucketIterator, ReversibleField

spacy_en = spacy.load("en_core_web_sm")

def tokenize(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


title = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)
score = Field(sequential=False, use_vocab=False)

fields = {'title': ('t', title), 'score':('s', score)}

train_data, test_data = TabularDataset.splits(
    path='brainrot_data',
    train='brainrot_dataset.tsv',
    test='brainrot_testset.tsv',
    format='tsv',
    fields=fields)

title.build_vocab(train_data, vectors="glove.6B.100d")

train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    sort_key=lambda x: len(x.title),
    batch_sizes=(12, 24, 36),
    device="cuda")

vocab = title.vocab
vocab.embed = nn.Embedding(len(vocab), embedding_dim=100)
vocab.embed.weight.data.copy_(vocab.vectors)

for batch in train_iterator:
    print(batch.t)