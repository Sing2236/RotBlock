#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 13:59:11 2024

@author: mant
"""

#current issue: tensor size mismatch due to having updated the embedding weights during training but using a pre-made class model
#todo: figure out how to fix that

import os
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch import nn
from flask import Flask, request, jsonify # Don't forget to install flask! pip install flask
from flask_cors import CORS
import json


model_dir = './model_save/'
state_dict_dir = './model_save/state_dict.pt'

global model
global tokenizer

def init_model():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('Using GPU: ', torch.cuda.get_device_name(0))
    else:
        print('No GPU detected. Using CPU.')
        device = torch.device("cpu")

        model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=2, ignore_mismatched_sizes=True)
        weights = model.bert.embeddings.word_embeddings.weight.data
        new_weights = torch.cat((weights, weights[101:3399]), 0)
        new_emb = nn.Embedding.from_pretrained(new_weights, padding_idx=0, freeze=False)
        model.bert.embeddings.word_embeddings = new_emb


        model.load_state_dict(torch.load(state_dict_dir, weights_only=True))
        model.config.label2id = {
            "Non-Brainrot": 0,
            "Brainrot": 1,
        }
        model.config.id2label = {
            "0": "Non-Brainrot",
            "1": "Brainrot"
        }

        return model

def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=True)
    return tokenizer
    
model = init_model()
tokenizer = init_tokenizer()


app = Flask(__name__)
CORS(app)

@app.route("/getBrainrot", methods=['POST']) # ideally, lets just only POST the API since we don't need to access it from a web browser anyways
def brainrotRequest():
    brainrotInput = request.form['text']
    print(brainrotInput)
    brainrotStatus = brainrot_detection(brainrotInput)
    #brainrotStatus = brainrot_detection(request.form['text']) # uncomment this line to actually use model
    return jsonify({"result": brainrotStatus})


def brainrot_detection(model_input: str) -> dict:

    '''
   Performs brainrot prediction on the given input text

    Args:
        model_input (str): The text conversation

    Returns:
        dict: A dictionary where keys are speaker labels and values are their personality predictions
        '''
    if len(model_input) == 0:
        ret = {
            "Non-Brainrot": float(0),
            "Brainrot": float(0),
        }
        return ret
    else:
        dict_custom = {}
        preprocess_part1 = model_input[:len(model_input)]
        dict1 = tokenizer.encode_plus(preprocess_part1, max_length=64, padding=True, truncation=True)
        dict_custom['input_ids'] = [dict1['input_ids'], dict1['input_ids']]
        dict_custom['token_type_ids'] = [dict1['token_type_ids'], dict1['token_type_ids']]
        dict_custom['attention_mask'] = [dict1['attention_mask'], dict1['attention_mask']]
        outs = model(torch.tensor(dict_custom['input_ids']), token_type_ids=None, attention_mask=torch.tensor(dict_custom['attention_mask']))
        b_logit_pred = outs[0]
        pred_label = torch.sigmoid(b_logit_pred)
        ret = {
            "NonBrainrot": float(pred_label[0][0]),
            "Brainrot": float(pred_label[0][1]),
        }
        return ret

def brainrot_prediction(): #unused
    text_input = ""
    while text_input != "exit":
        text_input = input("Title to evaluate (type exit to exit): ")
        if text_input != "exit":
            prediction = brainrot_detection(text_input)
            print(prediction)
        else:
            break
