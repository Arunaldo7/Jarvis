# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 00:30:26 2017

@author: arunk
"""

import os;
import json;
import nltk;
import gensim;
import numpy as np;
import pickle;
from gensim import corpora,models,similarities;


model = gensim.models.Word2Vec.load('word2vec.bin');

with open('conversations.corpus.json','r') as file:
    data = json.load(file);
conv = data['conversations'];
    
x = [];
y = [];

#process data and form question and answer model
for i in range(len(conv)):
    for j in range(len(conv[i])):
        if j < len(conv[i]) - 1:
            x.append(conv[i][j]);
            y.append(conv[i][j+1]);
            
#tokenize
tok_x = [];
tok_y = [];
for i in range(len(x)):
    tok_x.append(nltk.word_tokenize(x[i].lower()));
    tok_y.append(nltk.word_tokenize(y[i].lower()));
    
sent_end = np.ones((300,),dtype=np.float32);

#create word vector
vec_x = [];
for sent in tok_x:
    sent_vec = [model[w] for w in sent if w in model.vocab];
    vec_x.append(sent_vec);

vec_y = [];
for sent in tok_y:
    sent_vec = [model[w] for w in sent if w in model.vocab];
    vec_y.append(sent_vec);
    
    
for tok_sent in vec_x:
    tok_sent[14:] = [];
    tok_sent.append(sent_end);

for tok_sent in vec_x:
    if len(tok_sent) < 15:
        for i in range(15 - len(tok_sent)):
            tok_sent.append(sent_end);
            
for tok_sent in vec_y:
    tok_sent[14:] = [];
    tok_sent.append(sent_end);

for tok_sent in vec_y:
    if len(tok_sent) < 15:
        for i in range(15 - len(tok_sent)):
            tok_sent.append(sent_end);
            
with open('conversation.pkl','wb') as file:
    pickle.dump([vec_x,vec_y],file);