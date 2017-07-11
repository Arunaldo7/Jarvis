# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 00:30:26 2017

@author: arunk
"""

import os;
#os.environ['THEANO_FLAGS'] = "device=gpu0" 
import json;
import nltk;
import gensim;
import numpy as np;
import pickle;
from keras.models import Sequential;
from keras.layers.recurrent import LSTM,SimpleRNN;
from sklearn.model_selection import train_test_split;
import theano;
#theano.config.optimizer = 'None';
#theano.config.device = 'gpu'
theano.config.floatX = 'float32'
#load pickle data
with open('conversation.pkl','rb') as file:
    vec_x,vec_y = pickle.load(file);
    
vec_x = np.array(vec_x,dtype = np.float32);
vec_y = np.array(vec_y,dtype = np.float32);    

x_train,x_test,y_train,y_test = train_test_split(vec_x,vec_y,test_size=0.2,random_state=1);

model = Sequential();
model.add(LSTM(output_dim=300,input_shape=x_train.shape[1:],return_sequences=True,init='glorot_normal',
               inner_init='glorot_normal',activation='sigmoid'));
model.add(LSTM(output_dim=300,input_shape=x_train.shape[1:],return_sequences=True,init='glorot_normal',
               inner_init='glorot_normal',activation='sigmoid'));
model.add(LSTM(output_dim=300,input_shape=x_train.shape[1:],return_sequences=True,init='glorot_normal',
               inner_init='glorot_normal',activation='sigmoid'));
model.add(LSTM(output_dim=300,input_shape=x_train.shape[1:],return_sequences=True,init='glorot_normal',
               inner_init='glorot_normal',activation='sigmoid'));
model.compile(loss='cosine_proximity',optimizer='adam',metrics=['accuracy']);

model.fit(x_train,y_train,epochs=500,batch_size=8,validation_data=(x_test,y_test));
model.save('LTSM500.h5');

model.fit(x_train,y_train,epochs=500,validation_data=(x_test,y_test));
model.save('LTSM1000.h5');

model.fit(x_train,y_train,epochs=500,validation_data=(x_test,y_test));
model.save('LTSM1500.h5');

model.fit(x_train,y_train,epochs=500,validation_data=(x_test,y_test));
model.save('LTSM2000.h5');

model.fit(x_train,y_train,epochs=500,validation_data=(x_test,y_test));
model.save('LTSM2500.h5');

model.fit(x_train,y_train,epochs=500,validation_data=(x_test,y_test));
model.save('LTSM3000.h5');