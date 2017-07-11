# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 02:23:36 2017

@author: arunk
"""
import tensorflow as tf;
import tflearn;
import pickle;
import numpy as np;
from sklearn.model_selection import train_test_split;

#load pickle data
with open('conversation.pkl','rb') as file:
    vec_x,vec_y = pickle.load(file);
    
vec_x = np.array(vec_x,dtype = np.float32);
vec_y = np.array(vec_y,dtype = np.float32);    

x_train,x_test,y_train,y_test = train_test_split(vec_x,vec_y,test_size=0.2,random_state=1);

# reset underlying graph data
tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 500)
net = tflearn.fully_connected(net, 500)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)
net.add(LSTM(output_dim=300,input_shape=x_train.shape[1:],return_sequences=True,init='glorot_normal',
               inner_init='glorot_normal',activation='sigmoid'));
net = tflearn.layers.recurrent.lstm();
# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=500, batch_size=8, show_metric=True)
model.save('model.tflearn')