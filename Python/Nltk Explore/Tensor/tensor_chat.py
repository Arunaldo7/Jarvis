# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 17:25:08 2017

@author: arunk
"""

#dependencies
import numpy as np;
import tensorflow as tf;
import helpers; ##formatting daata and generating random sequence data
#from tensorflow.contrib.seq2seq.python.ops import helper;

tf.reset_default_graph();
sess = tf.InteractiveSession();

#constants for padding and end of sentence
PAD = 0;
EOS = 1;

#number of words
vocab_size = 10;

#character length
input_embedding_size = 20;

#hidden_units
encoder_hidden_units = 20;
decoder_hidden_units = encoder_hidden_units * 2;

#placeholders
encoder_inputs = tf.placeholder(shape = (None,None),dtype = tf.int32,name = 'encoder_inputs');
encoder_inputs_length = tf.placeholder(shape = (None,),dtype = tf.int32,name = 'encoder_inputs_length');
decoder_targets = tf.placeholder(shape = (None,None),dtype = tf.int32,name = 'decoder_targets');

#embeddings
embeddings = tf.Variable(tf.random_uniform(shape = (vocab_size,input_embedding_size),
                                           minval = -1.0,maxval = 1,dtype = tf.float32),
                                            dtype = tf.float32);
encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings,encoder_inputs);

#define encoder
from tensorflow.python.ops.rnn_cell import LSTMCell,LSTMStateTuple;
encoder_cell = LSTMCell(encoder_hidden_units);

((encoder_fw_outputs,encoder_bw_outputs),
 (encoder_fw_final_state,encoder_bw_final_state)) = (
         tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                         cell_bw=encoder_cell,
                                         inputs=encoder_inputs_embedded,
                                         sequence_length=encoder_inputs_length,
                                         dtype=tf.float32,time_major=True));
 
#Concatenates tensors along one dimension.
encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

#letters h and c are commonly used to denote "output value" and "cell state". 
#http://colah.github.io/posts/2015-08-Understanding-LSTMs/ 
#Those tensors represent combined internal state of the cell, and should be passed together. 

encoder_final_state_c = tf.concat(
    (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

encoder_final_state_h = tf.concat(
    (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

#TF Tuple used by LSTM Cells for state_size, zero_state, and output state.
encoder_final_state = LSTMStateTuple(
    c=encoder_final_state_c,
    h=encoder_final_state_h
) 