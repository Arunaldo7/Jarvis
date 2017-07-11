# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 22:28:05 2017

@author: arunk
"""

import tensorflow as tf;
from tensorflow.examples.tutorials.mnist import input_data;

#read dataset
mnist = input_data.read_data_sets(r"F:\Programs\Machine Learning\Programs\Python\Nltk Explore\Tensorflow\data",
                                  one_hot = True);

#define number of nodes in each hidden neural layers
n_nodes_hl1 = 500;
n_nodes_hl2 = 700;
n_nodes_hl3 = 900;

n_classes = 10;
batch_size = 100;

#height * weight(for image pixels)
x = tf.placeholder('float',[None,784]);
y = tf.placeholder('float');

#weights = tf.Variable(tf.random_normal([784,500]));
#print(weights)

def neural_network_model(data):
    hidden_layer_1 = {'weights' : tf.Variable(tf.random_normal([784,n_nodes_hl1])),
                      'biases' : tf.Variable(tf.random_normal([n_nodes_hl1]))};
    hidden_layer_2 = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
                      'biases' : tf.Variable(tf.random_normal([n_nodes_hl2]))};
    hidden_layer_3 = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
                      'biases' : tf.Variable(tf.random_normal([n_nodes_hl3]))};
    output_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
                      'biases' : tf.Variable(tf.random_normal([n_classes]))};
              
    l1 = tf.add(tf.matmul(data,hidden_layer_1['weights']),hidden_layer_1['biases']);
    l1 = tf.nn.relu(l1);
    
    l2 = tf.add(tf.matmul(l1,hidden_layer_2['weights']),hidden_layer_2['biases']);
    l2 = tf.nn.relu(l2);
    
    l3 = tf.add(tf.matmul(l2,hidden_layer_3['weights']),hidden_layer_3['biases']);
    l3 = tf.nn.relu(l3);
    
    output = tf.add(tf.matmul(l3,output_layer['weights']),output_layer['biases']);
    
    return output;

def train_neural_network():
    print(x)
    prediction = neural_network_model(x);
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y));
    optimizer = tf.train.AdamOptimizer().minimize(cost);
    
    #cycles of feed forward + backpropagation
    n_epochs = 10;
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer());
    
        
        for epoch in range(n_epochs):
            epoch_loss = 0;
            for _ in range (int(mnist.train.num_examples/batch_size)):
                epoch_x,epoch_y = mnist.train.next_batch(batch_size);
                _,c = sess.run([optimizer,cost], feed_dict = {x:epoch_x,y:epoch_y});
                epoch_loss += c;
            print('Epoch : ' , epoch, ' completed out of ' , n_epochs , ' loss : ',epoch_loss);
        
        correct = tf.equal(tf.arg_max(prediction,1),tf.arg_max(y,1));
        accuracy = tf.reduce_mean(tf.cast(correct,'float'));
        
        print('Accuracy : ', accuracy.eval({x:mnist.test.images,y:mnist.test.labels}));
            
train_neural_network();        