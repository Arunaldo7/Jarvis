# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 22:28:05 2017

@author: arunk
"""
import json;
import tensorflow as tf;
from nltk import word_tokenize;
from nltk import WordNetLemmatizer,LancasterStemmer;
from sklearn.feature_extraction.text import CountVectorizer;
import numpy as np;

#read dataset
intents = [];
with open(r"F:\Programs\Machine Learning\Programs\Python\Nltk Explore\Tensorflow\data\intents.json","r") as json_data:
    intents = json.load(json_data);

#Pre Process Dataset
cleaned_sent = [];
bag_of_words = [];
tags = [];
ignore_words = ['?'];
ls = LancasterStemmer();
wn = WordNetLemmatizer();

for intent in intents['intents']: 
    for pattern in intent['patterns']:
        pattern = word_tokenize(pattern);
        cleaned_sent = [ls.stem(wn.lemmatize(word)) for word in pattern if word not in ignore_words];
        cleaned_sent = " ".join(cleaned_sent);
        bag_of_words.append(cleaned_sent);
        tags.append(intent['tag'])

#Create parse matrix using bag of words for train data
cv_x = CountVectorizer();
cv_x.fit(bag_of_words);
x_train = cv_x.transform(bag_of_words).toarray();
#x_train = [list(bag_words_list) for bag_words_list in cv_x.fit_transform(bag_of_words).toarray()];

cv_y = CountVectorizer();
y_train = cv_y.fit_transform(tags).toarray();        
#y_train = [list(tag_list) for tag_list in cv_y.fit_transform(tags).toarray()];

#define number of nodes in each hidden neural layers
n_nodes_hl1 = 500;
n_nodes_hl2 = 700;
n_nodes_hl3 = 1000;

n_classes = np.shape(y_train)[1];
batch_size = 1;

input_size = np.shape(x_train)[1];

#height * weight(for image pixels)
x = tf.placeholder('float');
y = tf.placeholder('float');
x_test = [];
statement_test = [];
#weights = tf.Variable(tf.random_normal([784,500]));
#print(weights)
#

saver = tf.train.Saver();
save_file = './model.ckpt';
result = [];
def neural_network_model(data):
    hidden_layer_1 = {'weights' : tf.Variable(tf.random_normal([input_size,n_nodes_hl1])),
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
            for i in range (len(tags)):
                _ = "";
                epoch_x = x_train[i];
                epoch_x = np.reshape(epoch_x,(1,input_size));
                epoch_x = epoch_x.astype(np.float32);
                epoch_y = y_train[i];
                _,c = sess.run([optimizer,cost], feed_dict = {x:epoch_x,y:epoch_y});
                epoch_loss += c;
            print('Epoch : ' , epoch, ' completed out of ' , n_epochs , ' loss : ',epoch_loss);
        saver.save(sess, save_file);
#        correct = tf.equal(tf.arg_max(prediction,1),tf.arg_max(y,1));
#        accuracy = tf.reduce_mean(tf.cast(correct,'float'));
        
#        print('Accuracy : ', accuracy.eval({x:mnist.test.images,y:mnist.test.labels}));
 

#Predict the input  
def predict_intent(statement):
    prediction = neural_network_model(x);
    print(statement)
    statement_test = [ls.stem(wn.lemmatize(word)) for word in word_tokenize(statement) if word not in ignore_words];
    statement_test = " ".join(statement_test);
    statement_corpus = [];
    statement_corpus.append(statement_test);
    x_test = cv_x.transform(statement_corpus).toarray();
    x_test = np.reshape(x_test,(1,input_size));
    x_test = x_test.astype(np.float32);
    
#    with tf.Session() as sess:
#        sess.run(tf.global_variables_initializer());
#        saver.restore(sess, 'model.ckpt');
#        result = (sess.run(tf.arg_max(prediction.eval(feed_dict = {x:x_test}),1)));
#        print(cv_y.decode(result));
#        print(result[0]);
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer());
        saver.restore(sess, 'model.ckpt');
        output = sess.run(prediction,feed_dict={x: x_test});
    print(output);
    print(cv_y.decode(output));
       
train_neural_network();   
predict_intent("What are your hours today");     