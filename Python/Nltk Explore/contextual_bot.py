# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 10:04:20 2017

@author: arunk
"""
#Things for NLP
import nltk;
from nltk.corpus import stopwords;
from sklearn.feature_extraction.text import CountVectorizer;
from nltk import LancasterStemmer;
stemmer = LancasterStemmer();

#Things for TensorFlow
import numpy as np;
import tensorflow as tf;
import tflearn;
import random;

#Import Intents json file
import json;
with open('intents.json','r') as json_data:
    intents = json.load(json_data);


corpus = []
classes = []
words = []
ignore_words = ['?']
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        review = pattern.lower();
        review = review.split();
        review = [stemmer.stem(w.lower()) for w in review if w not in set(ignore_words)];
        review = " ".join(review);
        corpus.append(review)
        classes.append(intent['tag']);
        w = nltk.word_tokenize(pattern)
        # add to our words list
        words.extend(w);

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)));
# create our training data
train= []
output = []

#Vectorize
cv_x = CountVectorizer();
cv_x = cv_x.fit(corpus);
train_x=[];
#train_x = [list(bag_words_list) for bag_words_list in cv_x.transform(corpus).toarray()];
train_x = cv_x.transform(corpus).toarray();

#Encoding the dependent variables               
#from sklearn.preprocessing import LabelEncoder
#labelLencoder_y = LabelEncoder();
#training_y = labelLencoder_y.fit_transform(classes);
cv_y = CountVectorizer();
train_y=[];
#train_y = [list(bag_words_list) for bag_words_list in cv_y.fit_transform(classes).toarray()];
train_y = cv_y.fit_transform(classes).toarray();

# reset underlying graph data
tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 500)
net = tflearn.fully_connected(net, 500)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=500, batch_size=8, show_metric=True)
model.save('model.tflearn')

print(len(train_y[0]) , ":",len(train_x[0]))
# save all of our data structures
import pickle
pickle.dump( {'words':words,'classes':sorted(list(set(classes))),'vectorizer':cv_x, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )
#print(train_x[0])