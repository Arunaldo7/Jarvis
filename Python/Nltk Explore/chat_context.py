# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 13:22:09 2017

@author: arunk
"""
#imports
import nltk;
from nltk import LancasterStemmer;
import pickle;
import json;    
import tensorflow as tf;
import tflearn;
import numpy as np;
import random;
ERROR_THRESHOLD = 0.25;

class chat_context:
    ignore_words = ["?"];
    test_x = [];
    words = [];
    classes = [];
    train_x = [];
    train_y = [];
    vectorizer = None ;
    model = None;
    intents = [];
    text_x = [];
    context = {};
    #function to find bag of words
    def bow(self,sentence,showDetails=False):
        corpus = [];
        ls = LancasterStemmer();
        sentence = [ls.stem(word.lower()) for word in nltk.word_tokenize(sentence) 
                                                    if(word not in self.ignore_words)];
        sentence = " ".join(sentence);
#        print(self.vectorizer);
        corpus.append(sentence);
        self.test_x = [list(bag_words_list) 
                            for bag_words_list in 
                                (self.vectorizer).transform(corpus).toarray()];
#        print(" ".join(str(self.test_x[0])))
        return (self.text_x)
    #restore all of our data structures
    def restoreValues(self):    
        data = pickle.load( open( "training_data", "rb" ) )
        self.words = data['words']
        self.classes = data['classes']
        self.train_x = data['train_x']
        self.train_y = data['train_y']
        self.vectorizer = data['vectorizer']
#        print(self.classes)
        # import our chat-bot intents file
        with open('intents.json') as json_data:
            self.intents = json.load(json_data)
    
    #load our saved model
    def getTfModel(self):
        # reset underlying graph data
        tf.reset_default_graph()
#        # Build neural network
        net = tflearn.input_data(shape=[None, len(self.train_x[0])])
        net = tflearn.fully_connected(net, 500)
        net = tflearn.fully_connected(net, 500)
        net = tflearn.fully_connected(net, len(self.train_y[0]), activation='softmax')
        net = tflearn.regression(net)
#        
#        # Define model and setup tensorboard
        self.model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
        self.model.load('./model.tflearn')  
        
    #classify
    def classify(self,sentence):
        # generate probabilities from the model
        self.bow(sentence);
#        print(self.test_x[0]);
#        input_query = " ".join(str(self.test_x[0]));
        results = self.model.predict([self.test_x[0]])[0]
        print(results)
        prob = 0;
        for r in results:
            prob = prob + r;
        print(prob)
        # filter out predictions below a threshold
        results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
        print(results)
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        
        for r in results:
            print(r);
            return_list.append((self.classes[r[0]], r[1]))
        # return tuple of intent and probability
        return return_list;
    
    #response
    def response(self,sentence, userID='123', show_details=False):
        results = self.classify(sentence)
        print(results[0])
        # if we have a classification then find the matching intent tag
        if results:
            # loop as long as there are matches to process
            while results:
                for i in self.intents['intents']:
                    # find a tag matching the first result
                    if i['tag'] == results[0][0]:
                        return print(random.choice(i['responses']));
    
        results.pop(0);
            
chat = chat_context();
chat.restoreValues();
chat.getTfModel();
sentence = "fidfkjhgdkufhgudfs";
#chat.bow(sentence);
#chat.classify(sentence);
chat.response(sentence);
#print(chat.bow("is your shop open today?"));  