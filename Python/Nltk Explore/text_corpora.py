# -*- coding: utf-8 -*-
"""
Created on Sat May 20 15:07:04 2017

@author: arunk
"""
import nltk;
from nltk.corpus import gutenberg;
from nltk.corpus import webtext;
from nltk.corpus import nps_chat;
from nltk.corpus import brown;
import numpy as np;

np.set_printoptions(threshold=np.nan);
#print(nltk.corpus.gutenberg.fileids());
emma = nltk.Text(nltk.corpus.gutenberg.words("austen-emma.txt"));
#emma.concordance("surprize");
print("*********Books**********");
#for file_id in gutenberg.fileids():
#    text = nltk.Text(gutenberg.words(file_id));
#    print(file_id , " has ", len(text) , " words.");
    
print("*********Web Text**********");
for file_id_web in webtext.fileids():
    print(file_id_web , len(set(nltk.Text(webtext.words(file_id_web)))));

print("*********Chat**********");
chatroom = nps_chat.posts("10-19-20s_706posts.xml");  \
print(chatroom[705]);
  
print("*********Brown**********");
#for categ in brown.categories():    
 #   print(categ, brown.words(categories = categ));
    
#print(brown.sents(categories = brown.categories()));    

genre_word = [(genre,word)
              for genre in ["news","romance"]
              for word in brown.words(categories = genre)];
print(genre_word[:4]);
print(genre_word[-4:]);

cfd = nltk.ConditionalFreqDist(genre_word);
print(cfd["romance"].most_common(10));

sent = ['In', 'the', 'beginning', 'God', 'created', 'the', 'heaven',
        'and', 'the', 'earth', '.'];
print(list(nltk.bigrams(sent)));

print("*********Genesis**********");
def generate_model(cfd,word,num=15):
    for i in range(num):
        print(word, end=" ");
        word = cfd[word].max();

text = nltk.corpus.genesis.words("english-kjv.txt");
bigrams = nltk.bigrams(text);
#print(bigrams);
cfd = nltk.ConditionalFreqDist(bigrams);
print(cfd["creature"].max());
generate_model(cfd,"living");

name_bigram = [(name_files,name_end[-1])
                for name_files in nltk.corpus.names.fileids()
                for name_end in nltk.corpus.names.words(name_files)];
#nltk.ConditionalFreqDist(name_bigram).plot();

names = nltk.corpus.names;
males = names.words(list(names.fileids())[0]);
females = names.words(list(names.fileids())[1]);

print([w for w in males if w in females]);                    

print("********Word Net********");
from nltk.corpus import wordnet as wn
print([lemma_name.lemma_names() for lemma_name in wn.synsets("car")]);
