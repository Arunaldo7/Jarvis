# -*- coding: utf-8 -*-
"""
Created on Mon May 22 17:59:58 2017

@author: arunk
"""

import nltk;
from nltk import word_tokenize;
from nltk.corpus import stopwords;

sample = "And now for something completely different";
text = word_tokenize(sample);
text = [w for w in text if w not in set(stopwords.words("english"))];
print(nltk.pos_tag(text));

print("******** With Homonymns **********");
text = word_tokenize("The girl is pretty isn't she? The girl is beautiful isn't she?");
text = [w for w in text if w not in set(stopwords.words("english"))];
print(nltk.pos_tag(text));
#print(text);
print("********* Similar words **********");
text1 = nltk.Text(word_tokenize("The girl is pretty isn't she? The girl is beautiful isn't she?"));
print(text1.similar("pretty"));
#text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())
#print(text.similar('woman'));
print("******* Tagged Corpora ********");
tagged_token = nltk.tag.str2tuple('fly/NN');
print(tagged_token);
print("******** Dictionary **********");
from operator import itemgetter;
pos = {};
pos["F"] = 4;
pos["B"] = 2;
pos["c"] = 6;
pos["C"] = 9;

print(pos["B"]);
print(list(pos));
pair = ("NP",10);
print(itemgetter(1)(pair));

unique = set([w.lower() for w in pos.keys()]);
print(unique);

for w in unique:
    print(w.upper() + " : ", pos[w.upper()]);

print("******** Default Dictionary ********");    
from collections import defaultdict;
dict1 = defaultdict(int);
dict1["12"]  = 4;   
print(dict1["121"]);
dict1 = defaultdict(lambda : "default");
dict1["12"]  = 4;   
print(dict1["121"]);
print("*********** Unigram Tagger ************");
from nltk.corpus import brown;
brown_tagged_sents = brown.tagged_sents(categories = "news");
brown_sents = brown.sents(categories = "news");
unigram_tagger = nltk.UnigramTagger(brown_tagged_sents);
tagged_sent = unigram_tagger.tag(brown_sents[2007]);
print(tagged_sent);
size = int(len(brown_tagged_sents) * ( 0.9));
train_sets = brown_tagged_sents[ : size];
test_sets = brown_tagged_sents[size : ];
unigram_tagger = nltk.UnigramTagger(train_sets);
print(unigram_tagger.evaluate(test_sets));
print("********* Bigram Tagger *********");
bigram_tagger = nltk.BigramTagger(train_sets);
print(bigram_tagger.evaluate(test_sets));
print("********* Combine Taggers ***********");
t0 = nltk.DefaultTagger("NN");
t1 = nltk.UnigramTagger(train_sets , backoff = t0);
t2 = nltk.BigramTagger(train_sets ,cutoff = 0 , backoff = t1);
print(t2.evaluate(test_sets));

print("******* Storing Taggers **********");
from pickle import dump,load;
output = open("t2.pkl","wb");
dump(t2,output,-1);
output.close();

input1 = open("t2.pkl","rb");
tagger = load(input1);
input1.close();

print(tagger.tag(word_tokenize( """The board's action shows what free enterprise
                               is up against in our complex maze of regulatory laws .""")));
                               