# -*- coding: utf-8 -*-
"""
Created on Fri May 19 16:20:20 2017

@author: arunk
"""

from nltk.book import *;
from nltk import bigrams;
import nltk;

def lexical_diversity(text1):
    return len(set(text1))/len(text1);

def percentage(count,total):
    return 100 * count / total;

print(lexical_diversity(text2));
print(percentage(3,7));

sent1 = ["Hi","welcome"];
sent2 = ["Hello","there"];
sent1.append("world");

print(sent1 + sent2);

print(text5[16715:16717]);

print(text5[16715]);

sent = ['word1', 'word2', 'word3', 'word4', 'word5','word6', 'word7', 'word8', 'word9', 'word10'];
sent1 = sent.copy();
sent[0] = "First";
sent[5] = "Fifth";

sent[2:10] = ["Arun"];

print("sent " , sent);

str1 = "abc"
print("string multiply " , (str1 * 2));

print(sent1[-2:-1]);

fdist1 = FreqDist(text1);
print("frequency ", fdist1);
print(fdist1.most_common(3));
print("the comes ", fdist1["the"] ," times" );
#fdist1.plot(50,cumulative=True);
print(len(fdist1.hapaxes()));

v = set(text4);
long_words = (w for w in v if (len(w)) > 15)
print(sorted(long_words));

freqdist5 = FreqDist(text5);
freq_long_words = (w for w in freqdist5 if(len(w) > 7) and freqdist5[w] > 7);
print(sorted(freq_long_words));

print(list(bigrams(['more', 'is', 'said', 'than', 'done'])));
print("before collocation");
text4.collocations();

freqDist3 = FreqDist(len(w) for w in text1);
#print([w for w in freqDist3.elements()]);
print(len(list(freqDist3.most_common())));

print([w.upper() for w in sent7 if len(w) == 4]);

#nltk.chat.chatbots();