# -*- coding: utf-8 -*-
"""
Created on Sun May 21 11:15:48 2017

@author: arunk
"""

import nltk,re,pprint,numpy as np;
from nltk.book import FreqDist;
from nltk import word_tokenize;
from urllib import request;

np.set_printoptions(threshold=np.nan);
url = "http://www.gutenberg.org/files/2554/2554.txt";
response = request.urlopen(url);
raw = response.read().decode("utf8");
print("********** Raw Format *********");
print(len(raw));
print("********** Tokenize ***********");
tokens = word_tokenize(raw);
print(tokens[:10]);
print("********** Text ************");
text = nltk.Text(tokens);
fd = FreqDist(text);
print(fd.max(), " : " ,fd[fd.max()]);
print("******Remove a String ********");
print(raw.find("PART I"));
raw = raw[raw.find("PART I") : raw.rfind("End of Project Gutenberg's Crime")];
#print(raw[:100]);
print(raw.find("PART I"));
print("******** Read URL **********");
url = "http://www.nltk.org/book/ch03.html";
html = request.urlopen(url).read().decode("utf8");
#print(html);
from bs4 import BeautifulSoup
raw = BeautifulSoup(html,"html.parser").get_text();
#print(raw)
tokens = word_tokenize(raw);
print(tokens[0:10])
text = nltk.Text(tokens);
text.concordance("significant");
print("******** Simple Tokenise ***********");
tokens = word_tokenize("hello the weather is very hot heartily so exceedingly and sunny heated today");
text = nltk.Text(tokens);
print(text.similar("monstrous"));
print("********* Regular Expressions");
wordlist = [w for w in nltk.corpus.words.words("en") if w.islower()];
print([w for w in wordlist if re.search("j..t..$",w)][:10]);
#print([w for w in wordlist if re.search("^[abc][wxyz][def]",w)])
#print([w for w in wordlist if re.search("^[abc]+e$",w)]);
word = 'supercalifragilisticexpialidocious';
print(re.findall("(a|e|i|o|u)",word));
print(re.findall(r'^(.*)(oces|ss|ly|ed|ious|ies|ive|es|ces|ment)(sin|sing)$', 'processing'));
from nltk.corpus import gutenberg, nps_chat;
moby = nltk.Text(gutenberg.words('melville-moby_dick.txt'));
print(moby.findall(r"(a ) (<.*>) <man>"));