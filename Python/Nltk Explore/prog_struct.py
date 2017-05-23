# -*- coding: utf-8 -*-
"""
Created on Mon May 22 14:56:09 2017

@author: arunk
"""

nested = [[]] * 3;
nested[1] = ['Monty'];
nested[1].append('Python');

print(nested);

empty = [];
nested = [empty, empty, empty];
print(nested);
nested[1].append ("Py");
print(nested);

print("********* zip and enumerate *********");
words = ['I', 'turned', 'off', 'the', 'spectroroute'];
tags = ['noun', 'verb', 'prep', 'det'];
print(list(enumerate(zip(words,tags))));

