# -*- coding: utf-8 -*-
"""
Created on Tue May 16 22:06:56 2017

@author: arunk
"""

from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np

data = pd.read_csv(r"F:\Programs\Machine Learning\Programs\Python\Nlp\machine_learning_examples-master\nlp_class\spambase.data.csv");
                  
np.random.shuffle(data);

x = data[:,48];
y = data[:,-1];
                  