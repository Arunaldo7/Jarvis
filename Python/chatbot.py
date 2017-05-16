# -*- coding: utf-8 -*-
"""
Created on Sun May 14 13:00:21 2017

@author: arunk
"""

import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;

np.set_printoptions(threshold=np.nan);
#importing the dataset
dataset = pd.read_csv(r"F:\Programs\Machine Learning\Machine Learning A-Z\Part 7 - Natural Language Processing\Section 36 - Natural Language Processing\Natural_Language_Processing\chatbot.tsv",delimiter = "\t", quoting = 3);

testset = pd.read_csv(r"F:\Programs\Machine Learning\Machine Learning A-Z\Part 7 - Natural Language Processing\Section 36 - Natural Language Processing\Natural_Language_Processing\chatbot_test.tsv",delimiter = "\t", quoting = 3);                     
#cleaning the texts
import re;
import nltk;
nltk.download("stopwords")

from nltk.corpus import stopwords;
from nltk.stem.porter import PorterStemmer

corpus = [];
#retain only alphabets
for i in range (0 , np.shape(dataset)[0]) : 
    review = re.sub("[^a-z A-Z]", " ", dataset["Queries"][i]);
    review = review.lower();
    review = review.split();
    ps = PorterStemmer();
    review = [ps.stem(word) for word in review if word not in set(stopwords.words("english"))];         
    review = " ".join(review);    
    corpus.append(review);

cleanedData = dataset.copy();
cleanedData["Queries"] = corpus;
           
#Creating bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer();
x = cv.fit_transform(corpus).toarray();           
y = dataset.iloc[:,1].values;                 

#Encoding the dependent variables               
from sklearn.preprocessing import LabelEncoder
labelLencoder_y = LabelEncoder();
y = labelLencoder_y.fit_transform(y);
                
#splitting into train and test set                
#from sklearn.cross_validation import train_test_split
x_train = x[:-4,];
x_test = x[17:21,];

y_train = y[:-4];         
y_test = y[17:21];
          
                                                
#feature scaling
#from sklearn.preprocessing import StandardScaler
#sc_x = StandardScaler();
#x_train = sc_x.fit_transform(x_train);        
#x_test = sc_x.fit_transform(x_test);                                           

                            
#fitting logistic regression on training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train,y_train);


                     
#predict the results
y_pred = classifier.predict(x_test);                      
                           
#Encoding the dependent variables               
#from sklearn.preprocessing import LabelEncoder
#labelLencoder_y_test = LabelEncoder();
#y_test = labelLencoder_y_test.fit_transform(y_test);
                                           
#making the confusion matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test,y_pred);                     