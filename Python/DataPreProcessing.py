# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 01:52:15 2017

@author: arunk
"""

import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;

np.set_printoptions(threshold=np.nan);
#importing the dataset
dataset = pd.read_csv(r"F:\Programs\Machine Learning\Machine Learning A-Z\Part 1 - Data Preprocessing\Data.csv");
x = dataset.iloc[:,:-1].values;
y = dataset.iloc[:,3].values;

#taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = np.nan,strategy = "mean", axis = 0);
x[:,1:3] = imputer.fit_transform(x[:, 1:3]);
 
print(x);
     
#encoding cateoprial data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder();
x[:,0] = labelencoder_x.fit_transform(x[:,0]);      

onehotencoder = OneHotEncoder(categorical_features=[0]);
x = onehotencoder.fit_transform(x[:,:]).toarray();         
                               
labelencoder_y = LabelEncoder();
y = labelencoder_y.fit_transform(y);                                   
                                
#split train set and test set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0);

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler();
x_train = sc_x.fit_transform(x_train);                      
x_test = sc_x.transform(x_test);
                                             