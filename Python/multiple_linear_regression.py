# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 13:16:15 2017

@author: arunk
"""

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
dataset = pd.read_csv(r"F:\Programs\Machine Learning\Machine Learning A-Z\Part 2 - Regression\Section 5 - Multiple Linear Regression\Multiple_Linear_Regression\50_Startups.csv");
x = dataset.iloc[:,:-1].values;
y = dataset.iloc[:,4].values;

#Encoding the independent variables               
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelLencoder_x = LabelEncoder();
x[:,3] = labelLencoder_x.fit_transform(x[:,3]);

one_hot_encoder_x = OneHotEncoder(categorical_features = [3]);
x = one_hot_encoder_x.fit_transform(x).toarray();                                 
                                     
#Avoiding Dummy variable trap
x = x[:,1:];
                                   
#split train set and test set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0);

#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler();
x_train = sc_x.fit_transform(x_train);                      
x_test = sc_x.transform(x_test);"""

#Fitting multiple linear regression in training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(); 
regressor.fit(x_train,y_train);

y_pred = regressor.predict(x_test);                                         
                          
#Bulding a backward elimination model
import statsmodels.formula.api as sm
x = np.append(arr = np.ones(shape = (np.shape(x)[0],1)), values = x,axis = 1) ;

x_opt = x[:,[0,1,2,3,4,5]];
regressor_OLS = sm.OLS(endog = y,exog = x_opt).fit();
print(regressor_OLS.summary());                      
     
x_opt = x[:,[0,1,3,4,5]];
regressor_OLS = sm.OLS(endog = y,exog = x_opt).fit();
print(regressor_OLS.summary());       
     
x_opt = x[:,[0,3,4,5]];
regressor_OLS = sm.OLS(endog = y,exog = x_opt).fit();
print(regressor_OLS.summary());          
     
x_opt = x[:,[0,3,5]];
regressor_OLS = sm.OLS(endog = y,exog = x_opt).fit();
print(regressor_OLS.summary()); 

x_opt = x[:,[0,3]];
regressor_OLS = sm.OLS(endog = y,exog = x_opt).fit();
print(regressor_OLS.summary());             