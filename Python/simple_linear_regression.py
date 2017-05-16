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
dataset = pd.read_csv(r"F:\Programs\Machine Learning\Machine Learning A-Z\Part 2 - Regression\Section 4 - Simple Linear Regression\Simple_Linear_Regression\Salary_Data.csv");
x = dataset.iloc[:,:-1].values;
y = dataset.iloc[:,1].values;

#split train set and test set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0);

#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler();
x_train = sc_x.fit_transform(x_train);                      
x_test = sc_x.transform(x_test);"""
                       
#fit simple linear regression for train data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression();
regressor.fit(x_train,y_train);

#Predict the test data based on train data
y_pred = regressor.predict(x_test);
             
#Visualise the training set results
plt.figure();
plt.scatter(x_train,y_train,color = 'Red');
plt.plot(x_train,regressor.predict(x_train),color = 'Blue');
plt.title("Experiance ve Salary (Training Set)");
plt.xlabel("Years of Experiance");
plt.ylabel("Salary");
plt.show();          
        
#Visualise the test set results
plt.figure();
plt.scatter(x_test,y_test,color = 'Green');
plt.plot(x_train,regressor.predict(x_train),color = 'black');
plt.title("Experiance ve Salary (Test Set)");
plt.xlabel("Years of Experiance");
plt.ylabel("Salary");
plt.show(); 