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
dataset = pd.read_csv(r"F:\Programs\Machine Learning\Machine Learning A-Z\Part 2 - Regression\Section 6 - Polynomial Regression\Polynomial_Regression\Position_Salaries.csv");
x = dataset.iloc[:,1:2].values;
y = dataset.iloc[:,2].values;
                        
#Fitting simple linear regression in dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression(); 
lin_reg.fit(x,y);

#Fitting polynomial regression in dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4);
x_poly = poly_reg.fit_transform(x);
lin_reg2 = LinearRegression();
lin_reg2.fit(x_poly,y);

#Visualise Linear regression results
plt.figure();
plt.title("Salary vs Experiance");
plt.xlabel("Experiance");
plt.ylabel("Salary");          
plt.scatter(x,y);
plt.plot(x,lin_reg.predict(x));   
plt.show();

#Visualise Polynomial regression results
x_grid = np.arange(min(x),max(x),0.1);
x_grid = x_grid.reshape(len(x_grid),1);
plt.figure();
plt.title("Salary vs Experiance");
plt.xlabel("Experiance");
plt.ylabel("Salary");          
plt.scatter(x,y);
plt.plot(x_grid,lin_reg2.predict(poly_reg.fit_transform(x_grid)));                                  
plt.show();        
        
print(lin_reg.predict(6.5));
print(lin_reg2.predict(poly_reg.fit_transform(6.5)));             