#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd 
import pylab as pl
import numpy as np
%matplotlib inline

import wget 
url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
file_name=wget.filename_from_url(url)
print(file_name)

# show the dataset
df=pd.read_csv(url)
df.head()

# extract some items from datasets
cdf=df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

# plot Emission values with respects to Engine Size
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='green')
plt.xlabel("Engine Size")
plt.yalbel("CO2 Emissions")
plt.show()

# create train and test dataset
msk=np.random.rand(len(df))<0.8
train=cdf[msk]
test=cdf[~msk]
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color="green")
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emission")
plt.show()

# create multiple linear regression model
from sklearn import linear_model
regr= linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[["CO2EMISSIONS"]])
regr.fit(x,y)
print("Coefficients:", regr.coef_)

"""OLS is a method for estimating the unknown parameters in a linear regression model. 
   OLS chooses the parameters of a linear function of a set of explanatory variables by
   minimizing the sum of the squares of the differences between the target dependent 
   variable and those predicted by the linear function. In other words, 
   it tries to minimizes the sum of squared errors (SSE) or mean squared error (MSE) between
   the target variable (y) and our predicted output ( 𝑦̂  ) over all samples in the dataset.
"""

y_hat=regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x=np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y=np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares: %.2f" % np.mean((y_hat-y)**2))
print("Variance score: %.2f" % regr.score(x,y))

# mulitiple linear regression model
regr= linear_model.LinearRegression()
x= np.asanyarray(train[['ENGINESIZE', 'CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y= np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x,y)
print('Coefficients: ', regr.coef_)
y_= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
x= np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y= np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares: %.2f" % np.mean((y_-y)**2))
print('Variance score: %.2f' % regr.score(x,y))





















