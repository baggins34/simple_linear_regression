"""
Simple Linear Regression
by
?brahim Halil Bayat, PhD
?stanbul Technical University
?stanbul, Turkey

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl

df = pd.read_csv("FuelConsumptionCo2.csv")
print(df.describe())

cdf = df[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB", "CO2EMISSIONS"]]
print(cdf.head())

cdf.hist()
plt.show()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color = 'blue')
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emissions")
plt.title("Engine Size vs CO2 Emission")
plt.show()

plt.scatter(cdf["FUELCONSUMPTION_COMB"], cdf["CO2EMISSIONS"], color = 'blue')
plt.xlabel("Fuel Consumption Comb")
plt.ylabel("CO2 emissions")
plt.title("Fuel Consumption vs CO2 Emissions")
plt.show()

plt.scatter(cdf["CYLINDERS"], cdf["CO2EMISSIONS"], color = 'blue')
plt.title("Cylinders vs CO2 Emissions")
plt.xlabel("Cylinders")
plt.ylabel("CO2 Emissions")
plt.show()


msk = np.random.rand(len(df)) < 0.8
# Choose a random data in the size of df but less than 80 percent of the data
print(msk)
train = cdf[msk]
test = cdf[~msk]

print(train.shape)
print(test.shape)
print(cdf.shape)

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color = 'purple')
plt.xlabel("Engine Size")
plt.ylabel("Co2 Emissions")
plt.title("Engine Size and CO2 Emission of Train Data Set")
plt.show()

plt.scatter(test.ENGINESIZE, test.CO2EMISSIONS, color ='green')
plt.xlabel("Engine Size")
plt.ylabel("Co2 Emissions")
plt.title("Engine Size and CO2 Emission of Test Data Set")
plt.show()

"""
Time to train the data 
"""
from sklearn import linear_model as lm
obj_regression = lm.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
obj_regression.fit (train_x, train_y)
print("The Coefficients: ", obj_regression.coef_)
print("The Intercept: ", obj_regression.intercept_)

plt.scatter(train[['ENGINESIZE']], train[['CO2EMISSIONS']], color = 'pink')
plt.plot(train_x, obj_regression.coef_[0][0]*train_x + obj_regression.intercept_[0], '-r')
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emissions")
plt.show()

"""
R2 Score
"""

from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_predicted = obj_regression.predict(test_x)

print("Mean Absolute Error:  %.2f" % np.mean(np.absolute(test_y_predicted - test_y)))
print("RMSE:  %.3f   " % np.mean(test_y_predicted - test_y)**2 )
print("R2-Score:  %.4f  " % r2_score(test_y_predicted, test_y))