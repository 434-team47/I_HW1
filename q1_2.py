import pandas as pd
import numpy as np
import math
import csv
import sys
import matplotlib.pyplot as plt

class CsCal:
    def __init__(self, X, y):
        self.data = X
        self.target = y
        self.weight = ((self.data.T * self.data).I) * (self.data.T * self.target)

    # Compute the optimal weight vector (XT X)−1 XT Y .
    def weight_vector(self):
        print(self.weight)

    #Compute average squared error
    def ase(self):
        '''returns sum of squared errors (model vs actual)'''
        ase = 0
        predict = self.data * self.weight
        for j in range(len(self.data)):
            ase = ase+ ((self.target[j] - predict[j]) ** 2)

        ans = ase/len(self.data)
        print(ans[0,0])


#command line argument variables
trainData = sys.argv[1]
testData = sys.argv[2]

#Add the Labels
house_train = pd.read_csv(trainData, names =['CRIM','ZIN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PIRATIO','B','LSTAT','MEDV'] )
house_test = pd.read_csv(testData, names =['CRIM','ZIN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PIRATIO','B','LSTAT','MEDV'] )

# Selecting first 13 columns of data frame with all rows
x_train = house_train.iloc[:,0:-1].values
x_test = house_test.iloc[:,0:-1].values

# Adding dummy variables
x_train = np.insert(x_train, 0, 1, axis=1)
x_test = np.insert(x_test, 0, 1, axis=1)

# Selecting last column of data frame
y_train = house_train.iloc[:,-1].values
y_train = np.matrix(y_train, dtype=float)
y_train =y_train.T

y_test = house_test.iloc[:,-1].values
y_test = np.matrix(y_test, dtype=float)
y_test =y_test.T

# return matrix from array
x_train = np.matrix(x_train)
y_train = np.matrix(y_train)

x_test = np.matrix(x_test)
y_test = np.matrix(y_test)

output = CsCal(x_train, y_train)
output2 = CsCal(x_test, y_test)

print("The learned weight vector:")
output.weight_vector()
print("\nASE over the training data")
output.ase()
print("ASE over the Test data")
output2.ase()
