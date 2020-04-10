#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
%matplotlib inline

#import the dataset and Extract the Dependent and Independant variables
house_data = pd.read_csv('housing_train.csv', names =['CRIM','ZIN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PIRATIO','B','LSTAT','MEDV'] )

x_train = house_data.iloc[:, :13].values
y_train = house_data.iloc[:, 13].values

house_data.head()


#Heatmap plot
sns.heatmap(house_data.corr())
