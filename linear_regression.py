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

#Check for missing values in dataset 
house_data.isnull().sum()

#Heat Map 
correlation_house = house_data.corr().round(2)
sns.heatmap(data=correlation_house, annot=True)

#Distribution for Medv (Median value of owner-occupied homes in $1000s)
sns.set(rc={'figure.figsize':(12,9)})
sns.distplot(house_data['MEDV'], bins=30)
plt.show()
