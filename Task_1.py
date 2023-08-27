import numpy
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
iris=pd.read_csv(r'C:\Users\HP\Documents\iNeuBytes\irisflowerdataset.csv')
print(iris.shape)
X=iris['sepal_length']
y=iris['petal_length']
X_train, X_test, y_train, y_test = train_test_split(
  X,y , random_state=104,test_size=0.25, shuffle=True)
 
print('X_train : ')
print(X_train.head())
print(X_train.shape)
 
print('')
print('X_test : ')
print(X_test.head())
print(X_test.shape)
 
print('')
print('y_train : ')
print(y_train.head())
print(y_train.shape)
 
print('')
print('y_test : ')
print(y_test.head())
print(y_test.shape)
