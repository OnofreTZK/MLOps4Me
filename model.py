import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import *
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

credit = pd.read_csv('Credit.csv')

print(credit.shape)
print(credit.head())

for col in credit.columns:
    if credit[col].dtype == 'object':
        ''' transforming our values from categories to numbers '''
        credit[col] = credit[col].astype('category').cat.codes

print(credit.shape)
print(credit.head())

forecasters = credit.iloc[:,0:20].values

classes = credit.iloc[:,20].values

print(forecasters)

x_training, x_test, y_training, y_test = train_test_split(forecasters, classes,
                                                          test_size=0.3, random_state=123)
# Model
naive_bayes = GaussianNB()
naive_bayes.fit(x_training, y_training)

forecasts = naive_bayes.predict(x_test)

accuracy = accuracy_score(y_test, forecasts)
print(accuracy)
