import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

#The csv's columns names
names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']

df = pd.read_csv('Datasets/data.cleveland.csv', sep=';', names=names)

df_healthy = df[df.iloc[:, 13] == 1]
df_sick = df[df.iloc[:, 13] == 2]

X = df_healthy.iloc[:, :-1].values
y = df_healthy.iloc[:, -1].values

plt.scatter(X[:, 0], X[:, 1])

X = df_sick.iloc[:, :-1].values
y = df_sick.iloc[:, -1].values

plt.scatter(X[:, 0], X[:, 1])
