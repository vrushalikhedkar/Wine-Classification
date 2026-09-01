import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from sklearn.datasets import load_wine

dataset = load_wine()

dataset

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

df['target'] = dataset.target

df

df.head()

X = df.iloc[:, 0:13].values

y = df.iloc[:, 13].values

X.shape

y.shape

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train.shape

X_test.shape

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

model.predict([[14.2, 1.7, 2.4, 15.6, 127, 2.8, 3.0, 0.3, 2.0, 5.5, 1.0, 3.2, 1000]])

model.predict([[60,30,40,50,60,70,10,12,23,54,67,42,15]])




