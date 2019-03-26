import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

df = pd.read_csv("diabites.txt", names=names)

X = df.iloc[:, :8].values
y = df.iloc[:, -1].values

#/////////////////////////////////////////
# scale = StandardScaler()
# X = scale.fit_transform(X)
# norm = Normalizer().fit(X)
# X = norm.transform(X)

stand = MinMaxScaler(feature_range=(0,1))
X = stand.fit_transform(X) #higest accuracy is obtained in MinMaxScaler

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# np.set_printoptions(precision=2)

#training model
classifier = SVC(gamma='auto')
classifier.fit(X_train, Y_train)

accuracy = classifier.score(X_test, Y_test)

print(accuracy)
