import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPRegressor



data = fetch_california_housing()
X = data.data
y = data.target

for i in range(8):
    X[:, i] /= (np.max(X[:, i]) - np.min(X[:, i]))
y /= (np.max(y) - np.min(y))

score = []

m = [1, 100, 200, 300, 400, 500 ,600, 700, 800, 900, 1000, 1300, 1500, 1750, 2000, 2300, 2500, 2750, 3000, 3500, 4000, 5000, 7500, 10000]


for epochs in range(5, 76, 5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    network = MLPRegressor(solver='adam', hidden_layer_sizes=(100, 100, 100, 100), max_iter = epochs, tol = 0.0001, activation = 'tanh')
    network.fit(X_train, y_train)
    print(network.score(X_test, y_test))
    score.append(network.score(X_test, y_test))
    print(f"epochs: {epochs}")

plt.scatter(range(5, 76, 5), score)
plt.title('Corelation between number of epochs and accurracy')
plt.xlabel('Number of epochs')
plt.ylabel('Fitting accurracy')
plt.savefig('ex4.png')