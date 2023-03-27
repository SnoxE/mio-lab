import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPRegressor

splits = [0.2, 0.35, 0.5, 0.65, 0.8]

data = fetch_california_housing()
X = data.data
y = data.target

for i in range(8):
    X[:, i] /= (np.max(X[:, i]) - np.min(X[:, i]))
y /= (np.max(y) - np.min(y))

for split in splits:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split)

    network = MLPRegressor(solver='adam', hidden_layer_sizes=(100, 100, 100, 100), max_iter = 5000  , tol = 0.0001, activation = 'tanh')
    network.fit(X_train, y_train)
    # print(network.score(X_test, y_test))
    print(f"test split: {split}, score: {network.score(X_test, y_test)}")
