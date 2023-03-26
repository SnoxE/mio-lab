import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits

digits = load_digits()

X = load_digits().data
y = load_digits().target

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.2)


# for array in [[2, 2], [2, 2, 2], [2, 3], [3, 2], [3, 3], [8, 8], [8, 8, 8], [15, 15], [50, 50]]:
#     network = MLPClassifier(solver='adam', hidden_layer_sizes = array, max_iter = 1000, tol = 0.001, activation = 'relu')
#     network.fit(X_train, y_train)
#     predicted_labels = network.predict(X_train)
#     matrix = confusion_matrix(y_train, predicted_labels)
#     print(array)
#     print(matrix)
#     print(str(network.score(X_test, y_test)) + "\n")

for array in [[2, 2], [3, 3], [8, 8]]:
    for i in range(0, 3):
        network = MLPClassifier(solver='adam', hidden_layer_sizes = array, max_iter = 1000, tol = 0.001, activation = 'relu')
        network.fit(X_train, y_train)
        predicted_labels = network.predict(X_train)
        matrix = confusion_matrix(y_train, predicted_labels)
        print(array)
        print(matrix)
        print(str(network.score(X_test, y_test)) + "\n")