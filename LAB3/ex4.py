import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits


X = np.loadtxt("yeast.data", usecols = (1,2,3,4,5,6,7,8))
y_text = np.loadtxt("yeast.data", usecols = (9), dtype=str)

X_train, X_test, y_train, y_test = train_test_split(X, y_text, stratify = y_text, test_size = 0.2)

network = MLPClassifier(solver = 'adam', hidden_layer_sizes = (80, 70, 60), max_iter = 1000, tol = 0.001, activation = 'relu')
network.fit(X_train, y_train)
predicted_labels = network.predict(X_train)
matrix = confusion_matrix(y_train, predicted_labels)
print(matrix)
print(network.score(X_test, y_test))