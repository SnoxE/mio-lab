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

score = 0
for solv in ['sgd', 'adam']:
    for j in [10, 50, 300, 1000]:
        for i in [[4, 4]]:#, [3, 3], [8, 8], [15, 15]]:
            for it in range(0, 5):
                network = MLPClassifier(solver = solv, hidden_layer_sizes = (i), max_iter = j, tol = 0.001, activation = 'relu')
                network.fit(X_train, y_train)
                predicted_labels = network.predict(X_train)
                matrix = confusion_matrix(y_train, predicted_labels)
                # print(matrix)
                score += network.score(X_test, y_test)
            print("solver: " + str(solv) + ", " + "max_iter: " + str(j) + ", " + "hls: " + str(i))
            if solv == 'sgd':
                print(network.learning_rate)
            print("score: " + str(score / 5))
            score = 0
        print("\n")
    print("\n")