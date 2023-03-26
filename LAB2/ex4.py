from sklearn.linear_model import Perceptron
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data[:, :2]
y = iris.target
neuron = Perceptron(tol=1e-3, max_iter = 20)

score = 0
#1 
for i in range(0, 5):
    train_data, test_data, train_label, test_label = train_test_split(X, y, test_size = 0.2)

    neuron.fit(train_data, train_label)
    score += neuron.score(test_data, test_label)
    print("train_test_split 0.8/0.2: " + str(neuron.score(test_data, test_label)))

print("Average train_test_split 0.8/0.2 score: " + str(score/5) + "\n")

score2 = 0
#2
for i in range(0, 5):
    indices = np.random.permutation(X.shape[0])
    test_size = int(X.shape[0] * 0.2)
    train_data2, test_data2 = X[indices[test_size:]], X[indices[:test_size]]
    train_label2, test_label2 = y[indices[test_size:]], y[indices[:test_size]]

    neuron.fit(train_data2, train_label2)
    score2 += neuron.score(test_data2, test_label2)
    print("Permutation: " + str(neuron.score(test_data2, test_label2)))

print("Average perm score: " + str(score2/5) + "\n")

score3 = 0
#3
for i in range(0, 5):
    test_size = int(0.2 * len(X))
    y = y.ravel()

    test_idxs = np.random.choice(range(len(X)), size = test_size, replace = False)
    train_idxs = np.array(list(set(range(len(X))) - set(test_idxs)))

    train_data3, train_label3 = X[train_idxs], y[train_idxs]
    test_data3, test_label3 = X[test_idxs], y[test_idxs]
    neuron.fit(train_data3, train_label3)
    score3 += neuron.score(test_data3, test_label3)
    print("Python rand: " + str(neuron.score(test_data3, test_label3)))

print("Average python rand score: " + str(score3/5))