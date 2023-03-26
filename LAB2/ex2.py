from sklearn.linear_model import Perceptron
import numpy as np
import matplotlib.pyplot as plt

X = np.loadtxt("fuel.txt")


for i in range(0, 5):
    neuron = Perceptron(tol=1e-3, max_iter = 40)
    neuron.fit(X[:, 0:2], X[:, 3])
    print(neuron.score(X[:, 0:2], X[:, 3]))

