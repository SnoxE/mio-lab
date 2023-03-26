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

for i in range(0, 5):
    train_data, test_data, train_label, test_label = train_test_split(X, y, test_size = 0.2)
    neuron.fit(train_data, train_label)

    print(neuron.score(test_data, test_label))

    predictions_train = neuron.predict(train_data)
    predictions_test = neuron.predict(test_data)
    train_score = confusion_matrix(predictions_train, train_label)
    # print(train_score)
    test_score = confusion_matrix(predictions_test, test_label)
    print(test_score)
    accuracy = (train_score[0,0] + train_score[1,1] + train_score[2,2] ) / np.sum(train_score)
    # print("dane uczace:", accuracy)
    accuracy2 = (test_score[0,0] + test_score[1,1] + test_score[2,2]) / np.sum(test_score)
    # print("dane testujace:", accuracy2)
    # print(neuron.score(train_data, train_label))
    print("\n")