import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPRegressor

def approxfunction(x):
    return 3.18170067 * np.tanh(-0.6766211 * x + 2.02225719) + 3.33241661 * np.tanh(-0.66203193 * x + -1.98992291) + -9.52853027 * np.tanh(-0.12396696 * x + -0.03793958)

def betterapprox(x, network):
    return network.coefs_[1][0][0] * np.tanh(network.coefs_[0][0][0] * x + network.intercepts_[0][0]) + network.coefs_[1][1][0] * np.tanh(network.coefs_[0][0][1] * x + network.intercepts_[0][1]) + network.coefs_[1][2][0] * np.tanh(network.coefs_[0][0][2] * x + network.intercepts_[0][2]) + network.intercepts_[1][0]

x = np.arange(-2 * np.pi, 2*np.pi, 0.1)   # start,stop,step
y = np.sin(x)
plt.figure()
plt.plot(x,y)


network = MLPRegressor(solver='adam', hidden_layer_sizes=(3), max_iter = 500000, tol = 0.0000001, activation = 'tanh')
network.fit(x.reshape(-1,1), y)
y_predicted = network.predict(x.reshape(-1,1))

# zobaczmy, jak wygląda funkcja aproksymująca:
plt.plot(x, y, "b")
plt.plot(x, y_predicted, "r")
plt.plot(x, betterapprox(x, network), "g")
plt.show()

print(network.coefs_)
# print(network.coefs_[0][0][0], network.coefs_[0][0][1], network.coefs_[0][0][2])
# print(network.coefs_[1][0][0],  network.coefs_[1][1][0], network.coefs_[1][2][0])
print(network.intercepts_)

# print(network.intercepts_[1][0])