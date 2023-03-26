import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

data = np.loadtxt("Advertising.csv", delimiter = ",", skiprows = 1, usecols=(1,2,3,4))

X = data[:, 0:3]
y = data[:, 3]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

for act in ['relu', 'tanh']:
    network = MLPRegressor(solver='adam', hidden_layer_sizes=(80, 80, 80), max_iter = 1000, tol = 0.001, activation = act)

    network.fit(X_train, y_train)

    # uwaga, ten score to już nie jest accuracy!
    print("activation " + act + ": " + str(network.score(X_train, y_train)))

    # print(X, y)