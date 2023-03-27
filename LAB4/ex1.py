import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

data = np.loadtxt("Advertising.csv", delimiter = ",", skiprows = 1, usecols=(1,2,3,4))

X = data[:, 0:3]
y = data[:, 3]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

sum = 0

for hls in [(80, 80, 80), (30, 30)]:
    for act in ['relu', 'tanh']:
        for i in range(0, 5):
            network = MLPRegressor(solver='adam', hidden_layer_sizes=(80, 80, 80), max_iter = 5000, tol = 0.001, activation = act)
            network.fit(X_train, y_train)

            # uwaga, ten score to już nie jest accuracy!
            print("activation " + act + ", network " + str(hls) + ": " + str(network.score(X_train, y_train)))
            sum += network.score(X_train, y_train)
            # print(X, y)
        print(f"average for {hls} w/ {act} over 5 iters: {sum/5}")
        sum = 0
