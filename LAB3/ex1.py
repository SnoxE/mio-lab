import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

data = np.loadtxt("treatment.txt", delimiter = ",")

X = data[:, 0:2]
y = data[:, 2]

X[:, 0] /= np.max(X[:, 0])
X[:, 1] /= np.max(X[:, 1])

# print(X)
# print(y)

# # all the points
# plt.figure(1)
# plt.scatter(X[:, 0], X[:, 1], c = y)
# plt.savefig("data")

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.2)

# porownac adama i sgd #TODO
counter = 0
for array in [[8, 8, 8], [20, 15, 15]]:
    network = MLPClassifier(solver='adam', hidden_layer_sizes = (array), max_iter = 1000, tol = 0.001, activation = 'relu')
    network.fit(X_train, y_train)
    predicted_labels = network.predict(X_train)
    matrix = confusion_matrix(y_train, predicted_labels)
    print(matrix)
    print(network.score(X_test, y_test))

    # siatka podzialu
    xx, yy = np.meshgrid(np.arange(0.0, 1.0, 0.025), np.arange(0.0, 1.0, 0.025))
    test_points = np.transpose(np.vstack((np.ravel(xx),np.ravel(yy))))
    prediction = network.predict(test_points)
    plt.figure(counter)
    plt.title(str(array))
    counter += 1
    plt.scatter(test_points[:, 0], test_points[:, 1], c = prediction)
    plt.savefig("ex1/" + str(array) + ".jpg")

# counter = 0
# for array in [[500, 300, 300], [4, 2], [4, 4], [10, 5], [20, 5], [20, 15], [100, 90]]:
#     network = MLPClassifier(solver='adam', hidden_layer_sizes = (array), max_iter = 1000, tol = 0.001, activation = 'relu')
#     network.fit(X_train, y_train)
#     predicted_labels = network.predict(X_train)
#     matrix = confusion_matrix(y_train, predicted_labels)
#     print(matrix)
#     print(str(network.score(X_test, y_test)) + "\n")

#     # siatka podzialu
#     xx, yy = np.meshgrid(np.arange(0.0, 1.0, 0.025), np.arange(0.0, 1.0, 0.025))
#     test_points = np.transpose(np.vstack((np.ravel(xx),np.ravel(yy))))
#     prediction = network.predict(test_points)
#     plt.figure(counter)
#     plt.title(str(array))
#     counter += 1
#     plt.scatter(test_points[:, 0], test_points[:, 1], c = prediction)
#     plt.savefig("ex1/" + str(array) + ".jpg")

