from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt

data = load_iris()
train_data, test_data, train_label, test_label = train_test_split(data.data, data.target, test_size = 0.2)

res = []
epochs = [1, 2, 5, 10, 20, 50, 150, 250]

for n in range(1, 1501):
    # clf = MLPClassifier(hidden_layer_sizes = (10, ), max_iter = n, random_state = 42, tol = 0.0001, early_stopping = False)
    neuron = Perceptron(tol=1e-3, max_iter = n, early_stopping = False)
    neuron.fit(train_data, train_label)
    # clf.fit(train_data, train_label)
    score = neuron.score(test_data, test_label)
    res.append(score)
    print(f"epochs: {n}, score: {score}")

plt.plot(range(1, len(res) + 1), res)
plt.title('Corelation between number of epochs and accurracy')
plt.xlabel('Number of epochs')
plt.ylabel('Fitting accurracy')
plt.savefig('ex5.png')