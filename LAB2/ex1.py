from sklearn.linear_model import Perceptron
import numpy as np
import matplotlib.pyplot as plt

n = [5, 10, 20, 100]

for i in n:

    X = np.concatenate((np.random.normal([0, -1], [1, 1], [i, 2]), np.random.normal([1, 1], [1, 1], [i, 2])))
    label = np.concatenate([[0] * i, [1] * i])

    Xtest = np.concatenate((np.random.normal([0, -1], [1, 1], [200, 2]), np.random.normal([1, 1], [1, 1], [200, 2])))
    labeltest = np.concatenate([[0] * 200, [1] * 200])

    neuron = Perceptron(tol=1e-3, max_iter = 20)
    
    neuron.fit(X, label)
    print(neuron.score(Xtest, labeltest))

    x1 = np.linspace(-2.5, 4, 200)
    # wzór x2 = a*x1+c wymaga trochę prostych przekształceń algebraicznych z postaci w1x1+w2x2+b=0
    x2 = -(1./neuron.coef_[0][1])*(neuron.coef_[0][0]*x1+neuron.intercept_[0])
    plt.figure(i)
    plt.plot(x1, x2, '-r')


    
    plt.scatter(np.array(Xtest)[:,0], np.array(Xtest)[:,1], c = np.concatenate([['red'] * 200, ['green'] * 200]))
    plt.scatter(np.array(X)[:,0], np.array(X)[:,1], c = label)
    plt.savefig('ex1/' + str(i) +'.png')



