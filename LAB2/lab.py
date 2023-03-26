from sklearn.linear_model import Perceptron
import numpy as np

# przygotowujemy wzorcowe dane uczące. X to współrzędne, y to klasa, do której należą
X = [[0.,0.],[1.,0.],[-1.,0],[-1.,-1],[1.,1]]
y = [0,1,0,0,1]

# przygotowujemy perceptron.
neuron = Perceptron(tol=1e-3, max_iter = 20)

# uczymy neuron przez wskazaną liczbę epok lub do zatrzymania się uczenia 
neuron.fit(X, y)

# możemy sprawdzić jak udane było uczenie:
print(neuron.score(X, y))