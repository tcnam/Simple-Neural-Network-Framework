#%%
from typing import List

import numpy as np

from neuralnet.train import train
from neuralnet.nn import NeuralNet
from neuralnet.layers import Linear, Tanh
from neuralnet.optim import SGD

def fizz_buzz_encode(x: int) -> List[int]:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]


def binary_encode(x: int) -> List[int]:
    return [x >> i & 1 for i in range(10)]

inputs = np.array([
    binary_encode(x)
    for x in range(101, 1024)
])

targets = np.array([
    fizz_buzz_encode(x)
    for x in range(101, 1024)
])

net = NeuralNet([
    Linear(input_size=10, output_size=50),
    Tanh(),
    Linear(input_size=50, output_size=4)
])

train(net,
      inputs,
      targets,
      num_epochs=5000,
      optimizer=SGD(lr=0.001))
count=0
for x in range(1, 10000):
    predicted = net.forward(binary_encode(x))
    predicted_idx = np.argmax(predicted)
    actual_idx = np.argmax(fizz_buzz_encode(x))
    labels = [str(x), "fizz", "buzz", "fizzbuzz"]
    if labels[predicted_idx]== labels[actual_idx]:
        count+=1
    # print(x, labels[predicted_idx], labels[actual_idx])
print(f"Accuracy: {count/10000*100}")