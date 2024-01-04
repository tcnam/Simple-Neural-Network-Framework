from neuralnet.train import train
from neuralnet.nn import NeuralNet
import numpy as np
from neuralnet.layers import Linear, Tanh

if __name__=="__main__":
    inputs=np.array([
        [0,0],
        [1,0],
        [0,1],
        [1,1]
    ])
    targets=np.array([
        [1,0],
        [0,1],
        [0,1],
        [1,0]
    ])
    
    net=NeuralNet([
        Linear(input_size=2, output_size=2),
        Tanh(),
        Linear(input_size=2, output_size=2)
    ])
    
    train(net=net, inputs=inputs, targets=targets)
    
    for x, y in zip(inputs, targets):
        prediction=net.forward(x)
        print(x, prediction, y)