from neuralnet.layers import Layer
from typing import Sequence
import numpy as np

class NeuralNet:
    def __init__(self, layers:Sequence[Layer]) -> None:
        self.layers=layers
    
    def forward(self, inputs:np.ndarray) -> np.ndarray:
        for layer in self.layers:
            inputs=layer.forward(inputs=inputs)
        return inputs

    def backward(self, grad: np.ndarray) ->np.ndarray:
        for layer in reversed(self.layers):
            grad=layer.backward(grad=grad)
        return grad