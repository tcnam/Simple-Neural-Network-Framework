from neuralnet.layers import Layer
from typing import Sequence, Iterator, Tuple
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
    
    def params_and_grads(self) ->Iterator[Tuple[np.ndarray, np.ndarray]]:
        for layer in self.layers:
            for name, param in layer.params.items():
                grad=layer.grads[name]
                yield param, grad