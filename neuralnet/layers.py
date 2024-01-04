import numpy as np
from typing import Dict, Callable

class Layer:
    def __init__(self) -> None:
        self.params:Dict[str, np.ndarray]={}
        self.grads:Dict[str, np.ndarray]={}
    
    def forward(self, inputs:np.ndarray)->np.ndarray:
        raise NotImplementedError
    
    def backward(self, grad:np.ndarray) -> np.ndarray:
        raise NotImplementedError

class Linear(Layer):
    def __init__(self, input_size:int, output_size:int) -> None:
        super().__init__()
        self.params['w']=np.random.randn(input_size, output_size)
        self.params['b']=np.random.randn(output_size)
    
    def forward(self, inputs:np.ndarray) ->np.ndarray:
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.grads['b']=np.sum(grad, axis=0)
        self.grads['w']=self.inputs.T @ grad
        return grad @ self.params['w'].T

F=Callable[[np.ndarray], np.ndarray]

class Activation(Layer):
    def __init__(self, f:F, f_prime:F) -> None:
        super().__init__()
        self.f=f
        self.f_prime=f_prime
        
    def forward(self, inputs: np.ndarray) ->np.ndarray:
        self.inputs=inputs
        return self.f(inputs)
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        return self.f_prime(self.inputs)*grad

def tanh(x:np.ndarray) -> np.ndarray:
    return np.tanh(x)

def tanh_prime(x:np.ndarray) -> np.ndarray:
    return 1- tanh(x)**2

def relu(x:np.ndarray) ->np.ndarray:
    return np.maximum(x, 0)

def relu_prime(x:np.ndarray)->np.ndarray:
    return x>0

def softmax(x:np.ndarray) ->np.ndarray:
    return np.exp(x)/np.sum(np.exp(x))

def softmax_prime(x:np.ndarray) ->np.ndarray:
    n=np.size(x)
    tmp=np.tile(x, n)
    return tmp*(np.identity(n)-np.transpose(tmp))


class Tanh(Activation):
    def __init__(self) -> None:
        super().__init__(tanh, tanh_prime)

class Relu(Activation):
    def __init__(self) -> None:
        super().__init__(relu, relu_prime)
        
class Softmax(Activation):
    def __init__(self) -> None:
        super().__init__(softmax, softmax_prime)
