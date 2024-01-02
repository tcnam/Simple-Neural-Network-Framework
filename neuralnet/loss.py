from neuralnet.tensor import Tensor
import numpy as np

class Loss:
    def loss(self, prediction: Tensor, actual: Tensor)-> float:
        raise NotImplementedError
    
    def grad(self, prediction: Tensor, actual: Tensor)-> Tensor:
        raise NotImplementedError
    
class MSE(Loss):
    """
    Mean squared error
    """
    def loss(self, prediction: Tensor, actual: Tensor)-> float:
        return np.sum((prediction-actual)**2)
    
    def grad(self, prediction: Tensor, actual: Tensor)-> Tensor:
        return 2*(prediction-actual)