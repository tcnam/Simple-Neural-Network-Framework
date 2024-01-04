import numpy as np

class Loss:
    def loss(self, prediction: np.ndarray, actual:np.ndarray)->float:
        raise NotImplementedError
    def grad(self, prediction: np.ndarray, actual:np.ndarray)-> float:
        raise NotImplementedError

class MSE(Loss):
    def loss(self, prediction:np.ndarray, actual:np.ndarray)->float:
        return np.sum((prediction-actual)**2)
    
    def grad(self, prediction:np.ndarray, actual:np.ndarray)->np.ndarray:
        return 2*(prediction-actual)