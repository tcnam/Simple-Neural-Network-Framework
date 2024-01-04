from neuralnet.nn import NeuralNet
from neuralnet.loss import Loss, MSE
from neuralnet.optim import Optimizer, SGD
from neuralnet.data import DataIterator, BatchIterator
import numpy as np

def train(net:NeuralNet, 
          inputs:np.ndarray, 
          targets:np.ndarray, 
          num_epochs:int=5000,
          iterator: DataIterator=BatchIterator(),
          loss: Loss=MSE(),
          optimizer: Optimizer=SGD()) -> None:
    for epoch in range(num_epochs):
        epoch_loss=0.0
        for batch in iterator(inputs, targets):
            prediction=net.forward(batch.inputs)
            epoch_loss+=loss.loss(prediction=prediction, actual=batch.targets)
            grad=loss.grad(prediction=prediction, actual=batch.targets)
            net.backward(grad=grad)
            optimizer.step(net=net)
        print(epoch, epoch_loss)