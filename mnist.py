#%%
import pandas as pd
from neuralnet.train import train
from neuralnet.nn import NeuralNet
import numpy as np
from neuralnet.layers import Linear, Tanh

#%%
def main():
    data_train_pd=pd.read_csv(filepath_or_buffer='./data/train.csv')
    data_train_np = np.array(data_train_pd)
    y_train = data_train_np[:,0]
    x_train = data_train_np[:,1:data_train_np.shape[1]]
    net=NeuralNet([
        Linear(input_size=784, output_size=64),
        Tanh(),
        Linear(input_size=64, output_size=10)
    ])
    
    train(net=net, inputs=x_train, targets=y_train)
    
    for x, y in zip(x_train, y_train):
        prediction=net.forward(x)
        print(prediction, y)

#%%
if __name__=="__main__":
    main()



#%%
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28))
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
    
#%%
test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)