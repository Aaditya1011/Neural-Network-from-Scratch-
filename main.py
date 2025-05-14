import numpy as np
from neural_network import NeuralNetwork
import matplotlib.pyplot as plt

# function for loading images.
def load_images(filename):
    
    with open(filename,'rb') as f:
        f.read(16)
        data = np.frombuffer(f.read(),dtype=np.uint8)
        return data.reshape(-1,28*28) /255

# function for loading labels.
def load_labels(filename):
    
    with open(filename,'rb') as f:
        f.read(8)
        data = np.frombuffer(f.read(),dtype=np.uint8)
        return data

# using smaller dataset.
def smaller_data(X,y,size=1000):

    X_small = []
    y_small = []

    for label in np.unique(y):
        idx = np.where(y == label)[0]
        chosen_idx = np.random.choice(idx,size,replace=False)
        X_small.append(X[chosen_idx])
        y_small.append(y[chosen_idx])
    
    return np.vstack(X_small),np.hstack(y_small)

# Shuffling.
def shuffle_data(X,y):
        
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    X_shuffled = X[indices]
    y_shuffled = y[indices]

    return X_shuffled,y_shuffled

# object initialization.
nn = NeuralNetwork([784,128,64,10])


# loading dataset.
print("\nLoading Datasets.")
X_train = load_images("Mnist/train-images.idx3-ubyte")
y_train = load_labels("Mnist/train-labels.idx1-ubyte")

X_test = load_images("Mnist/t10k-images.idx3-ubyte")
y_test = load_labels("Mnist/t10k-labels.idx1-ubyte")
print("\nDataset loaded.")

X_train_small,y_train_small = smaller_data(X_train,y_train,2500)
X_test_mall,y_test_small = smaller_data(X_train,y_train,500)

# shuffling.
X_train_shuffled,y_train_shuffled = shuffle_data(X_train_small,y_train_small)
X_test_shuffled,y_test_shuffled = shuffle_data(X_test_mall,y_test_small)


# training.
print("\nTraining Neural Network.")
nn.train(X_train_shuffled, y_train_shuffled,20,0.001)
print("\nTraining Completed.")
