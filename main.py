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


# object initialization.
nn = NeuralNetwork([784,128,64,10])


# loading dataset.
print("\nLoading Datasets.")
X_train = load_images("Mnist/train-images.idx3-ubyte")
y_train = load_labels("Mnist/train-labels.idx1-ubyte").reshape(-1,1)

X_test = load_images("Mnist/t10k-images.idx3-ubyte")
y_test = load_labels("Mnist/t10k-labels.idx1-ubyte").reshape(-1,1)
print("\nDataset loaded.")

# training.
print("\nTraining Neural Network.")
nn.train(X_train[:50], y_train[:50],10,1e-3)

print("\nTraining Completed.")


# plt.imshow(X_train[0].reshape(28,28))
# plt.show()

# print(X_train[0].reshape(28,28))