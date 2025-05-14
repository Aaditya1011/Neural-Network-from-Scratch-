import numpy as np

class NeuralNetwork:

    def __init__(self,layer_size):

        self.layer_size = layer_size
        self.weights = []
        self.biases = []
        self.cache = []

        for i in range(len(self.layer_size) - 1):

            input_size = layer_size[i]
            output_size = layer_size[i+1]

            W = np.random.randn(input_size,output_size) * np.sqrt(2./input_size)
            b = np.zeros((1,output_size))

            self.weights.append(W)
            self.biases.append(b)

    # Relu function.
    def relu(self,Z):
        return np.maximum(0,Z)

    # Relu derivative function.
    def relu_derivative(self,Z):
        return (Z > 0).astype(float) 

    # softmax function.
    def softmax(self,Z):
        Z = Z - np.max(Z,axis=1,keepdims=True)
        Z_exp = np.exp(Z)
        output = Z_exp/np.sum(Z_exp,axis=1,keepdims=True)
        return output

    # forward propagation.
    def forward(self,X):
        self.cache = []

        Z1 = X @ self.weights[0] + self.biases[0]
        A1 = self.relu(Z1)
        self.cache.append((Z1,A1))

        Z2 = A1 @ self.weights[1] + self.biases[1]
        A2 = self.relu(Z2)
        self.cache.append((Z2,A2))

        Z3 = A2 @ self.weights[2] + self.biases[2]
        A3 = self.softmax(Z3)
        self.cache.append((Z3,A3))
        
        return A3

    # loss function.
    def compute_loss(self,Y_train, Y_pred):
        if Y_train.ndim == 1:
            Y_train = np.eye(self.layer_size[-1])[Y_train]

        m = Y_train.shape[0]
        loss = -np.sum(Y_train * np.log(Y_pred+1e-8)) / m
        return loss 

    # backward propagation.
    def backward(self,X,Y,alpha):
        if Y.ndim == 1:
            Y = np.eye(self.layer_size[-1])[Y]
        
        Z1,A1 = self.cache[0]
        Z2,A2 = self.cache[1]
        Z3,A3 = self.cache[2]

        # backward derivative using chain rule.
        dZ3 = A3 - Y
        dW3 = A2.T @ dZ3
        db3 = np.sum(dZ3,axis=0,keepdims=True)

        dA2 = dZ3 @ self.weights[2].T
        dZ2 = dA2 * self.relu_derivative(Z2)
        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2,axis=0,keepdims=True)

        dA1 = dZ2 @ self.weights[1].T
        dZ1 = dA1 * self.relu_derivative(Z1)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1,axis=0,keepdims=True)

        # gradient clipping.
        for grad in [dW1,db1,dW2,db2,dW3,db3]:
            np.clip(grad,-4.0,4.0,out=grad)

        # update parameters.
        self.weights[0] -= alpha * dW1
        self.biases[0] -= alpha * db1

        self.weights[1] -= alpha * dW2
        self.biases[1] -= alpha * db2

        self.weights[2] -= alpha * dW3
        self.biases[2] -= alpha * db3


    # training.
    def train(self, X_train,y_train,epochs,learning_rate):

        for epoch in range(epochs):
            
            y_pred = self.forward(X_train)

            loss = self.compute_loss(y_train,y_pred)

            self.backward(X_train,y_train,learning_rate)

            if np.any(np.isnan(loss)) or np.any(np.isinf(loss)):
                print("NaN or Inf detected in loss!")
                break

            print(f"Loss : {loss:.9f}")

    # prediction.
    def predict(self,X):
        return self.forward(X)

print(__name__)

if __name__ == "__main__":

    # object initialization.
    nn = NeuralNetwork([784,128,64,10])
        
    # dimensions for weights and biases.
    print("\nLayers Architecture :- \n")
    for n,i in enumerate(nn.weights):
        print(f"W{n}",i.shape)

    for m,j in enumerate(nn.biases):
        print(f"b{m}",j.shape)
