from draw_NN import DrawNN
from collections import deque
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import time

def ReLU(Z):
    # This will be applied to each element of Z
    return np.maximum(0, Z)

def softmax(Z):
    e_x = np.exp(Z - np.max(Z))
    return e_x / e_x.sum(axis=0)

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, int(Y.max() + 1)))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z):
    return Z>0

def get_predictions(A):
    return np.argmax(A, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

class NeuralNetwork:
    def __init__(self, layers, size_input = 784, size_output = 10, seed = 10 ):
        
        if len(layers) < 2:
            print(f'!! Insert at least TWO hidden layer in the neural network !!')
            layers = [10, 10]
            
        if layers[-1] != size_output:
            print(f'!! The size of the last layer has to be the same of the output ({size_output}) !!')
            layers[-1] = size_output
        
        self.layers = layers
        self.num_layers = len(layers)
        self.size_input = size_input
        self.size_output = size_output
        
        self.W, self.b = self.init_params(seed = seed)
    
    def init_params(self, seed):
        
        W = []
        b = []
        np.random.seed(seed)
        W.append( np.random.rand(self.layers[0], self.size_input) - 0.5)
        b.append(np.random.rand(self.layers[0], 1) - 0.5)
        
        for i in range(self.num_layers - 1):
            W.append( np.random.rand(self.layers[i+1], self.layers[i]) - 0.5)
            b.append(np.random.rand(self.layers[i+1], 1) - 0.5)
        
        return W, b

    def prediction(self, X):
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        Z = self.W[0].dot(X) + self.b[0]
        
        for i in range(1,self.num_layers):
            A = ReLU(Z)
            if A.ndim == 1:
                A = A.reshape(-1, 1)
            Z = self.W[i].dot(A) + self.b[i]
        
        output = softmax(Z)
        
        return get_predictions(output)

    def forward_prop(self, X):
        Z = []
        A = []
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        Z.append(self.W[0].dot(X) + self.b[0])
        
        for i in range(1,self.num_layers):
            A.append(ReLU(Z[-1]))
            if A[-1].ndim == 1:
                A[-1] = A[-1].reshape(-1, 1)
            Z.append(self.W[i].dot(A[-1]) + self.b[i])
        
        A.append(softmax(Z[-1]))
        
        return Z, A 
    
    def backward_prop(self, Z, A, X, Y):
        dW = deque()
        db = deque()
        m = Y.size
        one_hot_Y = one_hot(Y)
        
        dZ = A[-1] - one_hot_Y
        
        for i in range(2,self.num_layers+1):
            dW.appendleft(1 / m * dZ.dot(A[-i].T))
            db.appendleft((1/ m * dZ.sum(axis=1)).reshape(-1,1))
            dZ = self.W[-i + 1].T.dot(dZ) * deriv_ReLU(Z[-i])
            
        dW.appendleft(1 / m * dZ.dot(X.T))
        db.appendleft((1/ m * dZ.sum(axis=1)).reshape(-1,1))
        
        return list(dW), list(db)
    
    def update_param(self, dW, db, alpha = 0.1):
        
        for i in range(self.num_layers):            
            self.W[i]= self.W[i] - alpha * dW[i]
            self.b[i] = self.b[i] - alpha * db[i]
            
    def iterate(self, X, Y, alpha = 0.1):
        
        Z, A = self.forward_prop(X)
        dW, db = self.backward_prop(Z, A, X, Y)
        self.update_param(dW, db, alpha = alpha)
        
        return get_accuracy(get_predictions(A[-1]), Y)
        
    def training_gradient_descent(self, X, Y, X_val, Y_val, iterations = 100, alpha = 0.1, show = False):
        t_start = time.time()
        accuracy = []
        val_accuracy = []
        stop_warnings = 0
        
        W_final = self.W
        b_final = self.b
        final_version = 0
        for i in range(iterations):
            result = self.iterate(X, Y, alpha = alpha)
            
            accuracy.append(result)
            
            prediction = self.prediction(X_val)
            val_accuracy.append(get_accuracy(prediction, Y_val))
            
            if len(val_accuracy)>2 and val_accuracy[-1] < val_accuracy[-2]:
                stop_warnings +=1
            else:
                W_final = self.W
                b_final = self.b
                final_version = i
                stop_warnings = 0
                
            if stop_warnings >= 50:
                break
            
            if i % 200 == 0 and show:
                print(f'Iteration {i}')
                print(f'Accuracy : {accuracy[-1]}')
        
        if show:
            plt.plot(accuracy, color='r', label='Training Accuracy')
            plt.plot(val_accuracy, color='g', label='Validation Accuracy')
            plt.axvline(x = final_version, color = 'b', linestyle = '--')
            
            plt.xlabel("Iteration")
            plt.ylabel("Accuracy")
            plt.title("Training and Validation comparation")
            
            plt.legend()
            
            plt.show()
            
        print(f'Training finished in {i+1} iterations')       
        print(f'Final Accuracy : {accuracy[-1]}')   
        
        self.W = W_final
        self.b = b_final 
        return time.time() - t_start, accuracy    
        
    def draw(self):
        network = DrawNN( [1] + self.layers + [1])
        network.draw()
        

