# coding utf-8
import numpy as np
import os

class MLP_MNIST(object):
    """
    MLP for MNIST classification
    Notification: one-hot labels are required
    """

    def __init__(self, input_nodes, h1_nodes, h2_nodes, output_nodes, learning_rate=0.1):
        """
        # Arguments
            input_nodes: The number of the features in the input data
            h1_nodes: The number of the nodes in the hidden layer 1
            h2_nodes: The number of the nodes in the hidden layer 2
            output_nodes: The number of the nodes in the output layer
            learning_rate: learning rate
        """
        self.input_nodes = input_nodes
        self.h1_nodes = h1_nodes
        self.h2_nodes = h2_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        
        # initialize weight & bias
        # input to hidden_1
        self.W1 = np.random.randn(self.input_nodes, self.h1_nodes) / np.sqrt(self.input_nodes)
        self.b1 = np.zeros(self.h1_nodes)
        # hidden_1 to hidden_2
        self.W2 = np.random.randn(self.h1_nodes, self.h2_nodes) / np.sqrt(self.h1_nodes)
        self.b2 = np.zeros(self.h2_nodes)
        # hidden_2 to output
        self.W3 = np.random.randn(self.h2_nodes, self.output_nodes) / np.sqrt(self.h2_nodes)
        self.b3 = np.zeros(self.output_nodes)
    
    ###### Train ######

    def train(self, X, y):
        """Update trainable parameters using backpropagation"""
        h1_input, h1_output, h2_input, h2_output, final_output = self.__forwardpass_train(X)
        dW1, db1, dW2, db2, dW3, db3 = self.__backpropagation(X, y, h1_input, h1_output, h2_input, h2_output, final_output)
        self.__update_weights(dW1, db1, dW2, db2, dW3, db3)
    
    def __forwardpass_train(self, X):
        """Make forward pass and return the computation results for backpropagation"""
        # hidden_1
        h1_input = np.dot(X, self.W1) + self.b1
        h1_output = self.__relu(h1_input)
        # hidden_2
        h2_input = np.dot(h1_output, self.W2) + self.b2
        h2_output = self.__relu(h2_input)
        # output
        o_input = np.dot(h2_output, self.W3) + self.b3
        final_output = self.__softmax(o_input)
        return h1_input, h1_output, h2_input, h2_output, final_output
    
    def __backpropagation(self, X, y, h1_input, h1_output, h2_input, h2_output, final_output):
        """Backpropagate the loss and calculate the partial derivative of the trainable parameters"""
        batch_size = y.shape[0]

        output_error_term = self.__softmax_cross_entropy_loss(y, final_output)
        db3 = np.sum(output_error_term, axis = 0) / batch_size
        dW3 = np.dot(h2_output.T, output_error_term) / batch_size

        h2_error_term = np.dot(output_error_term, self.W2.T) * self.__relu_derivative(h2_input) 
        db2 = np.sum(h2_error_term, axis=0) / batch_size
        dW2 = np.dot(h1_output.T, h2_error_term) / batch_size
        
        h1_error_term = np.dot(h2_error_term, self.W1.T) * self.__relu_derivative(h1_input)       
        db1 = np.sum(h1_error_term, axis=0) / batch_size
        dW1 = np.dot(h1_output.T, h1_error_term) / batch_size
        
        return dW1, db1, dW2, db2, dW3, db3
    
    def __update_weights(self, dW1, db1, dW2, db2, dW3, db3):
        self.W1 += -self.learning_rate * dW1
        self.b1 += -self.learning_rate * db1
        self.W2 += -self.learning_rate * dW2
        self.b2 += -self.learning_rate * db2
        self.W3 += -self.learning_rate * dW3
        self.b3 += -self.learning_rate * db3

    ###### Predict ######

    def predict_proba(self, X):
        # hidden_1
        h1_input = np.dot(X, self.W1) + self.b1
        h1_output = self.__relu(h1_input)
        # hidden_2
        h2_input = np.dot(h1_output, self.W2) + self.b2
        h2_output = self.__relu(h2_input)
        # output
        o_input = np.dot(h2_output, self.W3) + self.b3
        y_hat = self.__softmax(o_input)
        return y_hat
    
    def predict_label(self, X):
        prob = self.predict_proba(X)
        return np.argmax(prob)

    ###### Save/Load ######

    def save_model(self, dir_path):
        """store parameters as .npz format"""
        np.savez(dir_path + os.path.sep + "weights.npz", 
                 W1=self.W1, W2=self.W2, W3=self.W3)
        np.savez(dir_path + os.path.sep + "biases.npz",
                 b1=self.b1, b2=self.b2, b3=self.b3)

    def load_model(self, dir_path):
        """load parameters from .npz file"""
        weights = np.load(dir_path + os.path.sep + "weights.npz")
        biases = np.load(dir_path + os.path.sep + "biases.npz")
        self.W1 = weights["w1"]
        self.W2 = weights["w2"]
        self.W3 = weights["w3"]
        self.b1 = biases["b1"]
        self.b2 = biases["b2"]
        self.b3 = biases["b3"]

    ##### Metrix #####
    
    def cross_entropy_loss(self, y, y_hat):
        delta = 1e-7
        return -np.sum(y_hat * np.log(y + delta))

    ##### Activation function
    
    def __relu(self, X):
        """Relu function"""
        X_ = np.copy(X)
        return np.maximum(0, X)
    
    def __relu_derivative(self, X):
        """"Derivative of the relu function"""
        X_ = np.zeros_like(X)
        X_[X >= 0] = 1
        return X_

    def __softmax(self, X):
        """"Softmax function"""
        const = np.max(X)
        exp_X = np.exp(X - const)
        return exp_X / np.sum(exp_X)
    
    def __softmax_cross_entropy_loss(self, y, y_hat):
        """Output error term, in case Activation=softmax and Loss=cross entropy"""
        return -(y- y_hat)
