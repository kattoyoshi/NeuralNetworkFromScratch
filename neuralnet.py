# coding utf-8
import numpy as np
import os
import functions

class MLP_MNIST():
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
        train_loss = functions.cross_entropy_loss(y, final_output)
        dW1, db1, dW2, db2, dW3, db3 = self.__backpropagation(X, y, h1_input, h1_output, h2_input, h2_output, final_output)
        self.__update_weights(dW1, db1, dW2, db2, dW3, db3)
        return train_loss
    
    def __forwardpass_train(self, X):
        """Make forward pass and return the computation results for backpropagation"""
        print(X)
        # hidden_1
        h1_input = np.dot(X, self.W1) + self.b1
        h1_output = functions.relu(h1_input)
        print(h1_output)
        # hidden_2
        h2_input = np.dot(h1_output, self.W2) + self.b2
        h2_output = functions.relu(h2_input)
        # output
        o_input = np.dot(h2_output, self.W3) + self.b3
        final_output = functions.softmax(o_input)
        return h1_input, h1_output, h2_input, h2_output, final_output
    
    def __backpropagation(self, X, y, h1_input, h1_output, h2_input, h2_output, final_output):
        """Backpropagate the loss and calculate the partial derivative of the trainable parameters"""
        batch_size = y.shape[0]s

        output_error_term = self.softmax_cross_entropy_loss(y, final_output) / batch_size
        db3 = np.sum(output_error_term, axis = 0) 
        dW3 = np.dot(h2_output.T, output_error_term)

        h2_error_term = np.dot(output_error_term, self.W3.T) * functions.relu_derivative(h2_input) 
        db2 = np.sum(h2_error_term, axis=0)
        dW2 = np.dot(h1_output.T, h2_error_term)
        
        h1_error_term = np.dot(h2_error_term, self.W2.T) * functions.relu_derivative(h1_input)       
        db1 = np.sum(h1_error_term, axis=0)
        dW1 = np.dot(X.T, h1_error_term)
        
        return dW1, db1, dW2, db2, dW3, db3
    
    def __update_weights(self, dW1, db1, dW2, db2, dW3, db3):
        self.W1 += -self.learning_rate * dW1
        self.b1 += -self.learning_rate * db1
        self.W2 += -self.learning_rate * dW2
        self.b2 += -self.learning_rate * db2
        self.W3 += -self.learning_rate * dW3
        self.b3 += -self.learning_rate * db3

    ###### Prediction ######

    def predict(self, X):
        # hidden_1
        h1_input = np.dot(X, self.W1) + self.b1
        h1_output = functions.relu(h1_input)
        # hidden_2
        h2_input = np.dot(h1_output, self.W2) + self.b2
        h2_output = functions.relu(h2_input)
        # output
        o_input = np.dot(h2_output, self.W3) + self.b3
        y_hat = functions.softmax(o_input)
        return y_hat
    
    def predict_label(self, X):
        prob = self.predict(X)
        return np.argmax(prob)
    
    ###### Evaluation ######

    def evaluate(self, X, y):
        """Return the loss of the network"""
        y_pred = self.predict(X)
        return functions.cross_entropy_loss(y, y_pred)

    ###### Loss of the Output Error term ######

    def softmax_cross_entropy_loss(self, y, y_hat):
        """Output error term, in case Activation=softmax and Loss=cross entropy"""
        return -(y- y_hat)

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
        self.W1 = weights["W1"]
        self.W2 = weights["W2"]
        self.W3 = weights["W3"]
        self.b1 = biases["b1"]
        self.b2 = biases["b2"]
        self.b3 = biases["b3"]
