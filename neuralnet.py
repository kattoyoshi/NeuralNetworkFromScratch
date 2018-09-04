# coding utf-8
import numpy as np

class MLP_MNIST(object):
    """
    MLP for MNIST classification
    Notification: one-hot labels are required
    """

    def __init__(self, input_nodes, h1_nodes, h2_nodes, output_nodes, eta=0.1):
        """
        # Arguments
            input_nodes: The bumber of the features in the input data
            h1_nodes: The number of the nodes in the hidden layer 1
            h2_nodes: The number of the nodes in the hidden layer 2
            output_nodes: The number of the nodes in the output layer
            eta: learning rate
        """
        self.input_nodes = input_nodes
        self.h1_nodes = h1_nodes
        self.h2_nodes = h2_nodes
        self.output_nodes = output_nodes
        self.eta = eta
        
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
        hidden_output, final_output = self.forwardpass_train(X)
        dW1, db1, dW2, db2 = self.backpropagation(X,y,hidden_output, final_output)
        self.update_weights(dW1, db1, dW2, db2)
    
    def forwardpass_train(self, X):
        # hidden_1
        h1_input = np.dot(X, self.W1) + self.b1
        h1_output = self.relu_func(h1_input)
        # hidden_2
        h2_input = np.dot(h1_output, self.W2) + self.b2
        h2_output = self.relu_func(h2_input)
        # output
        o_input = np.dot(h2_output, self.W3) + self.b3
        final_output = self.softmax_func(o_input)
        return h1_output, h2_output, final_output
    
    def backpropagation(self, X, y, h1_output, h2_output, final_output):
        
        batch_size = y.shape[0]
        
        error = self.cross_entropy_loss(y, final_output)
        output_error_term = error * 1
        
        db2 = np.sum(output_error_term, axis = 0) / batch_size
        dW2 = np.dot(hidden_output.T, output_error_term) / batch_size
        hidden_error = np.dot(output_error_term, self.W2.T)
        hidden_error_term = (1 - hidden_output) * hidden_output * hidden_error
        
        db1 = np.sum(hidden_error_term, axis=0) / batch_size
        dW1 = np.dot(X.T, hidden_error_term) / batch_size
        
        return dW1, db1, dW2, db2
    
    def update_weights(self, dW1, db1, dW2, db2, dW3, db3):
        self.W1 += dW1
        self.b1 += db1
        self.W2 += dW2
        self.b2 += db2
        self.W3 += dW3
        self.b3 += db3

    ###### Inference ######

    def predict_proba(self, X):
        # hidden_1
        h1_input = np.dot(X, self.W1) + self.b1
        h1_output = self.relu_func(h1_input)
        # hidden_2
        h2_input = np.dot(h1_output, self.W2) + self.b2
        h2_output = self.relu_func(h2_input)
        # output
        o_input = np.dot(h2_output, self.W3) + self.b3
        y_hat = self.softmax_func(o_input)
        return y_hat
    
    def predict_label(self, X):
        prob = self.predict_proba(X)
        return np.argmax(prob)

    ################
    
    def relu_func(self, X):
        return np.maximum(0, X)
    
    def softmax_func(self, X):
        const = np.max(X)
        exp_X = np.exp(X - const)
        return exp_X / np.sum(exp_X)
    
    def cross_entropy_loss(self, y, y_hat):
        delta = 1e-7
        return -np.sum(y_hat * np.log(y + delta))
    
    


def main():
    # load iris dataset
    iris_data = datasets.load_iris()
    print(iris_data.keys())
    #print(iris_data)


def TwoLayerNet():
    pass


if __name__ == '__main__':
    main()

