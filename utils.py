# coding utf-8
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os
import gzip


def batch_generator(X, y, batch_size=32, shuffle=True):
    n_data = len(X)
    index = np.arange(len(y))
    if shuffle:
        index = np.random.shuffle(index)
    
    n_loop = n_data // batch_size
    n_last_batch = n_data % batch_size
    if n_last_batch != 0:
        n_loop += 1
    
    for ii in range(0, n_data, batch_size):
        if n_data - ii < batch_size:
            features = X[index[ii:]]
            targets = y[index[ii:]]
        else:
            features = X[index[ii:ii+batch_size]]
            targets = y[index[ii:ii+batch_size]]
        yield features, targets


class mnist_dataset():
    
    def __init__(self, dir_path):
        
        assert os.path.exists(dir_path), "Arguments error: dir_path does not exist"

        # store path of the data directory
        self.dir_path = dir_path + os.sep
        # define file name
        self.ftrain_feature = self.dir_path + "train-images-idx3-ubyte.gz"
        self.ftrain_labels = self.dir_path + "train-labels-idx1-ubyte.gz"
        self.ftest_features = self.dir_path + "t10k-images-idx3-ubyte.gz"
        self.ftest_labels = self.dir_path + "t10k-labels-idx1-ubyte.gz"

        for path in [self.ftrain_feature, self.ftrain_labels, self.ftest_features, self.ftest_labels]:
            print(path)
            assert os.path.exists(path), "File error: " + path + " does not exist"
    
    def load(self):
        X_train = self.load_features(self.ftrain_feature)
        X_test = self.load_features(self.ftest_features)
        y_train = self.load_labels(self.ftrain_labels)
        y_test = self.load_labels(self.ftest_labels)

        # one-hot encoding
        y_train = self.one_hot_encoding(y_train)
        y_test = self.one_hot_encoding(y_test)
        
        return X_train, X_test, y_train, y_test

    def load_features(self, file_path):
        """Load images as 1D array"""
        with gzip.open(file_path, 'rb') as f:
            features = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
        return features.reshape(-1, 28*28)
    
    def load_labels(self, file_path):
        """Load labels as 1D array"""
        with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        return labels

    def one_hot_encoding(self, y):
        """Convert binary labels into one-hot encoding"""
        y = y.reshape(1,-1)
        y = y.transpose()
        encoder = OneHotEncoder()
        return encoder.fit_transform(y).toarray()

