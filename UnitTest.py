import unittest
import numpy as np
import functions

class TestFunctions(unittest.TestCase):
    """test class of functions.py"""

    def test_relu(self):
        test_data = np.array([[1, 0, -0.1],
                              [-0.5, -0.001, 0.001]])
        expected = np.array([[1, 0, 0],
                            [0, 0, 0.001]])
        actual = functions.relu(test_data)
        self.assertTrue((expected==actual).all())
    
    def test_relu_derivative(self):
        test_data = np.array([[1, 0, -0.1],
                              [-0.5, -0.001, 0.001]])
        expected = np.array([[1, 0, 0],
                            [0, 0, 1]])
        actual = functions.relu_derivative(test_data)
        self.assertTrue((expected==actual).all())
    
    def test_softmax(self):
        test_data = np.array([[0.3, 2.9, 4.0],
                              [-0.5, -0.001, 0.001]])
        expected = np.array([[1., 1.]])

        actual = functions.softmax(test_data)
        print(actual)
        actual = np.sum(actual, axis=1)
        print(actual)
        self.assertTrue((expected==actual).all())
    
    #def test_cross_entropy_loss(self):


if __name__ == "__main__":
    unittest.main()