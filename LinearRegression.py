import numpy as np

class LinearRegression():

    # init function to set lr, epochs
    # fit function to fit linear regression on data
    # predict function to predict value of input data

    def __init__(self, lr = 0.001, epochs = 100):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def fit(self,X,Y):
        # Define weights and bias based on data
        num_points, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Define gradients of weights and bias
        # L(oss) = 1/N * sum_i((y_i - y_i_pred)**2)
        # y_pred = wx + b
        # dL/dB = dL/dy * dy/db = 2/N * sum_i(y_i - y_i_pred) * 1
        # dL/dW = dL/dy * dy/dW = 2/N * sum_i((y_i - y_i_pred) * x_i)
        for _ in range(self.epochs):
            Y_pred = np.dot(X,self.weights) + self.bias

            dw = (2/num_points)*np.dot(X.T,Y-Y_pred)
            db = (2/num_points)*(Y-Y_pred)

            self.weights = self.weights + self.lr*dw
            self.bias += self.lr*db

    def predict(self,X):
        return np.dot(X,self.weights) + self.bias
