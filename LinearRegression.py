import numpy as np


class LinearRegrssion:
    def __init__(self,lr=0.0001,epochs=1000):
        self.lr = lr
        self.epochs =epochs
        self.weights = 0
        self.bias = 0
    
    def fit(self,X,y):

        X = np.array(X)
        y = np.array(y).flatten()
        
        n_samples , n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for i in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            print(y_pred)
            if np.isnan(y_pred).any():
                print(f"NaN detected in y_pred at epoch {i}")
                break
            dw = (1/n_samples) * np.dot(X.T,(y_pred-y))
            db = (1/n_samples) * np.sum((y_pred-y))
            
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias -self.lr * db
    def predict(self , X):
        X = np.array(X)
        y_pred = np.dot(X,self.weights)+ self.bias
        return y_pred