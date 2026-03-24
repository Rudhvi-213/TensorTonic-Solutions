import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    x_samples, x_features = X.shape
    w = np.zeros(x_features)
    b= 0

    for _ in range(steps):
        Z = X @ w + b
        output = _sigmoid(Z)

        error = output - y

        dw = (X.T @ error) / x_samples
        db = np.mean(error)

        w -= lr * dw
        b -= lr * db
    
    return w, b