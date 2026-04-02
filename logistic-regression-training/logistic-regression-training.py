import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X = np.array(X,dtype=float)
    y = np.array(y,dtype= float).reshape(-1,1)

    N, d = X.shape
    w = np.zeros((d, 1))
    b = 0.0
    
    for _ in range(steps):
        z = X @ w + b
        p = _sigmoid(z)
        
        error = p - y
        dw = (1 / N) * (X.T @ error)
        db = (1 / N) * np.sum(error)
        
        w-= lr * dw
        b -= lr * db
    
    return w.flatten(), b