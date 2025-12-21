# MyTorch/MyLinear.py
import numpy as np
# ReLU(x) = x if x>0 else 0
# for relu, we needs to cache ∂L/∂z = ∂L/∂a * ∂a/∂z
# where a = ReLU(z), ∂a/∂z = 1 if z>0 else 0

class Linear:
    def __init__(self, in_dim, out_dim):
        # xavier initialization
        self.W = np.random.randn(in_dim,out_dim) * np.sqrt(1/in_dim) # (in_dim, out_dim)
        self.b = np.zeros(out_dim) # (out_dim)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]

    def forward(self, x):
        # x = (batch, in_dim)
        # ∂L/∂w = ∂L/∂z * ∂z/∂w, where ∂L/∂z is from previous and ∂z/∂w = x since z = Wx
        self.cache = x
        return x @ self.W + self.b
    
    def backward(self, dZ):
        self.dW += self.cache.T @ dZ
        self.db += np.sum(dZ, axis=0)
        return dZ @ self.W.T
    

    