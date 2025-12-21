# MyTorch/MyActivation.py
import numpy as np

class BaseActivation:
    def params(self):
        return []

class ReLU(BaseActivation):
    def forward(self,x):
        self.a = np.maximum(0,x)
        return self.a
    
    def backward(self, dA):
        # elementwise multiplication
        return dA * (self.a > 0)
    
class Sigmoid(BaseActivation):
    # dσ/dx = σ(x)*(1-σ(x)) = a(1-a)

    def forward(self,z):
        self.a = 1 / (1 + np.exp(-z))
        return self.a
    
    def backward(self, dA):
        # elementwise multiplication
        return dA * self.a * (1-self.a) 
    
class Tanh(BaseActivation):
    # tanh(x)=2σ(2x)−1
    def forward(self,z):
        self.a = np.tanh(z)
        return self.a
    
    def backward(self, dA):
        # elementwise multiplication
        return dA * (1 - self.a**2)
    
class GeLU(BaseActivation):
    # erf(x)=2/sqrt(π​)∫0-x ​e−t^2dt
    # GELU(x)=x⋅1/2[1+erf(x/sqrt(2​)]
    # GELU approximation using tanh:
    # 0.5 * X * (1+tanh( sqrt(2/pi) * (x+0.44715x^3) ))
    def forward(self,x):
        self.x = x
        self.phi = 0.5 * (1 + np.tanh(np.sqrt(2/np.pi)*(x + 0.44715 * x**3)))
        return self.x * self.phi

    def backward(self,dA):
        return self.phi + self.x * 1/np.sqrt(2*np.pi) * np.exp(-self.x**2 /2) * dA
        


class LeakyReLU(BaseActivation):
    # x > 0 --> x 
    # x < 0 --> alpha
    def __init__(self, alpha):
        self.alpha = alpha

    def forward(self,x):
        self.x = x
        return np.where(x>0, x, self.alpha*x)
    
    def backward(self, dA):
        # elementwise multiplication
        return dA * np.where(self.x>0, 1, self.alpha)
    
    

    
    

