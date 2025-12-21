#  MyTorch/MyOptimizer.py
import numpy as np
from typing import Union

# class BaseOptimizer:
    # def get_param():

class SGD:
    # SGD randomly pick one gradient for update
    def __init__(self, layers, lr=0.001) -> None:
        self.layers = layers
        self.lr = lr

    def step(self):
        for layer in self.layers:
            for weight,gradient in layer.params():
                # (W(in_dim,out_dim), dW(in_dim,out_dim))
                weight -= self.lr * gradient
                    
    def zero_grad(self):
        for layer in self.layers:
            for _, gradient in layer.params():
                # param = (W,dW)
                gradient[:] = 0
                

# Adaptive Momentum Estimation
# It's like SGD but smarter — it adapts the learning rate for each parameter based on:
# First moment (m): Running average of gradients (like momentum)
# Second moment (v): Running average of squared gradients (like RMSprop)

class Adam:
    def __init__(self, layers, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.layers = layers
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # timestamp for bias correction
        self.t = 0

        # store m_t, v_t for later
        self.m_t = []
        self.v_t = []

        for layer in layers:
            cur_m = []
            cur_v = []
            
            for param, _  in layer.params():
                cur_m.append(np.zeros_like(param))
                cur_v.append(np.zeros_like(param))

            self.m_t.append(cur_m)
            self.v_t.append(cur_v)

    def step(self):
        for layer_idx, layer in enumerate(self.layers):
            self.t += 1
            for param_idx, (weight,gradient) in enumerate(layer.params()):
                # 1. Update m_t (first moment)
                m_t = self.m_t[layer_idx][param_idx]
                m_t = self.beta1*m_t + (1-self.beta1)*gradient
                self.m_t[layer_idx][param_idx] = m_t

                # 2. Update v_t (second moment)
                v_t = self.v_t[layer_idx][param_idx]
                v_t = self.beta2*v_t + (1-self.beta2)*gradient**2
                self.v_t[layer_idx][param_idx] = v_t

                # 3. Bias correction
                m_t_hat = m_t / (1 - self.beta1**self.t)
                v_t_hat = v_t / (1 - self.beta2**self.t)

                # 4. Update parameter
                weight -= self.alpha * (m_t_hat / (np.sqrt(v_t_hat)+self.epsilon) )
            
    def zero_grad(self):
        for layer in self.layers:
            for _,gradient in layer.params():
                # inplace operation
                gradient[:] = 0