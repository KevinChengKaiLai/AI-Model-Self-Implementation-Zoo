# Train.py
import numpy as np
from MyActivation import ReLU, Sigmoid
from MyLinear import Linear
from MyLoss import MSELoss
from MyOptimizer import SGD

# XOR — classic test that requires hidden layers 
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) 
y = np.array([[0], [1], [1], [0]])

batch, in_dim = X.shape

linear1 = Linear(in_dim=in_dim,out_dim=32)
act1 = ReLU()
linear2 = Linear(in_dim=32,out_dim=4)
act2 = ReLU()
linear3 = Linear(in_dim=4,out_dim=1)
act3 = Sigmoid()
layers = [linear1, act1,linear2 , act2,linear3, act3]
loss = MSELoss()
optimizer = SGD(layers = layers , lr=0.1)


epochs = 1000

for epoch in range(epochs):
    
    # forward
    out = X
    for layer in layers:
        out = layer.forward(out)
    
    loss_val = loss.forward(y,out)

    print("Predictions:")
    print(out)
    print("Targets:")
    print(y)

    # backward
    optimizer.zero_grad()
    da = loss.backward()
    for layer in layers[::-1]:
        da = layer.backward(da)
    
    # step
    optimizer.step()

    print(f'epoch {epoch} -- loss: {loss_val}')

    


