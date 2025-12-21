# Train_CNN.py
import numpy as np
import os
import cv2
import time

from MyCNN import MyConv2D, MaxPool2D, Flatten
from MyLinear import Linear
from MyActivation import ReLU, Sigmoid
from MyLoss import SoftmaxCrossEntropy
from MyOptimizer import Adam

TRAIN_PATH = 'xray_dataset_covid19/train'
TEST_PATH = 'xray_dataset_covid19/test'
CLASS_MAP = {'NORMAL':0, 'PNEUMONIA':1}
np.random.seed(42)

def load_data(data_dir, img_size=(128,128)):

    images = []
    labels = []

    for class_name, class_idx in CLASS_MAP.items():
        class_dir = os.path.join(data_dir,class_name)

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir,img_name)

            # load grey scale directly
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, img_size)
            # normalize
            img = img/255.0 

            images.append(img)
            labels.append(class_idx)

    X = np.array(images).reshape(-1,1,*img_size)
    y = np.array(labels)

    return X,y

class CNN:
    def __init__(self):
        self.conv1 = MyConv2D(in_channels=1, out_channels=8, kernel_size=3)
        self.act1 = ReLU()
        self.pool1 = MaxPool2D(kernel_size=2)

        self.conv2 = MyConv2D(in_channels=8, out_channels=16, kernel_size=3)
        self.act2 = ReLU()
        self.pool2 = MaxPool2D(kernel_size=2)
        
        self.flatten = Flatten()

        self.linear1 = Linear(in_dim=3136, out_dim=64)
        self.act3 = ReLU()
        self.linear2 = Linear(in_dim=64, out_dim=2) # 2 classes
        
        self.layers = [
            self.conv1, self.act1, self.pool1,
            self.conv2, self.act2, self.pool2,
            self.flatten,
            self.linear1, self.act3, self.linear2
        ]

    def forward(self,x):
        out = x
        for layers in self.layers:
            out = layers.forward(out)

        return out
    
    def backward(self, grad):
        for layer in self.layers[::-1]:
            grad = layer.backward(grad)
    
        return grad
    
    def params(self):
        all_params = []
        for layer in self.layers:
            all_params.extend(layer.params())

        return all_params

def train_model(X_train, y_train, X_test, y_test, model, loss_fn, optimizer, epochs=10, batch_size=16, log_fn=print):

    for epoch in range(epochs):
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        # we don't need to shuffle test set since it doesn't change training process
        
        total_loss = 0
        num_batches = 0

        # train
        for i in range(0,len(X_train), batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]

            optimizer.zero_grad()
            out = model.forward(X_batch)
            loss = loss_fn.forward(out,y_batch)

            grad = loss_fn.backward()
            grad = model.backward(grad)

            optimizer.step()

            total_loss  += loss
            num_batches += 1

        avg_loss = total_loss / num_batches
        log_fn(f'---  Epoch {epoch+1}/{epochs}  ---')
        log_fn(f'    Train Loss: {avg_loss:.4f}')

        # eval
        total_loss = 0
        num_batches = 0
        total_correct = 0
        total_samples = 0
        for i in range(0,len(X_test), batch_size):
            X_batch = X_test[i:i+batch_size]
            y_batch = y_test[i:i+batch_size]

            out = model.forward(X_batch)
            loss = loss_fn.forward(out,y_batch)
            total_loss  += loss
            num_batches += 1

            prediction = np.argmax(out,axis=1)
            correct = np.sum(prediction == y_batch)
            total_correct += correct
            total_samples += len(y_batch)


        avg_loss = total_loss / num_batches
        accuracy = total_correct/total_samples
        log_fn(f'    Validation Loss: {avg_loss:.4f}')
        log_fn(f'    Validation Accuracy: {accuracy:.4f}')

    return model

    

if __name__ == '__main__':

    print('Loading Train data ...')
    X_train, y_train = load_data(TRAIN_PATH, img_size=(64, 64))
    print('Loading Test data ...')
    X_test, y_test = load_data(TEST_PATH, img_size=(64, 64))

    # print(f"Train: X={X_train.shape}, y={y_train.shape}")
    # print(f"Test: X={X_test.shape}, y={y_test.shape}")
    # print(f"Classes: {np.unique(y_train)}")

    print('Building CNN model ...')
    model = CNN()

    # opitmizer and loss
    loss_fn = SoftmaxCrossEntropy()
    optimizer = Adam(layers=model.layers)

    with open('training_log.txt', 'w') as f:
        def log(msg):
            print(msg)
            f.write(msg + '\n')
            f.flush()
        
        start_time = time.time()
        model = train_model(X_train,y_train,X_test,y_test, model, loss_fn, optimizer, epochs=10, batch_size=16, log_fn=log)
        end_time = time.time()

        log(f'time used: {end_time - start_time}')

    
