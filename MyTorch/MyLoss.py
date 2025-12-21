# MyTorch/MyLoss.py
import numpy as np

class BaseLoss:
    def params(self):
        return []


class MSELoss(BaseLoss):
    def forward(self, y, y_hat):
        # y_hat (batch, out_dim)
        self.batch_size, self.out_dim = y_hat.shape
        self.size = self.batch_size * self.out_dim
        self.y_hat = y_hat
        self.y = y
        # y = ground truth, y_hat = prediction
        return np.mean((y_hat - y)**2)
    
    def backward(self):
        # return size  (batch, out_dim)
        return 2 * (self.y_hat-self.y) / self.size
    


class SoftmaxCrossEntropy(BaseLoss):
    # y = (batch, 1) (1 = classes)
    # y_hat (batch, num_classes)
    def forward(self,logits:np.ndarray,y_true:np.ndarray):
        
        # if y_true is indices, do one hot encoding
        if y_true.ndim == 1 or y_true.shape[1] == 1:
            batch_size = y_true.shape[0]
            y_true = y_true.flatten()
            num_classes = logits.shape[1]
            one_hot = np.zeros_like(logits)
            one_hot[np.arange(batch_size),y_true] = 1
            y_true = one_hot

        # softmax
        # subtraction for stability 
        e_y = np.exp(logits - np.max(logits, axis=1, keepdims=True)) # (batch,num_classes)
        probability = e_y / np.sum(e_y,axis=1, keepdims=True) # (batch,num_classes) / (batch,1 -> broadcast to num_classes) = (batch,num_classes)
        # important !! axis = 1, sum over classes not batch
        # keepdims=True preserves shape for broadcasting

        # cache 
        self.y_true = y_true      # (batch, num_classes)
        self.y_hat = probability  # (batch, num_classes)

        # cross entropy  (batch,num_classes)  batch,num_classes
        cross_entropy = np.sum(-y_true * np.log(probability + 1e-8),axis=1) # sum over classes, 1e-8 prevent log0
        cross_entropy = np.mean(cross_entropy)
        
        return cross_entropy
    
    def backward(self):
        # y_hat  (batch,num_classes) 
        # y_true (batch,num_classes)
        grad = self.y_hat-self.y_true
        return grad # (batch,num_classes)