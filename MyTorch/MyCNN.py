# MyCNN.py
import numpy as np

# def im2col(x:np.ndarray, kernel_size, stride=1):
#     B, Cin, H, W = x.shape
#     K = kernel_size
#     h_out = (H - K) // stride + 1
#     w_out = (W - K) // stride + 1

#     x_col = np.zeros((B*h_out*w_out, Cin*K*K))
#     for h in range(h_out):
#         h_start = h * stride 
#         h_end = h_start + kernel_size
#         for w in range(w_out):
#             w_start = w * stride 
#             w_end = w_start + kernel_size
            
#             row_start = (h * w_out + w) * B
#             x_patch = x[:,:,h_start:h_end,w_start:w_end].reshape((B,-1)) # (B, Cin*K*K)
#             x_col[row_start:row_start+B] = x_patch
        
#     return x_col

def im2col(x, kernel_size, stride=1):
    B, Cin, H, W = x.shape
    K = kernel_size
    h_out = (H - K) // stride + 1
    w_out = (W - K) // stride + 1

    # Standard shape: (B, Cin, K, K, h_out, w_out)
    # Then transposed to make it easy to reshape
    col = np.zeros((B, Cin, K, K, h_out, w_out))
    
    for h in range(K):
        h_max = h + stride * h_out
        for w in range(K):
            w_max = w + stride * w_out
            col[:, :, h, w, :, :] = x[:, :, h:h_max:stride, w:w_max:stride]
            
    # Reshape to (B * h_out * w_out, Cin * K * K)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(B * h_out * w_out, -1)
    return col

def col2im (x_col, x_shape, kernel_size, stride = 1):
    """
    x_col: (B*h_out*w_out, C_in*K*K) - the column matrix
    x_shape: (B, C_in, H, W) - original input shape
    Returns: (B, C_in, H, W) - reconstructed image with accumulated gradients
    """
    # x_col = (B*hout*wout, Cin*K*K)
    B, Cin, H, W = x_shape 
    K = kernel_size
    
    h_out = (H - K) // stride + 1
    w_out = (W - K) // stride + 1

    x = np.zeros(x_shape) # (B, C_in, H, W)
    
    for h in range(h_out):
        h_start = h * stride 
        h_end = h_start + kernel_size
        for w in range(w_out):
            w_start = w * stride 
            w_end = w_start + kernel_size
            
            row_start = (h * w_out + w) * B
            patch_grad = x_col[row_start:row_start+B] # (B, Cin*K*K)
            patch_grad = patch_grad.reshape((B,Cin,K,K)) # (B, C_in, K, K)

            x[:,:,h_start:h_end,w_start:w_end] += patch_grad
        
    return x
    


class MyConv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.W = np.random.randn(out_channels,in_channels, kernel_size,kernel_size)
        self.dw = np.zeros_like(self.W)
        self.b = np.zeros(out_channels)
        self.db = np.zeros_like(self.b)

    def params(self):
        return [(self.W,self.dw),(self.b,self.db)]
    
    def forward(self,x:np.ndarray):
        # x (batch, in_chnl, h, w), cahce for backward
        # self.x = x 
        # out (batch, out_chnl, h_out, w_out)
        B,C_in,H,W = x.shape
        self.x_shape = x.shape

        # precalculate h_out, w_out
        h_out = (H-self.kernel_size) // self.stride + 1
        w_out = (W-self.kernel_size) // self.stride + 1

        # im2col
        self.x_col = im2col(x, self.kernel_size, self.stride) # (B*hout*wout, Cin*K*K)
        W_2D = self.W.reshape((self.out_channels,-1)) # (Cout, Cin*K*K)

        out = self.x_col @ W_2D.T             # (B*hout*wout, Cout)
        out += self.b.reshape((1,-1))
        out = out.reshape((B,h_out,w_out,-1)) # (B, hout, wout, Cout)
        out = np.transpose(out, (0,3,1,2))                  # (B, Cout, hout, wout)

        return out
    
        # very slow for loop implementation

        # initialize output
        # out = np.zeros((B,self.out_channels,h_out,w_out)) 

        # for b in range(B):
        #     for c_out in range(self.out_channels):
        #         for h_idx in range(h_out):
        #             h_start = h_idx*self.stride
        #             h_end = h_start + self.kernel_size
        #             for w_idx in range(w_out):
        #                 w_start = w_idx*self.stride
        #                 w_end = w_start + self.kernel_size
                        
        #                 weight = self.W[c_out] #(in_channels, kernel_size,kernel_size))
        #                 cur_x = x[b,:,h_start:h_end, w_start:w_end] #(1,c_in,kernel_size,kernel_size)
        #                 cur_out = np.sum(weight * cur_x) + self.b[c_out] # (sum the elementwise multiplication)
        #                 out[b,c_out,h_idx,w_idx] = cur_out
        

    def backward(self, dOut:np.ndarray):
        # C = A @ B, dA = dC @ B.T, dB = A.T @ dC
        # dOut (batch, out_chnl, h_out, w_out)
        # db (out_chnl), dw ( out_chnl, in_chnl, k, k)

        batch, out_chnl, h_out, w_out = dOut.shape
        dOut_reshaped = dOut.transpose((2, 3, 0, 1))  # (h_out, w_out, B, C_out)
        dOut_reshaped = dOut_reshaped.reshape(-1, out_chnl)  # (B*h_out*w_out, C_out)

        self.db = np.sum(dOut, axis=(0,2,3)) # Sum over batch, height, width
        
        W_2D = self.W.reshape((self.out_channels,-1)) # (Cout, Cin*K*K)
        
        dx_col = dOut_reshaped @ W_2D # (B*hout*wout, Cin*K*K)  = (B*hout*wout, Cout)    (Cout, Cin*K*K)
        dw_2D = self.x_col.T @ dOut_reshaped # (Cout, Cin*K*K) = (Cin*K*K, B*hout*wout) (B*hout*wout, Cout)
        

        dx = col2im(dx_col, self.x_shape, self.kernel_size, self.stride)
        self.dw = dw_2D.T.reshape(self.W.shape)

        return dx
    
        # very slow for loop 
        # self.db = np.sum(dOut, axis=(0,2,3))
        # dX = np.zeros_like(self.x)
        # for b in range(batch):
        #     for c_out in range(self.out_channels):
        #         for h_idx in range(h_out):
        #             h_start = h_idx*self.stride
        #             h_end = h_start + self.kernel_size
        #             for w_idx in range(w_out):
        #                 w_start = w_idx*self.stride
        #                 w_end = w_start + self.kernel_size
                        
        #                 self.dw[c_out] += dOut[b,c_out,h_idx,w_idx] * self.x[b,:,h_start:h_end, w_start:w_end] #(c_out,1,k,k)
        #                 # (1,in,k,k)                             (1)                           (1,in,k,k)
        #                 dX[b,:,h_start:h_end, w_start:w_end] += dOut[b,c_out,h_idx,w_idx] * self.W[c_out] 
        
    

class MaxPool2D:
    def __init__(self, kernel_size=2, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size

    def params(self):
        return []
    
    def forward(self,x):
        # cache x (batch, c, h, w)
        B,C,H,W = x.shape
        self.x_shape = x.shape

        # use // to avoid float division!
        h_out = (H - self.kernel_size) // self.stride + 1
        w_out = (W - self.kernel_size) // self.stride + 1

        # im2col
        self.x_col = im2col(x, self.kernel_size, self.stride) # (B*hout*wout, Cin*K*K)
        BHW, _  = self.x_col.shape

        x_col_reshaped = self.x_col.reshape((BHW, C, -1))
        out = np.max(x_col_reshaped, axis=2) # (BHW,C)
        self.max_idx = np.argmax(x_col_reshaped,axis=2)
        
        out = out.reshape((B,h_out,w_out,C))  # (B,H,W,C)
        out = out.transpose((0,3,1,2)) # (B,C,H,W)
        return out
    
        # im2col approach
        # out = np.zeros((B,C,h_out,w_out))
        # self.cache = np.zeros((B,C,h_out,w_out))

        # for batch in range(B):
        #     for cnl in range(C):
        #         for h in range(h_out):
        #             h_start = h*self.stride
        #             h_end = h_start + self.kernel_size
        #             for w in range(w_out):
        #                 w_start = w*self.stride
        #                 w_end = w_start + self.kernel_size
                        
        #                 x_patch = x[batch,cnl,h_start:h_end,w_start:w_end]
        #                 max_idx = np.argmax(x_patch) # 0 or 1 or 2 or 3
        #                 self.cache[batch,cnl,h,w] = max_idx

        #                 # row = max_idx // self.kernel_size
        #                 # col = max_idx - row * self.kernel_size
        #                 # x_pooled = x_patch[row][col]
        #                 out[batch,cnl,h,w] = np.max(x_patch)



    
    def backward(self, dOut: np.ndarray): 
        B,C,H_out,W_out = dOut.shape
       
        # dOut_reshaped =  dOut.reshape((B*H_out*W_out,C)) this is wrong!
        dOut_reshaped =  dOut.transpose((0, 2, 3, 1)).reshape((B*H_out*W_out,C))

        dx_col = np.zeros(((B*H_out*W_out), C, self.kernel_size**2))
        np.put_along_axis(dx_col,self.max_idx[:,:,None], dOut_reshaped[:,:,None], axis=2)

        dx_col = dx_col.reshape((B*H_out*W_out, -1))
        dx = col2im(dx_col,self.x_shape,self.kernel_size,self.stride)

        return dx

        # dX = np.zeros(self.x_shape)
        # for batch in range(B):
        #     for cnl in range(C):
        #         for h in range(H_out):
        #             h_start = h * self.stride
        #             for w in range(W_out):
        #                 w_start = w * self.stride

        #                 max_idx = int(self.cache[batch,cnl,h,w])
        #                 row = max_idx // self.kernel_size 
        #                 col = max_idx - row * self.kernel_size

        #                 cur_row = h_start+row
        #                 cur_col = w_start+col

        #                 dX[batch,cnl,cur_row,cur_col] += dOut[batch, cnl, h, w]
        
        # return dX
                        
class Flatten:
    # (b,c,h,w) --> (b, c*h*w)
    def params(self):
        return []
    
    def forward(self,x:np.ndarray):
        # x (b,c,h,w)
        B,C,H,W = x.shape
        self.x_shape = x.shape
        out = x.reshape((B,C*H*W))

        return out

    def backward(self, dOut:np.ndarray):
        # dOut (b, c*h*w)
        # dx (b,c,h,w) (need to flow to previous)
        B,C,H,W = self.x_shape
        dx = dOut.reshape(B,C,H,W)

        return dx



        