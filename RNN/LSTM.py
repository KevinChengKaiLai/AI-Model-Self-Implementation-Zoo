# LSTM.py -- Long Short Term Memory
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn

'''
LSTM has 4 gates: forget (f), input (i), candidate (g), output (o)

'''

class LSTM_classification(nn.Module):
    def __init__(self,in_dim,hidden_size):
        super(LSTM_classification,self).__init__()
        self.in_dim = in_dim
        self.hidden_size = hidden_size

        # size of concat [h,x]
        input_size = in_dim + hidden_size

        self.forget_gate = nn.Linear(input_size,hidden_size)
        self.input_gate = nn.Linear(input_size,hidden_size)
        self.candidate_gate = nn.Linear(input_size,hidden_size)
        self.output_gate = nn.Linear(input_size,hidden_size)


    def forward(self,x:torch.Tensor):
        # x (batch, seq_len, indim)
        batch_size, seq_len, _ = x.size()

        # Cell State - Long Term Memory (Carry information to future timestep)
        c = torch.zeros((batch_size,self.hidden_size))
        # Hidden State - Short Term Memory (Prepare for LSTM output)
        h = torch.zeros((batch_size,self.hidden_size))

        for t in range(seq_len):
            x_t = x[:,t,:] # (batch, in_dim)

            # Combine context + new input
            combined = torch.cat((h,x_t),dim=1) # (batch, in_dim + hidden_size)


            # Why sigmoid? Range [0,1] acts like a percentage

            # Forget Gate: decide what to forget in LONG term Memory (Cell)
            f_t = torch.sigmoid(self.forget_gate(combined))  # (batch, in_dim + hidden_size)(in_dim + hidden_size, hidden_size)
            
            # Input Gate: decide how many new information go into LONG term Memory (Cell)
            i_t = torch.sigmoid(self.input_gate(combined))   # (batch, in_dim + hidden_size)(in_dim + hidden_size, hidden_size)
            # Candidate Gate: Represent the information from xt

            # Tanh [-1,1] so model can differentiate if this info is positive or negative
            # Purpose: Create the candidate content to add
            g_t = torch.tanh(self.candidate_gate(combined))  # (batch, in_dim + hidden_size)(in_dim + hidden_size, hidden_size)
            
            # Output Gate: Decide how many memory to reveal in this timesetp output 
            o_t = torch.sigmoid(self.output_gate(combined))  # (batch, in_dim + hidden_size)(in_dim + hidden_size, hidden_size)

            c = f_t * c + i_t * g_t 
            # tanh here is for normalization to [-1,1]
            # Purpose: Normalize the cell state before outputting
            # No need to add h_t-1 since Hidden state h_t is freshly computed from c_t (no previous h_{t-1})
            h = o_t * torch.tanh(c)

        return h
    

if __name__ == '__main__':
    print("Testing LSTM implementation...")
    
    # Test parameters
    batch_size = 4
    seq_len = 10
    in_dim = 64
    hidden_size = 128
    
    # Create LSTM
    lstm = LSTM_classification(in_dim=in_dim, hidden_size=hidden_size)
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, in_dim)
    
    # Forward pass
    h_final = lstm(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {h_final.shape}")
    print(f"Expected: ({batch_size}, {hidden_size})")
    
    # Verify shape
    assert h_final.shape == (batch_size, hidden_size), "Shape mismatch!"
    
    # Check for NaN or Inf (gradient issues)
    assert not torch.isnan(h_final).any(), "Output contains NaN!"
    assert not torch.isinf(h_final).any(), "Output contains Inf!"
    
    print("✓ All tests passed! LSTM is working correctly!")

