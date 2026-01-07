from typing import Any
import torch
import torch.nn as nn


class PositionalEncoding():
    def __init__(self, max_len, d_model):
        pass
    
    def __call__(self,x):
        pass


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        pass
    
    def forward(self, q,k,v, mask = None):
        pass

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        pass
    
    def forward(self, x):
        pass

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):  
        pass

    def forward(self,x, mask = None):
        pass

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):  
        pass

    def forward(self,x, encoder_output, src_mask=None, tgt_mask=None):
        pass


class Encoder(nn.Module):
    def __init__(self, n_block, d_model, num_heads, d_ff, dropout): 
        pass

    def forward(self,x, mask = None):
        pass


class Decoder(nn.Module):
    def __init__(self, n_block, d_model, num_heads, d_ff, dropout): 
        pass

    def forward(self,x, encoder_output, src_mask=None, tgt_mask=None):
        pass


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size, 
                 tgt_vocab_size, 
                 position_enc_max_len=5000,
                 n_block=6, 
                 d_model=512, 
                 num_heads=8, 
                 d_ff=2048,
                 dropout=0.1
                ): 
        
        super(Transformer,self).__init__()

        self.enc_embedding = nn.Embedding(src_vocab_size,d_model)
        self.dec_embedding = nn.Embedding(tgt_vocab_size,d_model)

        self.enc_positional_encoding = PositionalEncoding(position_enc_max_len, d_model)
        self.dec_positional_encoding = PositionalEncoding(position_enc_max_len, d_model)

        self.encoder = Encoder(n_block, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(n_block, d_model, num_heads, d_ff, dropout)

        self.projection = nn.Linear(d_model, tgt_vocab_size)

    
    def forward(self, src, tgt, src_mask = None, tgt_mask = None):
        # src: (batch, src_seq_len)
        # tgt: (batch, tgt_seq_len)
        
        # encoder part
        src = self.enc_embedding(src)            # (batch, src_seq_len, d_model)
        src = self.enc_positional_encoding(src)  # (batch, src_seq_len, d_model)
        src = self.encoder(src, mask = src_mask) # (batch, src_seq_len, d_model)

        # decoder part
        tgt = self.dec_embedding(tgt)            # (batch, src_seq_len, d_model)
        tgt = self.dec_positional_encoding(tgt)  # (batch, src_seq_len, d_model)
        tgt = self.decoder(tgt, src, src_mask, tgt_mask)  # (batch, src_seq_len, d_model)
        
        tgt = self.projection(tgt)

        return tgt



        