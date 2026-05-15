import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.embed_dim = embed_dim

        self.Wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wo = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attention_weights = None

    def forward(self, x):
        """
        Input:
            x (batch_size, nr_tokens, embed_dim)
        """
        q = self.Wq(x) # (batch_size, nr_tokens, embed_dim)
        k = self.Wk(x)
        v = self.Wv(x)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        self.attention_weights = F.softmax(scores, dim=-1)  # (B, T, T)

        out = torch.matmul(self.attention_weights, v)
        out = self.Wo(out)
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.Wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wo = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attention_weights = None

    def forward(self, x):
        """
        Input:
            x (batch_size, nr_tokens, embed_dim)
        """
        batch_size, nr_tokens, embed_dim = x.shape

        q = self.Wq(x).view(batch_size, nr_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.Wk(x).view(batch_size, nr_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.Wv(x).view(batch_size, nr_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  
        self.attention_weights = F.softmax(scores, dim=-1)

        out = torch.matmul(self.attention_weights, v) 
        out = out.transpose(1, 2).contiguous().view(batch_size, nr_tokens, embed_dim)
        out = self.Wo(out)
        return out