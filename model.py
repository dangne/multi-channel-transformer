import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, constant_


class PositionalEncoding(nn.Module):
    def __init__(self, num_cycles, num_points, embed_dim, dropout=0.):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(num_cycles*num_points, embed_dim)
        position = torch.arange(0, num_cycles*num_points, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).view(-1, num_cycles, num_points, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

class CrossCycleSelfAttention(nn.Module):
    def __init__(self, num_cycles, num_points, embed_dim, dropout=0.):
        super(CrossCycleSelfAttention, self).__init__()
        self.num_cycles = num_cycles
        self.num_points = num_points
        self.embed_dim = embed_dim
        self.attn_weight = nn.Parameter(torch.empty(num_cycles, num_points, embed_dim))
        self.q_proj_weight = nn.Parameter(torch.empty(num_cycles, embed_dim, embed_dim))
        self.k_proj_weight = nn.Parameter(torch.empty(num_cycles, embed_dim, embed_dim))
        self.v_proj_weight = nn.Parameter(torch.empty(num_cycles, embed_dim, embed_dim))
        self.q_proj_bias = nn.Parameter(torch.empty(num_cycles, 1, embed_dim))
        self.k_proj_bias = nn.Parameter(torch.empty(num_cycles, 1, embed_dim))
        self.v_proj_bias = nn.Parameter(torch.empty(num_cycles, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def _init_parameters(self):
        xavier_normal_(self.attn_weight)
        xavier_normal_(self.q_proj_weight)
        xavier_normal_(self.k_proj_weight)
        xavier_normal_(self.v_proj_weight)
        constant_(self.q_proj_bias, 0.)
        constant_(self.k_proj_bias, 0.)
        constant_(self.v_proj_bias, 0.)

    def forward(self, query):
        attn_context_weights = self.attn_weight * query
        context = attn_context_weights.sum(dim=-3, keepdim=True) - attn_context_weights
        q = F.relu(query @ self.q_proj_weight + self.q_proj_bias)
        k = F.relu(context @ self.k_proj_weight + self.k_proj_bias)
        v = F.relu(context @ self.v_proj_weight + self.v_proj_bias)
        q = q * (float(self.embed_dim) ** -0.5)
        attn_output_weights = F.softmax(q @ k.transpose(-2, -1), dim=-1)
        attn_output_weights = self.dropout(attn_output_weights)
        out = attn_output_weights @ v
        assert out.shape == query.shape
        return out 


class CycleWiseSelfAttention(nn.Module):
    def __init__(self, num_cycles, embed_dim, dropout=0.):
        super(CycleWiseSelfAttention, self).__init__()
        self.num_cycles = num_cycles 
        self.embed_dim = embed_dim

        self.q_proj_weight = nn.Parameter(torch.empty(num_cycles, embed_dim, embed_dim))
        self.k_proj_weight = nn.Parameter(torch.empty(num_cycles, embed_dim, embed_dim))
        self.v_proj_weight = nn.Parameter(torch.empty(num_cycles, embed_dim, embed_dim))
        self.q_proj_bias = nn.Parameter(torch.empty(num_cycles, 1, embed_dim))
        self.k_proj_bias = nn.Parameter(torch.empty(num_cycles, 1, embed_dim))
        self.v_proj_bias = nn.Parameter(torch.empty(num_cycles, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        self._init_parameters()

    def _init_parameters(self):
        xavier_normal_(self.q_proj_weight)
        xavier_normal_(self.k_proj_weight)
        xavier_normal_(self.v_proj_weight)
        constant_(self.q_proj_bias, 0.)
        constant_(self.k_proj_bias, 0.)
        constant_(self.v_proj_bias, 0.)
    
    def forward(self, query, key, value):
        # query.shape = batch_size x num_cycles x num_points x embed_dim
        q = F.relu(query @ self.q_proj_weight + self.q_proj_bias)
        k = F.relu(key @ self.k_proj_weight + self.k_proj_bias)
        v = F.relu(value @ self.v_proj_weight + self.v_proj_bias)
        q = q * (float(self.embed_dim) ** -0.5)
        attn_output_weights = F.softmax(q @ k.transpose(-2, -1), dim=-1)
        attn_output_weights = self.dropout(attn_output_weights)
        out = attn_output_weights @ v
        assert out.shape == query.shape
        return out 


class MCTransformer(nn.Module):
    def __init__(self, num_cycles, num_points, in_embed_dim, out_embed_dim, dropout=0, layer_norm_eps=1e-5):
        super(MCTransformer, self).__init__()
        self.num_cycles = num_cycles
        self.num_points = num_points
        self.in_embed_dim = in_embed_dim
        self.out_embed_dim = out_embed_dim

        self.in_proj_weight = nn.Parameter(torch.empty(num_cycles, in_embed_dim, out_embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(num_cycles, num_points, 1))
        self.pos_encoder = PositionalEncoding(num_cycles, num_points, out_embed_dim)

        self.cwa = CycleWiseSelfAttention(num_cycles, out_embed_dim, dropout)
        self.cca = CrossCycleSelfAttention(num_cycles, num_points, out_embed_dim, dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(out_embed_dim, layer_norm_eps)
        self.norm2 = nn.LayerNorm(out_embed_dim, layer_norm_eps)

        self._init_parameters()
        
    def _init_parameters(self):
        xavier_normal_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.)

    def forward(self, x):
        x = F.relu(x @ self.in_proj_weight + self.in_proj_bias)
        x = self.pos_encoder(x)

        x2 = self.cwa(x, x, x)
        x = x + self.dropout1(x2)
        x = self.norm1(x)

        x2 = self.cca(x)
        x = x + self.dropout2(x2)
        x = self.norm2(x)

        return x

    
class Regressor(nn.Module):
    def __init__(self, num_labels, num_cycles, num_points, in_embed_dim, out_embed_dim, hidden_dim, dropout=0.):
        super(Regressor, self).__init__()
        self.encoder = MCTransformer(num_cycles, num_points, in_embed_dim, out_embed_dim, dropout)
        self.w1 = nn.Parameter(torch.empty(num_points * out_embed_dim, hidden_dim))
        self.w2 = nn.Parameter(torch.empty(hidden_dim, num_labels))
        self.b1 = nn.Parameter(torch.empty(hidden_dim))
        self.b2 = nn.Parameter(torch.empty(num_labels))
        self.dropout = nn.Dropout(dropout)
    
    def _init_parameters(self):
        xavier_normal_(self.w1)
        xavier_normal_(self.w2)
        constant_(self.b1, 0.)
        constant_(self.b2, 0.)

    def forward(self, x):
        x = self.encoder(x)
        # Only use the embedding of the latest cycle for prediction
        x = torch.flatten(x[:, -1], start_dim=1)  
        x = self.dropout(F.relu(x @ self.w1 + self.b1))
        x = x @ self.w2 + self.b2

        return x