# new_script.py
import torch
import torch.nn as nn
from model import Mamba, ModelArgs, ResidualBlock, RMSNorm, MambaBlock

class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.dropout = nn.Dropout(dropout)
        self.head_dim = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
   
    @staticmethod     
    def attention(q, k, v, mask, dropout: nn.Dropout):
        head_dim = q.shape[-1]
        
        # (batch_size, num_heads, seq_len, head_dim) * (batch_size, num_heads, head_dim, seq_len) = (batch_size, num_heads, seq_len, seq_len
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(head_dim, dtype=torch.float32))
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            
        attention_scores = torch.softmax(attention_scores, dim=-1) # (batch_size, num_heads, seq_len, seq_len)
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
            
        return torch.matmul(attention_scores, v), attention_scores
            
        
    def forward(self, q, k, v, mask):
        querry = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        
        # Split the querry, key and value into num_heads
        # Shape: (batch_size, seq_len, num_heads, head_dim)
        querry = querry.view(querry.shape[0], -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(key.shape[0], -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(value.shape[0], -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (batch_size, num_heads, seq_len, head_dim)
        x, self.attention_scores = self.attention(querry, key, value, mask, self.dropout)
        
        # Concatenate the heads
        # Shape: (batch_size, seq_len, d_model)
        x = x.permute(0, 2, 1, 3).contiguous().view(x.shape[0], -1, self.d_model)
        
        return self.w_o(x)


class MambaEncoder(nn.Module):
    def __init__(self, args: ModelArgs):
        """Mamba encoder model."""
        super().__init__()
        self.args = args
        
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)


    def forward(self, input_ids):
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            logits: shape (b, l, d_model)
        """
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        
        return x

class MambaDecoder(nn.Module):
    def __init__(self, args: ModelArgs):
        """Full Mamba model."""
        super().__init__()
        self.args = args
        
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)
        self.cross_attention = MultiHeadAttentionBlock(args.d_model, args.num_heads)
        self.norm_f2 = RMSNorm(args.d_model)

    def forward(self, input_ids, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            logits: shape (b, l, vocab_size)
        """
        # print(input_ids.shape)
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        
        x = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        
        logits = self.norm_f2(x)

        return logits
    
class MambaProjection(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, x):
        return torch.log_softmax(self.lm_head(x), dim=-1)
    

class MambaSeq2Seq(nn.Module):
    def __init__(self, encoder: MambaEncoder, decoder: MambaDecoder, projection: MambaProjection):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.projection = projection
        
    def encode(self, src_input_ids):
        return self.encoder(src_input_ids)
    
    def decode(self, tgt_input_ids, encoder_output, src_mask=None, tgt_mask=None):
        return self.decoder(tgt_input_ids, encoder_output, src_mask=None, tgt_mask=None)
    
    def project(self, decoder_output):
        return self.projection(decoder_output)

def build_mambaseq2seq(config):
    # Initialize ModelArgs using the config
    args = ModelArgs(
        d_model=config['d_model'],
        n_layer=config['n_layer'],
        vocab_size=config['vocab_size'],  # Assuming the same vocab size for src and tgt
        d_state=config['d_state'],
        expand=config['expand'],
        dt_rank=config['dt_rank'],
        d_conv=config['d_conv'],
        pad_vocab_size_multiple=config['pad_vocab_size_multiple'],
        conv_bias=config['conv_bias'],
        bias=config['bias'],
        num_heads=config['num_heads'],
    )

    # Create Mamba encoder and decoder
    encoder = MambaEncoder(args)
    decoder = MambaDecoder(args)
    projection = MambaProjection(args.d_model, args.vocab_size)

    # Assemble them into a Seq2Seq model
    mamba_seq2seq = MambaSeq2Seq(encoder, decoder, projection)

    # Initialize parameters (optional, based on your preference)
    for p in mamba_seq2seq.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return mamba_seq2seq
