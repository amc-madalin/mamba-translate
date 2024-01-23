# new_script.py
import torch
import torch.nn as nn
from model import Mamba, ModelArgs, ResidualBlock, RMSNorm, MambaBlock

def main():
    # Example configuration
    config = {
        'd_model': 512,
        'n_layer': 6,
        'vocab_size': 10000,
        'd_state': 16,
        'expand': 2,
        'dt_rank': 'auto',
        'd_conv': 4,
        'pad_vocab_size_multiple': 8,
        'conv_bias': True,
        'bias': False
    }

    # Initialize ModelArgs using the config
    args = ModelArgs(
        d_model=config['d_model'],
        n_layer=config['n_layer'],
        vocab_size=config['vocab_size'],
        d_state=config['d_state'],
        expand=config['expand'],
        dt_rank=config['dt_rank'],
        d_conv=config['d_conv'],
        pad_vocab_size_multiple=config['pad_vocab_size_multiple'],
        conv_bias=config['conv_bias'],
        bias=config['bias']
    )

    # Create the Mamba model
    model = Mamba(args)


class MambaEncoder(nn.Module):
    def __init__(self, args: ModelArgs):
        """Mamba encoder model."""
        super().__init__()
        self.args = args
        
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)

        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # Tie output projection to embedding weights.
                                                     # See "Weight Tying" paper


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

    def forward(self, input_ids, encoder_output):
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            logits: shape (b, l, vocab_size)
        """
        # print(input_ids.shape)
        x = self.embedding(input_ids)
        x = x.permute(0, 2, 1)
        encoder_output = encoder_output.permute(0, 2, 1)
        # print(x.shape)
        # print(encoder_output.shape)
        # exit()
        

        
        x = x * encoder_output
        
        x = x.permute(0, 2, 1)
        
        for layer in self.layers:
            x = layer(x)
            
        logits = self.norm_f(x)

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
    
    def decode(self, tgt_input_ids, encoder_output):
        return self.decoder(tgt_input_ids, encoder_output)
    
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
        bias=config['bias']
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
