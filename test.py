import torch
import datasets
from utils import get_or_build_tokenizer, get_weights_file_path
from dataset import BilingualDataset
from modeltr import build_mambaseq2seq

import yaml
import os

def greedy_decode(model, source, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_src.token_to_id("[SOS]")
    eos_idx = tokenizer_src.token_to_id("[EOS]")
    
    # Precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source) # (batch_size, seq_len, d_model)
    
    # Initialize the decoder input with the SOS token
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device) # (batch_size, seq_len)
    while True:
        if decoder_input.size(1) == max_len:
            break
        
        # Build mask for target 
        # decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device) # (batch_size, seq_len, seq_len)

        out = model.decode(decoder_input, encoder_output) # (batch_size, seq_len, d_model)
        
        # Get the next token
        prob = model.project(out[:,-1])
        
        # Select the token with max probability
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1)
        
        if next_word == eos_idx:
            break
        
    return decoder_input.squeeze(0)

def load_model(model_path, config, device):
    model = build_mambaseq2seq(config)
    model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])
    model.to(device)
    return model

def test(model, test_ds, tokenizer_src, tokenizer_tgt, max_len, device):
    model.eval()

    with torch.no_grad():
        for batch in test_ds:
            encoder_input = batch['encoder_input'].to(device)

            model_output = greedy_decode(model, encoder_input, tokenizer_src, tokenizer_tgt, max_len, device)
            
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_output_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())

            print("Source:", source_text)
            print("Expected:", target_text)
            print("Predicted:", model_output_text)
            print("-" * 80)

if __name__ == "__main__":
    device = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    # Load config file
    config_path = 'config.yml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Load the model
    model_path = get_weights_file_path(config, '2')  # replace 'latest' with your model's epoch
    model = load_model(model_path, config, device)

    # Prepare the test dataset
    ds_raw = datasets.load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
    
    ds_test = BilingualDataset(ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    test_dataloader = torch.utils.data.DataLoader(ds_test, batch_size=1, shuffle=False)

    # Run test
    test(model, test_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device)
