import yaml
import torch
import torch.nn as nn
import datasets

from utils import get_or_build_tokenizer, get_weights_file_path
from dataset import BilingualDataset, causal_mask
from modeltr import build_mambaseq2seq

import os

def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)
    
    # print(f"torch backend MPS is available? {torch.backends.mps.is_available()}")
    # print(f"current PyTorch installation built with MPS activated? {torch.backends.mps.is_built()}")
    # print(f"check the torch MPS backend: {torch.device('mps')}")
    # print(f"test torch tensor on MPS: {torch.tensor([1,2,3], device='mps')}")
    
    # Load config file
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)
        
    # Make model folder
    model_folder = config['model_folder']
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    
    # Define the dataset
    ds_raw = datasets.load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    
    # Build or load tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
    
    # Assuming tokenizer_src and tokenizer_tgt are already initialized
    vocab_size_src = len(tokenizer_src.get_vocab())
    vocab_size_tgt = len(tokenizer_tgt.get_vocab())
    
    # Keep 90% for training and 10% for validation
    train_size = int(len(ds_raw) * 0.9)
    val_size = len(ds_raw) - train_size
    ds_train_raw, ds_val_raw = torch.utils.data.random_split(ds_raw, [train_size, val_size])
    
    ds_train = BilingualDataset(ds_train_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    ds_val = BilingualDataset(ds_val_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    
    # Define the model
    train_dataloader = torch.utils.data.DataLoader(ds_train, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(ds_val, batch_size=1, shuffle=False)
    
    # Initialize model using the loaded config
    model = build_mambaseq2seq(config)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-09)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"), label_smoothing=0.1)
    
    initial_epoch = 0
    global_step = 0
    if config['preload'] is not None:
        model_filename = get_weights_file_path(config, config['preload'])
        print("Loading weights from: ", model_filename)
        state = torch.load(model_filename)
        initial_epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer'])
        global_step = state['global_step']
        
    model.train()
    
    for epoch in range(initial_epoch, config['num_epochs']):
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            encoder_input = batch['encoder_input'].to(device) # (batch_size, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (batch_size, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch_size, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (batch_size, seq_len)
            
            
            
            encoder_output = model.encode(encoder_input)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
            projection_output = model.project(decoder_output)
            
            label = batch['label'].to(device) # (batch_size, seq_len)
            loss = criterion(projection_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            
            loss.backward()
            optimizer.step()
            
            global_step += 1
            
            if global_step % config['log_interval'] == 0:
                print(f"Epoch: {epoch}, Global step: {global_step}, Loss: {loss.item()}")
                
            # validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device)
            
            # model.train()
            
                
            save_epoch = 4
            # Save the model checkpoint
            model_filename = get_weights_file_path(config, save_epoch)
            print("Saving model to: ", model_filename)
            torch.save({
                'epoch': save_epoch,
                'global_step': global_step,
                'optimizer': optimizer.state_dict(),
                'model_state_dict': model.state_dict()
            }, model_filename)
                
                
if __name__ == "__main__":
    train()