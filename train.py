import yaml
import torch
import torch.nn as nn
import datasets

from utils import get_or_build_tokenizer
from dataset import SequenceTranslationDataset
from model import Mamba, ModelArgs

import os

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_filename']
    model_filename = f"{model_basename}{epoch}.pt"
    return os.path.join(model_folder, model_filename)

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
    
    ds_train = SequenceTranslationDataset(ds_train_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    ds_val = SequenceTranslationDataset(ds_val_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    
    # Define the model
    train_dataloader = torch.utils.data.DataLoader(ds_train, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(ds_val, batch_size=1, shuffle=False)
    
    # Initialize ModelArgs using the loaded config
    args = ModelArgs(
        d_model=config['d_model'],
        n_layer=config['n_layer'],
        vocab_size=vocab_size_tgt,
        d_state=config['d_state'],
        expand=config['expand'],
        dt_rank=config['dt_rank'],
        d_conv=config['d_conv'],
        pad_vocab_size_multiple=config['pad_vocab_size_multiple'],
        conv_bias=config['conv_bias'],
        bias=config['bias']
    )
    model = Mamba(args)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-09)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"), label_smoothing=0.1)
    
    initial_epoch = 0
    global_step = 0
    model.train()
    
    for epoch in range(initial_epoch, config['num_epochs']):
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            input_sequence = batch['input_sequence'].to(device)
            input_mask = batch['input_mask'].to(device)
            target_sequence = batch['target_sequence'].to(device)
            
            logits = model(input_sequence)
            loss = criterion(logits.view(-1, logits.size(-1)), target_sequence.view(-1))
            loss.backward()
            optimizer.step()
            
            global_step += 1
            
            if global_step % config['log_interval'] == 0:
                print(f"Epoch: {epoch}, Global step: {global_step}, Loss: {loss.item()}")
                
        # Save the model checkpoint
        model_filename = get_weights_file_path(config, epoch)
        print("Saving model to: ", model_filename)
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'optimizer': optimizer.state_dict(),
            'model_state_dict': model.state_dict()
        }, model_filename)
                
                
if __name__ == "__main__":
    train()