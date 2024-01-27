from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import os
import torch

def get_all_sentences(ds, lang):
    for example in ds:
        yield example['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    # config['tokenizer_path'] = 'tokenizer_{}.json'
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            # vocab_size=config['vocab_size'],
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_filename']
    model_filename = f"{model_basename}{epoch}.pt"
    return os.path.join(model_folder, model_filename)

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_src.token_to_id("[SOS]")
    eos_idx = tokenizer_src.token_to_id("[EOS]")
    
    # Initialize the decoder input with the SOS token
    input_sequence = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device) # (batch_size, seq_len)
    while True:
        if decoder_input.size(1) == max_len:
            break
        
        out = model(input_sequence)
        
        # Get the next token
        prob = out[:,-1]
        
        # Select the token with max probability
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1)
        
        if next_word == eos_idx:
            break
        
    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0
    
    source_texts = []
    expected = []
    predicted = []
    
    # Size of the control window
    console_width = 80
    
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device) # (batch_size, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch_size, seq_len)
            
            assert encoder_input.size(0) == 1 # Only one example at a time
            
            model_output = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_output_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())
            
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_output_text)
            
            # Print to console
            print_msg("-"*console_width)
            print_msg(f"Source: {source_text}")
            print_msg(f"Expected: {target_text}")
            print_msg(f"Predicted: {model_output_text}")
            
            if count == num_examples:
                break
            
        if writer:
            pass # TODO: TorchMetrics CharErrorRate, BLEU, WordErrorRate


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

def validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device) # (batch_size, seq_len)

            assert encoder_input.size(0) == 1 # Only one example at a time

            # Adjust greedy_decode for Mamba model
            model_output = greedy_decode(model, encoder_input, tokenizer_src, tokenizer_tgt, max_len, device)
            
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_output_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_output_text)

            print("-"*console_width)
            print(f"Source: {source_text}")
            print(f"Expected: {target_text}")
            print(f"Predicted: {model_output_text}")
            
            if count == num_examples:
                break
            
            
def causal_mask(seq_len):
    mask = torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1).type(torch.int)
    return mask == 0