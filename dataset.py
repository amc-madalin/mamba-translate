import torch
from torch.utils.data import Dataset

class SequenceTranslationDataset(Dataset):
    """
    Dataset class for sequence-to-sequence translation tasks, specifically designed for the Mamba architecture.
    
    Attributes:
    - ds: Dataset containing source-target pairs.
    - tokenizer_src: Tokenizer for the source language.
    - tokenizer_tgt: Tokenizer for the target language.
    - src_lang: Source language identifier.
    - tgt_lang: Target language identifier.
    - seq_len: Maximum sequence length.
    - sos_token, eos_token, pad_token: Special tokens for start, end, and padding.
    """

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        
        # Define special tokens
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        """
        Returns a single data pair (source and target sequence) from the dataset.

        Args:
        - idx: Index of the data pair in the dataset.

        Returns:
        - Dictionary containing:
        - 'input_sequence': Tokenized and padded source sequence.
        - 'input_mask': Mask for the source sequence.
        - 'target_sequence': Tokenized and padded target sequence.
        - 'src_text': Original source text.
        - 'tgt_text': Original target text.
        """
        # Extract source and target texts from the dataset
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Tokenize source and target texts
        src_tokens = self.tokenizer_src.encode(src_text).ids
        tgt_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Calculate the number of padding tokens needed for source and target
        src_padding = self.seq_len - len(src_tokens) - 2
        tgt_padding = self.seq_len - len(tgt_tokens) - 1

        # Check if the sequence length is too short
        if src_padding < 0 or tgt_padding < 0:
            raise ValueError("Sequence length is too short for the given text")

        # Prepare the input sequence by concatenating special tokens and padding
        input_sequence = torch.cat([
            self.sos_token,
            torch.tensor(src_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * src_padding, dtype=torch.int64)
        ])

        # Prepare the target sequence with padding
        target_sequence = torch.cat([
            torch.tensor(tgt_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * tgt_padding, dtype=torch.int64)
        ])

        # Ensure the input and target sequences are of the expected sequence length
        assert input_sequence.shape == target_sequence.shape == (self.seq_len,), "Inconsistent sequence lengths"

        return {
            'input_sequence': input_sequence,
            'input_mask': (input_sequence != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            'target_sequence': target_sequence,
            'target_mask': (target_sequence != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            'src_text': src_text,
            'tgt_text': tgt_text
        }
