import nltk
from nltk.tokenize import word_tokenize
import os
import json

# Ensure 'punkt' model is downloaded
nltk.download('punkt')

class NLTKTokenizer:
    def __init__(self, config=None):
        self.vocab = {}
        self.config = config or {}
        self.pad_id = self.vocab["pad"]  # Store pad_id for future use

    @classmethod
    def from_pretrained(cls, pretrained_path):
        """Load a tokenizer with a pre-built vocabulary from a saved file."""
        vocab_file = os.path.join(pretrained_path, "vocab.json")
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"Vocabulary file not found at {vocab_file}")
        
        with open(vocab_file, "r") as f:
            vocab = json.load(f)
        tokenizer = cls()
        tokenizer.vocab = vocab
        tokenizer.pad_id = vocab.get("PAD", 1)  # Ensure pad_id is set
        return tokenizer

    def build_vocab(self):
        """Build vocabulary from the given dataset."""
        from collections import Counter
        from datasets import load_dataset

        dataset = load_dataset(self.config["dataset"])
        train_dataset = dataset['train']
        vocab = Counter(self.vocab)
        for item in train_dataset:
            tokens = word_tokenize(item['text'].lower())
            vocab.update(tokens)
        self.vocab = {word: idx for idx, (word, _) in enumerate(vocab.items(), 1)}  # Index starts at 1
        #print(self.vocab)

    def tokenize(self, text):
        """Tokenize a given text using NLTK."""
        tokens = word_tokenize(text.lower())  # Tokenize the text into words
        token_ids = [self.vocab.get(token, self.vocab["UNK"]) for token in tokens]  # Get token IDs
        
        # token_ids = [self.vocab.get(token, np.zeros_like(self.vocab['UNK'])) for token in tokens]  # Get token IDs

        return {"tokens": tokens, "ids": token_ids}
    
    def save(self, folder_path):
        """Save the vocabulary to a file."""
        os.makedirs(folder_path, exist_ok=True)
        with open(os.path.join(folder_path, "vocab.json"), "w") as f:
            json.dump(self.vocab, f)
