"""
Keep track of a certain word vocabulary associated with a linguistic dataset.

The MIT License (MIT)
Originally created by Panos Achlioptas mid 2019, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab
"""

import pickle
from collections import Counter
from transformers import AutoTokenizer

PAD = '<pad>'
SOS = '<sos>'
EOS = '<eos>'
UNK = '<unk>'
SPECIAL_SYMBOLS = [PAD, SOS, EOS, UNK]

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self, special_symbols=None):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.intialize_special_symbols()

    def intialize_special_symbols(self):
        # Register special-symbols
        self.special_symbols = SPECIAL_SYMBOLS

        # Map special-symbols to ints
        for s in self.special_symbols:
            self.add_word(s)

        # Add special symbols as attributes for quick access
        for s in self.special_symbols:
            name = s.replace('<', '')
            name = name.replace('>', '')
            setattr(self, name, self(s))
        self.pad_token_id = self.pad

    def n_special(self):
        return len(self.special_symbols)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx[UNK]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def encode(self, text, max_len=None, add_begin_end=True):
        """
        :param text: (list) of tokens ['a', 'nice', 'sunset']
        :param max_len:
        :param add_begin_end:
        :return: (list) of encoded tokens.
        """
        encoded = [self(token) for token in text]
        if max_len is not None:
            encoded = encoded[:max_len]  # crop if too big

        if add_begin_end:
            encoded = [self(SOS)] + encoded + [self(EOS)]

        if max_len is not None:  # pad if too small (works because [] * [negative] does nothing)
            encoded += [self(PAD)] * (max_len - len(text))
        return encoded

    def decode(self, tokens):
        return [self.idx2word[token] for token in tokens]

    def decode_print(self, tokens):
        exclude = set([self.word2idx[s] for s in [SOS, EOS, PAD]])
        words = [self.idx2word[token] for token in tokens if token not in exclude]
        return ' '.join(words)

    def __iter__(self):
        return iter(self.word2idx)

    def save(self, file_name):
        """ Save as a .pkl the current Vocabulary instance.
        :param file_name:  where to save
        :return: None
        """
        with open(file_name, mode="wb") as f:
            pickle.dump(self, f, protocol=2)  # protocol 2 => works both on py2.7  and py3.x

    @staticmethod
    def load(file_name):
        """ Load a previously saved Vocabulary instance.
        :param file_name: where it was saved
        :return: Vocabulary instance.
        """
        with open(file_name, 'rb') as f:
            vocab = pickle.load(f)
        return vocab


def build_vocab(token_list, min_word_freq):
    """Build a simple vocabulary wrapper."""

    counter = Counter()
    for tokens in token_list:
        counter.update(tokens)

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= min_word_freq]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)

    return vocab

class HuggingfaceVocab(object):
    special_tokens_mapper = {
        'sos': '<s>',
        'eos': '</s>',
        'unk': '<unk>',
        'pad_token_id': '<pad>',
    }
    def __init__(self, file_name="dataset/custom-xlmroberta-tokenizer-60k/"):
        self.tokenizer = AutoTokenizer.from_pretrained(file_name, padding_side='right')
        self.initialize_special_tokens()
            
    def word2idx(self):
        return self.tokenizer.vocab
    
    def initialize_special_tokens(self):
        self.special_symbols = self.tokenizer.all_special_tokens
        for k, v in self.special_tokens_mapper.items():
            setattr(self, k, self.tokenizer.convert_tokens_to_ids(v))
    
    def n_special(self):
        return len(self.special_symbols)
    
    def add_word(self, word):
        self.tokenizer.add_tokens(word)
        
    def __call__(self, word):
        return self.tokenizer.convert_tokens_to_ids(word)
    
    def __len__(self):
        return len(self.tokenizer)
    
    def tokenize(self, text, max_len, add_begin_end=True):
        return self.tokenizer(text, add_special_tokens=add_begin_end, max_length=max_len,
                            truncation=True, padding='max_length',
                            return_attention_mask=False)['input_ids']
    
    def encode(self, text, max_len, add_begin_end=True):
        text = ' '.join(text)
        encoded = self.tokenize(self, text, max_len, add_begin_end=add_begin_end)
        return encoded
    
    def decode(self, tokens):
        return self.tokenizer.convert_ids_to_tokens(tokens)
        
    def decode_print(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def __iter__(self):
        return iter(self.tokenizer.vocab)
    
    def save(self, file_name):
        self.tokenizer.save_pretrained(file_name)
        
    @classmethod
    def load(cls, file_name):
        return cls(file_name)