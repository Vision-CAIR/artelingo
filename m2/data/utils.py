import contextlib, sys
from transformers import AutoTokenizer

class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout

def reporthook(t):
    """https://github.com/tqdm/tqdm"""
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        b: int, optionala
        Number of blocks just transferred [default: 1].
        bsize: int, optional
        Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
        Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return inner

def get_tokenizer(tokenizer):
    if callable(tokenizer):
        return tokenizer
    if tokenizer == "spacy":
        try:
            import spacy
            spacy_en = spacy.load('en')
            return lambda s: [tok.text for tok in spacy_en.tokenizer(s)]
        except ImportError:
            print("Please install SpaCy and the SpaCy English tokenizer. "
                  "See the docs at https://spacy.io for more information.")
            raise
        except AttributeError:
            print("Please install SpaCy and the SpaCy English tokenizer. "
                  "See the docs at https://spacy.io for more information.")
            raise
    elif tokenizer == "moses":
        try:
            from nltk.tokenize.moses import MosesTokenizer
            moses_tokenizer = MosesTokenizer()
            return moses_tokenizer.tokenize
        except ImportError:
            print("Please install NLTK. "
                  "See the docs at http://nltk.org for more information.")
            raise
        except LookupError:
            print("Please install the necessary NLTK corpora. "
                  "See the docs at http://nltk.org for more information.")
            raise
    elif tokenizer == 'revtok':
        try:
            import revtok
            return revtok.tokenize
        except ImportError:
            print("Please install revtok.")
            raise
    elif tokenizer == 'subword':
        try:
            import revtok
            return lambda x: revtok.tokenize(x, decap=True)
        except ImportError:
            print("Please install revtok.")
            raise
    raise ValueError("Requested tokenizer {}, valid choices are a "
                     "callable that takes a single string as input, "
                     "\"revtok\" for the revtok reversible tokenizer, "
                     "\"subword\" for the revtok caps-aware tokenizer, "
                     "\"spacy\" for the SpaCy English tokenizer, or "
                     "\"moses\" for the NLTK port of the Moses tokenization "
                     "script.".format(tokenizer))

class HuggingfaceVocab(object):
    special_tokens_mapper = {
        'sos': '<s>',
        'eos': '</s>',
        'unk': '<unk>',
        'pad_token_id': '<pad>',
    }
    def __init__(self, file_name="dataset/custom-xlmroberta-tokenizer-60k"):
        self.tokenizer = AutoTokenizer.from_pretrained(file_name, padding_side='right')
        self.initialize_special_tokens()
        self.initialize_stoi()
        
    def initialize_stoi(self):
        self.stoi = {
            '<pad>': self.pad_token_id,
            '<eos>': self.eos,
            '<bos>': self.sos,
            '<unk>': self.unk,
        }
            
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
                            return_attention_mask=False, return_tensors='pt')['input_ids']
    
    def encode(self, text, max_len, add_begin_end=True):
        text = ' '.join(text)
        encoded = self.tokenize(self, text, max_len, add_begin_end=add_begin_end)
        return encoded
    
    def decode(self, tokens):
        return self.tokenizer.convert_ids_to_tokens(tokens)
        
    def decode_print(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def batch_decode_print(self, tokens):
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
    
    def __iter__(self):
        return iter(self.tokenizer.vocab)
    
    def save(self, file_name):
        self.tokenizer.save_pretrained(file_name)
        
    @classmethod
    def load(cls, file_name):
        return cls(file_name)