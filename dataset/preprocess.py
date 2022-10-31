import numpy as np
import pandas as pd
import unicodedata
import os
import shutil
import os.path as osp
from artemis.emotions import emotion_to_int, language_to_int
from artemis.utils.vocabulary import HuggingfaceVocab
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--raw_data', type=str, default='raw/artelingo.csv')
parser.add_argument('--wikiart_dir', type=str, default='wikiart/')
args = parser.parse_args()

df = pd.read_csv(args.raw_data)
MAX_LEN = 46
languages = ['english', 'arabic', 'chinese']
dfs = {}
for l in languages:
    dfs[l] = df[df['language']==l].copy()
    dfs[l]['language'] = l
    dfs[l]['language_label'] = language_to_int(l)
    dfs[l]['emotion_label'] = dfs[l]['emotion'].apply(emotion_to_int)

vocab = HuggingfaceVocab('custom-xlmroberta-tokenizer-60k')

def tokenize(row):
    tokens_enc = vocab.encode(row['utterance'].split(), MAX_LEN)
    row['tokens_encoded'] = tokens_enc
    row['utterance_spelled'] = vocab.decode_print(tokens_enc)
    row['tokens_len'] = len(tokens_enc)
    row['tokens'] = vocab.decode(tokens_enc)
    return row

for l in languages:
    dfs[l] = dfs[l].apply(tokenize, axis=1)
    
all_data = pd.concat(list(dfs.values()))

base_dir = './'
PATH_TO_WIKIART = args.wikiart_dir

def func(row):
    return PATH_TO_WIKIART + row['art_style'] + '/' + row['painting'] + '.jpg'

def save_set(df, path, test=False):
    if test:
        os.makedirs(osp.split(path)[0], exist_ok=True)
        df.to_csv(path, index=False)
    else:
        os.makedirs(path, exist_ok=True)
        df.to_csv(osp.join(path, 'artemis_preprocessed.csv'), index=False)
        
all_data['painting'] = all_data['painting'].apply(lambda x: unicodedata.normalize('NFD', x))
all_data['art_style'] = all_data['art_style'].apply(lambda x: unicodedata.normalize('NFD', x))
all_data['image_file'] = all_data.apply(func, axis=1)
all_data['image_file'] = all_data['image_file'].apply(lambda x: unicodedata.normalize('NFD', x))
all_data['grounding_emotion'] = all_data['emotion']

arabic = all_data[all_data['language']=='arabic']
english = all_data[all_data['language']=='english']
chinese = all_data[all_data['language']=='chinese']
# save training
save_set(df=all_data, path=osp.join(base_dir, 'all_langs/train'))
save_set(df=arabic, path=osp.join(base_dir, 'arabic/train'))
save_set(df=english, path=osp.join(base_dir, 'english/train'))
save_set(df=chinese, path=osp.join(base_dir, 'chinese/train'))

# create all four test datasets
all_langs = all_data
for d in [all_data, arabic, english, chinese]:
    d_name = max([k for k, v in locals().items() if id(v) == id(d)], key=len)
    save_set(df=d[d['split']=='test'], path=osp.join(base_dir, f'test_{d_name}/test_{d_name}.csv'), test=True)