import torch
import pandas as pd
import itertools
import pickle
import numpy as np
import random
import argparse

from artemis.in_out.basics import unpickle_data
from artemis.utils.vocabulary import Vocabulary, HuggingfaceVocab
from artemis.evaluation.single_caption_per_image import apply_basic_evaluations

import jieba
jieba.enable_paddle()

def print_out_some_basic_stats(captions):
    """
    Input: captions dataframe with column names caption
    """
    print('Some basic statistics:')
    mean_length = captions.caption.apply(lambda x: len(x.split())).mean()
    print(f'average length of productions {mean_length:.4}')
    unique_productions = len(captions.caption.unique()) / len(captions)
    print(f'percent of distinct productions {unique_productions:.2}')
    maximizer = captions.caption.mode()
    print(f'Most common production "{maximizer.iloc[0]}"')
    n_max = sum(captions.caption == maximizer.iloc[0]) 
    print(f'Most common production appears {n_max} times -- {n_max/ len(captions):.2} frequency.')
    u_tokens = set()
    captions.caption.apply(lambda x: [u_tokens.add(i) for i in x.split()]);
    print(f'Number of distinct tokens {len(u_tokens)}')
    
    
# needed to make the custom test data same as the wikiart one
def split_img_file(row):
    img_file = row['image_file']
    splitted_img_file = img_file.split('/')
    art_style = splitted_img_file[-2]
    painting = splitted_img_file[-1].split('.')[0]
    row['art_style'] = art_style
    row['painting'] = painting
    return row

def add_space(cap):
    cap = jieba.cut(cap, use_paddle=True)
    cap = ' '.join(list(cap))
    return cap

ARTEMIS_EMOTIONS = ['amusement', 'awe', 'contentment', 'excitement',
                    'anger', 'disgust',  'fear', 'sadness', 'something else']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--references', type=str, default=None,
                        help='Path to GT captions file')
    parser.add_argument('--generations', type=str, default=None,
                        help='Path to generated captions file')
    return parser.parse_args()

def group_gt_annotations(preprocessed_dataframe, vocab):
    df = preprocessed_dataframe
    results = dict()
    for split, g in df.groupby('split'): # group-by split
        g.reset_index(inplace=True, drop=True)
        g = g.groupby(['art_style', 'painting']) # group-by stimulus

        # group utterances / emotions
        # a) before "vocabularization" (i.e., raw)
        refs_pre_vocab_grouped = g['utterance_spelled'].apply(list).reset_index(name='references_pre_vocab')
        # b) post "vocabularization" (e.g., contain <UNK>)
        tokens_grouped = g['tokens_encoded'].apply(list).reset_index(name='tokens_encoded')
        emotion_grouped = g['emotion_label'].apply(list).reset_index(name='emotion')

        assert all(tokens_grouped['painting'] == emotion_grouped['painting'])
        assert all(tokens_grouped['painting'] == refs_pre_vocab_grouped['painting'])

        # decode these tokens back to strings and name them "references"
        tokens_grouped['tokens_encoded'] =\
            tokens_grouped['tokens_encoded'].apply(lambda x: [vocab.decode_print(eval(sent)) for sent in x])
        tokens_grouped = tokens_grouped.rename(columns={'tokens_encoded': 'references'})

        # join results in a new single dataframe
        temp = pd.merge(emotion_grouped, refs_pre_vocab_grouped)
        result = pd.merge(temp, tokens_grouped)
        result.reset_index(drop=True, inplace=True)
        results[split] = result
    return results

args = parse_args()

evaluation_methods = {'bleu', 'cider', 'meteor', 'rouge'}
split = 'test'
gpu_id = 0
device = torch.device("cuda:" + str(gpu_id))
default_lcs_sample = [25000, 800]


txt2emo_clf = None
txt2emo_vocab = HuggingfaceVocab()

# output of preprocess_artemis_data.py
references_file = args.references

# the file with the samples
sampled_captions_file = args.generations


if 'pkl' in references_file:
    gt_data = next(unpickle_data(references_file))
    train_utters = gt_data['train']['references_pre_vocab']
    gt_data = gt_data[split]     
else:
    gt_data = pd.read_csv(references_file)
    gt_data = group_gt_annotations(gt_data, txt2emo_vocab)
    gt_data = gt_data[split]  
    train_utters = gt_data['references_pre_vocab']
    

train_utters = list(itertools.chain(*train_utters))  # undo the grouping per artwork to a single large list
print('Training Utterances', len(train_utters))
unique_train_utters = set(train_utters)
print('Unique Training Utterances', len(unique_train_utters))
print('Images Captioned', len(gt_data))

saved_samples = next(unpickle_data(sampled_captions_file))

for sampling_config_details, captions, attn in saved_samples:  # you might have sampled under several sampling configurations
    print('Sampling Config:', sampling_config_details)        
    print()            
    print_out_some_basic_stats(captions)
    print()
    
    # required to make the custom test data the same as the wikiart test set
    captions = captions.apply(split_img_file, axis=1) if 'csv' in references_file else captions

    if ('chinese' in references_file) or ('all_langs' in references_file):
        captions['caption'] = captions['caption'].apply(add_space)
        gt_data['references_pre_vocab'] = gt_data['references_pre_vocab'].apply(lambda x: [add_space(sent) for sent in x])
    
    merged = pd.merge(gt_data, captions)  # this ensures proper order of captions to gt (via accessing merged.captions)
    def compute_metrics(df):
        hypothesis = df.caption
        references = df.references_pre_vocab
        ref_emotions = df.emotion
        metrics_eval = apply_basic_evaluations(hypothesis, references, ref_emotions, txt2emo_clf, txt2emo_vocab, 
                                                nltk_bleu=False, lcs_sample=default_lcs_sample,
                                                train_utterances=unique_train_utters,
                                                methods_to_do=evaluation_methods)
        return pd.DataFrame(metrics_eval)

    metrics_eval = compute_metrics(merged)
    print(pd.DataFrame(metrics_eval))
    print()
    
print('#'*75)