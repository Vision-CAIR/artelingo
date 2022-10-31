import argparse
import pickle
import random

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

import evaluation
from data import (ArtEmis, ArtEmisDetectionsField, DataLoader, EmotionField, LanguageField, HuggingfaceVocab,
                  RawField, TextField)
from models.transformer import (MemoryAugmentedEncoder, MeshedDecoder,
                                ScaledDotProductAttentionMemory, Transformer)

import pathlib
import os.path as osp
import os
import pdb

import jieba

ROOT_DIR = osp.split(pathlib.Path(__file__).parent.parent.absolute())[0]

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

NUM_LANGS = 4

def predict_captions(model, dataloader, text_field, emotion_encoder=None, language_encoder=None, lang=None):
    import itertools
    if emotion_encoder is not None:
        emotion_encoder.eval()
    if language_encoder is not None:
        language_encoder.eval()
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        
        for it, (images, caps_emo_pair) in enumerate(iter(dataloader)):
            images = images.to(device)
            caps_gt, emotions, languages = caps_emo_pair
            if emotion_encoder is not None:
                emotions = torch.stack([torch.mode(emotion).values for emotion in emotions])
                emotions = F.one_hot(emotions, num_classes=9)
                emotions = emotions.type(torch.FloatTensor)
                emotions = emotions.to(device)
                enc_emotions = emotion_encoder(emotions)
                enc_emotions = enc_emotions.unsqueeze(1).repeat(1, images.shape[1], 1)
                images = torch.cat([images, enc_emotions], dim=-1)
            if language_encoder is not None:
                languages = torch.stack([torch.mode(language).values for language in languages]) # pick the most frequent language
                languages = F.one_hot(languages, num_classes=NUM_LANGS)
                languages = languages.type(torch.FloatTensor)
                languages = languages.to(device)
                enc_languages = language_encoder(languages)
                enc_languages = enc_languages.unsqueeze(1).repeat(1, images.shape[1], 1)
                images = torch.cat([images, enc_languages], dim=-1)

            with torch.no_grad():
                out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                if ('chinese' in lang) or ('all_langs' in lang):
                    gen_i = jieba.cut(gen_i, use_paddle=True)
                    gen_i = ' '.join(list(gen_i))
                    ll = []
                    for w in gts_i:
                        w = jieba.cut(w, use_paddle=True)
                        w = ' '.join(list(w))
                        ll.append(w)
                    gts_i = ll
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    store_dict = {'gen': gen,'gts': gts} 
    with open('test_results.pickle', 'wb') as f:
        pickle.dump(store_dict, f)

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)

    return scores


if __name__ == '__main__':
    device = torch.device('cuda')

    parser = argparse.ArgumentParser(description='Meshed-Memory Transformer')

    parser.add_argument('--exp_name', type=str, default='m2_transformer')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--features_path', type=str)
    parser.add_argument('--annotation_file', type=str)
    parser.add_argument('--use_emotion_labels', action='store_true')
    parser.add_argument('--use_language_labels', action='store_true')

    args = parser.parse_args()

    print('Meshed-Memory Transformer Evaluation')

    # Pipeline for image regions
    features_path = args.features_path if osp.isabs(args.features_path) else osp.join(ROOT_DIR, args.features_path)
    
    if os.path.exists('img_field.pkl'):
        with open('img_field.pkl', 'rb') as f:
            image_field = pickle.load(f)
        print('Loaded image field from file')
    else:
        image_field = ArtEmisDetectionsField(detections_path=features_path, max_detections=50)
        with open('img_field.pkl', 'wb') as f:
            pickle.dump(image_field, f)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # Pipeline for emotion
    emotions = [
        'amusement', 'awe', 'contentment', 'excitement', 
        'anger', 'disgust', 'fear', 'sadness', 'something else'
        ]
    emotion_field = EmotionField(emotions=emotions)

    languages = [
        'english','arabic','chinese', 'spanish'
        ]
    language_field = LanguageField(languages=languages)

    # Create the dataset
    annotation_file = args.annotation_file if osp.isabs(args.annotation_file) else osp.join(ROOT_DIR, args.annotation_file)
    dataset = ArtEmis(image_field, text_field, emotion_field, language_field, annotation_file)
    _, _, test_dataset = dataset.splits

    # text_field.vocab = pickle.load(open('vocab_%s.pkl' % args.exp_name, 'rb'))
    text_field.vocab = HuggingfaceVocab()
    text_field.max_len = 46

    # Model and dataloaders
    emotion_dim = 0
    emotion_encoder = None
    if args.use_emotion_labels:
        emotion_dim = 10
        emotion_encoder = torch.nn.Sequential(
            torch.nn.Linear(9, emotion_dim)
            )
        emotion_encoder.to(device)

    language_dim = 0
    language_encoder = None
    if args.use_language_labels:
        language_dim = 10
        language_encoder = torch.nn.Sequential(
            torch.nn.Linear(NUM_LANGS, language_dim)
            )
        language_encoder.to(device)

    encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory,
                                     attention_module_kwargs={'m': 40}, d_in=2048 + emotion_dim + language_dim)
    decoder = MeshedDecoder(len(text_field.vocab), 46, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

    fname = 'saved_m2_models/%s_best.pth' % args.exp_name
    data = torch.load(fname)
    model.load_state_dict(data['state_dict'])

    if emotion_encoder is not None:
        emotion_encoder.to(device)
        fname = 'saved_m2_models/%s_emo_best.pth' % args.exp_name
        data = torch.load(fname)
        emotion_encoder.load_state_dict(data)

    if language_encoder is not None:
        language_encoder.to(device)
        fname = 'saved_m2_models/%s_lang_best.pth' % args.exp_name
        data = torch.load(fname)
        language_encoder.load_state_dict(data)

    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField(), 'emotion': emotion_field, 'language': language_field})
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size, num_workers=args.workers)

    test_lang = annotation_file.split('/')[-3]
    print(test_lang)

    scores = predict_captions(model, dict_dataloader_test, text_field, emotion_encoder, language_encoder, lang=test_lang)
    print(scores)