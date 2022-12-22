# ArtElingo

Dataset and models' checkpoints can be found [here](https://forms.gle/pwbdJH1efv27EmHAA)

You will need to download the WikiArt images from [here](https://drive.google.com/file/d/1vTChp3nU5GQeLkPwotrybpUGUXj12BTK/view). This is provided by ArtGAN [repo](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset)

## Dataset Preparation

### Installing the raw dataset and tokenizer

Download the dataset from [here](https://drive.google.com/drive/folders/1lZWjqG7FT9ADN8TjGPrnSMloXi4pI2Ry?usp=sharing)

Place the dataset at `dataset/raw/`

Download our tokenizer from [here](https://drive.google.com/drive/folders/1A3rBfpBcI3tzUDN9h0yj_GaEAk4jz8M4?usp=sharing)

Place the tokenizer at `dataset/`

Download the WikiArt images from [here](https://drive.google.com/file/d/1vTChp3nU5GQeLkPwotrybpUGUXj12BTK/view)

unzip the images at `dataset/`

### Installing the required Env

```Console
cd sat
conda create -n artemis-sat python=3.6.9 cudatoolkit=10.0
conda activate artemis-sat
pip install -e .
cd ..
```

### Preprocessing the dataset

```Console
cd dataset
python preprocess.py --raw_data 'raw/artelingo.csv' --wikiart_dir 'wikiart/'
```

## Show, Attend and Tell

To train a SAT model,

```Console
conda activate artemis-sat
mkdir -p sat_logs/sat_english
python sat/artemis/scripts/train_speaker.py \
 -log-dir sat_logs/sat_english \
 -data-dir dataset/english/train/  \
 -img-dir dataset/wikiart/  \
 --use-emo-grounding True
```

The trained SAT model will be saved under the `sat_logs/sat_english`.
Alternatively, you can download one of our checkpoints from [here](https://drive.google.com/drive/folders/14OrL83Vyg5sSMS3nsH0nq_-qI-1r7XSI?usp=sharing)

To generate captions from a trained SAT model,

```Console
conda activate artemis-sat
mkdir -p sat_generations
python sat/artemis/scripts/sample_speaker.py \
-speaker-saved-args sat_logs/sat_english/config.json.txt \
-speaker-checkpoint sat_logs/sat_english/checkpoints/best_model.pt \
-img-dir dataset/wikiart/ \
-out-file  sat_generations/sat_english.pkl \
--custom-data-csv  dataset/test_english/test_english.csv 
```

The generations will be saved under `sat_generations/sat_english.pkl`.

Note that for training and testing, you can use any combination from the datasets found under `dataset/`

To evaluate the generated captions

```Console
conda activate artemis-sat
pip install jieba
python sat/get_scores.py \
--references dataset/test_english/test_english.csv  \
--generations sat_generations/sat_english.pkl \
```

## Meshed Memory Transformer

### Setting up the env

```Console
cd m2/
conda env create -f environment.yml
conda activate artemis-m2
python -m spacy download en
```

For training, sampling, and evaluation, please Follow the instructions in `m2/README.md`

## Emotion Prediction

### Setting up the env

```Console
cd emotion_prediction/
conda env create -f environment.yml
conda activate artemis-emo
```

We have 2 separate scripts for training the 3-headed transformer and the single head models respectively.

For the 3-headed transformer, you just need to run `emotion_prediction/three_heads.py` without any arguments, i.e.,

```Console
conda activate artemis-emo
python emotion_prediction/three_heads.py
```

For the single head models, you need to provide the tokenizer and the dataset language, i.e.,

```Console
conda activate artemis-emo
python emotion_prediction/single_head.py --bert_version bert-base-uncased --dataset_language english
```

For arabic, we used 'CAMeL-Lab/bert-base-arabic-camelbert-mix' tokenizer and model

For chinese, we used 'bert-base-chinese'

For evaluation metrics and analysis, `dataset/test_*.ipynb` are three notebooks for analyzing the different models.
