#%%
import torch
import torch.nn.functional as F
from transformers import AdamW, AutoModelForSequenceClassification, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
import datetime
from trainer import Trainer
from utils import fix_seed, load_artemis_dataloaders
import argparse

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_version', type=str, default='bert-base-uncased',
                        help='BERT model version to use')
    parser.add_argument('--dataset_language', type=str, default='english',
                        help='Dataset language to use')
    return parser.parse_args()

args = parse_args()
# loading Artemis dataset
ARTEMIS_EMOTIONS = ['amusement', 'awe', 'contentment', 'excitement',
                    'anger', 'disgust',  'fear', 'sadness', 'something else']
num_emo_classes = len(ARTEMIS_EMOTIONS)

BERT_version = args.bert_version #'CAMeL-Lab/bert-base-arabic-camelbert-mix', 'bert-base-chinese', 'xlm-roberta-base'
print('Start Tokenizing ......')
tokenizer = AutoTokenizer.from_pretrained(BERT_version, padding_side='right')

language = [args.dataset_language]
dataloaders = load_artemis_dataloaders(tokenizer, language, batch_size=32, root_path='dataset')
dataloaders = {k:v[language[0]] for k, v in dataloaders.items()} 
#%%
print('Loading Model ......')
full_model = AutoModelForSequenceClassification.from_pretrained(
                BERT_version,
                num_labels = num_emo_classes 
            )
full_model.to(device)
print('Done ......')
#%%
epochs = 5
num_batches = len(dataloaders['train'])
total_steps = num_batches * epochs
lr = 2e-5
optimizer = AdamW(full_model.parameters(), lr = lr, eps = 1e-8)
print(f'Adam learning rate: {lr}')
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0,
                                            num_training_steps = total_steps)
criterion = F.cross_entropy
seed_val = 42
fix_seed(seed_val)
#%%
current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
bert_name = args.bert_version.split('-')[0]
model_path = f'single_models/{bert_name}_{args.dataset_language}_{current_time}'

trainer = Trainer(full_model, dataloaders, optimizer, scheduler, 
                  criterion, epochs, num_batches, device, model_path)
trainer.train()
print("Training complete!")
