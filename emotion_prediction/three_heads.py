#%%
import torch
import torch.nn.functional as F
from transformers import AdamW, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
import datetime
from trainer_three_heads import Trainer
from utils import fix_seed, load_artemis_dataloaders
from models import ThreeHeadedMonster

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# loading Artemis dataset
ARTEMIS_EMOTIONS = ['amusement', 'awe', 'contentment', 'excitement',
                    'anger', 'disgust',  'fear', 'sadness', 'something else']
num_emo_classes = len(ARTEMIS_EMOTIONS)

BERT_version = 'xlm-roberta-base' 
print('Start Tokenizing ......')
tokenizer = AutoTokenizer.from_pretrained(BERT_version, padding_side='right')

datasets = ['english', 'arabic', 'chinese']
dataloaders = load_artemis_dataloaders(tokenizer)
#%%
print('Loading Model ......')
full_model = ThreeHeadedMonster(BERT_version, num_emo_classes)
full_model.to(device)
print('Done ......')
#%%
epochs = 5
num_batches = len(dataloaders['train']['english'])
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
model_path = f'three_model/Monster_{current_time}'

trainer = Trainer(full_model, dataloaders, optimizer, scheduler, 
                  criterion, epochs, num_batches, datasets, device, model_path)
trainer.train()
print("Training complete!")