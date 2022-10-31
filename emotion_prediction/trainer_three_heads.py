import torch
import torch.nn as nn
import numpy as np
import time
from collections import deque
from tqdm import trange
from utils import format_time
from ignite.metrics import Accuracy

class Trainer():
    
    def __init__(self, model, dataloaders, optimizer, scheduler, criterion, num_epochs, num_batches, datasets, device, model_path):
        self.model = model
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.device = device
        self.iter_dataloaders = {s:{} for s in self.dataloaders.keys()}
        for split, lang_dict in dataloaders.items():
            for lang, dl in lang_dict.items():
                self.iter_dataloaders[split][lang] = iter(dl)
        print(self.iter_dataloaders)
        self.datasets = datasets
        self.total_steps = self.num_batches * self.num_epochs
        self.accuracy_metric = {lang:Accuracy() for lang in self.datasets} # can make a list of metrics and loop over it in the reset function as well as in the update and compute functions
        self.best_val_acc = 0
        self.model_path = model_path
    
    def reset_metrics(self):
        for v in self.accuracy_metric.values():
            v.reset()
        
    def train(self):
        for epoch_i in range(0, self.num_epochs):
            self.train_one_epoch(epoch_i)

            avg_val_acc = []
            for lang in self.datasets:
                val_loss, val_acc = self.evaluate(epoch_i, 'val', lang)
                avg_val_acc.append(val_acc)

            for lang in self.datasets:   
                _ = self.evaluate(epoch_i, 'test', lang)
        
            if np.mean(avg_val_acc) > self.best_val_acc:
                self.best_val_acc = np.mean(avg_val_acc)
                self.model.save_pretrained(self.model_path)
        
    def evaluate(self, epoch_i, split, lang):
        # Method to evaluate the model on a given split.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.num_epochs))
        print(f'{split}-{lang} evaluation...')
        t0 = time.time()
        self.reset_metrics()
        total_emo_loss = deque(maxlen=320)
        self.model.eval()
        dataloader = self.dataloaders[split][lang]
        num_batches = len(dataloader)
        t = trange(num_batches, desc='ML')
        for step, batch in zip(t, dataloader):
            step_loss = self.evaluate_one_step(batch, split, lang)
            total_emo_loss.append(step_loss)
            mean_acc = self.accuracy_metric[lang].compute()
            t.set_description(f'ML {split}-{lang} (loss={np.mean(total_emo_loss):.5f}) (acc={mean_acc:.5f})')
        avg_val_emo_loss = np.mean(total_emo_loss)
        
        print("")
        print(f"  {split}-{lang} loss:      {avg_val_emo_loss:.5f}")
        print(f"  {split}-{lang} Accuracy:  {self.accuracy_metric[lang].compute():.5f}")
        print(f"  {split}-{lang} eval took: {format_time(time.time() - t0)}")
        return avg_val_emo_loss, self.accuracy_metric[lang].compute()

    def evaluate_one_step(self, batch, split, lang):
        '''
        Method to evaluate one step of the model.
        '''            
        input_ids = batch[0].to(self.device)
        input_mask = batch[1].to(self.device)
        labels = batch[2].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, 
                                token_type_ids=None, 
                                attention_mask=input_mask,
                                language=lang) 
            emo_loss = self.criterion(outputs, labels, reduction = 'mean')
        self.accuracy_metric[lang].update((outputs, labels.argmax(dim=1)))       
        return emo_loss.item()

    def train_one_epoch(self, epoch_i):
        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.num_epochs))
        print('Training...')
        # Measure how long the training epoch takes.
        t0 = time.time()
        self.reset_metrics()
        total_emo_loss = deque(maxlen=320)
        self.model.train()
        t = trange(self.num_batches, desc='ML')
        for step in t:
            step_loss = self.train_one_step(step)
            total_emo_loss.append(step_loss)
            mean_acc = np.mean([v.compute() for v in self.accuracy_metric.values()])
            t.set_description(f'ML (loss={np.mean(total_emo_loss):.5f}) (acc={mean_acc:.5f})')
        avg_train_emo_loss = np.mean(total_emo_loss)

        print("")
        print("  Average training loss: {0:.5f}".format(avg_train_emo_loss))
        for lang in self.datasets:
            print(f"  {lang} Accuracy: {self.accuracy_metric[lang].compute():.5f}")
        print(f"  Avg Accuracy: {mean_acc:.5f}")
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))
    
    def train_one_step(self, step):
        '''
        Method to train one step of the model.
        '''
        total_emo_loss = list()
        for lang in self.datasets:
            dl = self.iter_dataloaders['train'][lang]
            try:
                batch = next(dl)
            except:
                self.iter_dataloaders['train'][lang] = iter(self.dataloaders['train'][lang])
                batch = next(self.iter_dataloaders['train'][lang])
                
            input_ids = batch[0].to(self.device)
            input_mask = batch[1].to(self.device)
            labels = batch[2].to(self.device)
        
            self.optimizer.zero_grad()  
            outputs = self.model(input_ids, 
                                token_type_ids=None, 
                                attention_mask=input_mask,
                                language=lang) 
            emo_loss = self.criterion(outputs, labels, reduction = 'mean')
            total_emo_loss.append(emo_loss.item())
            emo_loss.backward()
            
            self.accuracy_metric[lang].update((outputs, labels.argmax(dim=1)))
            
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

        return np.mean(total_emo_loss)