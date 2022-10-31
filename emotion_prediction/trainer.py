import torch
import torch.nn as nn
import numpy as np
import time
from collections import deque
from tqdm import trange
from utils import format_time
from ignite.metrics import Accuracy

class Trainer():
    
    def __init__(self, model, dataloaders, optimizer, scheduler, criterion, num_epochs, num_batches, device, model_path):
        self.model = model
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.device = device
        self.iter_dataloaders = {k: iter(dl) for k, dl in dataloaders.items()}
        self.total_steps = self.num_batches * self.num_epochs
        self.accuracy_metric = {split:Accuracy() for split in ['train', 'val', 'test']} # can make a list of metrics and loop over it in the reset function as well as in the update and compute functions
        self.best_val_acc = 0
        self.model_path = model_path
    
    def reset_metrics(self):
        for v in self.accuracy_metric.values():
            v.reset()
        
    def train(self):
        for epoch_i in range(0, self.num_epochs):
            self.train_one_epoch(epoch_i)
            val_loss, val_acc = self.evaluate(epoch_i, 'val')
            _ = self.evaluate(epoch_i, 'test')
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.model.save_pretrained(self.model_path)
        
    def evaluate(self, epoch_i, split):
        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.num_epochs))
        print(f'{split} evaluation...')
        t0 = time.time()
        self.reset_metrics()
        total_emo_loss = deque(maxlen=320)
        self.model.eval()
        num_batches = len(self.dataloaders[split])
        t = trange(num_batches, desc='ML')
        for step, batch in zip(t, self.dataloaders[split]):
            step_loss = self.evaluate_one_step(batch, split)
            total_emo_loss.append(step_loss)
            mean_acc = self.accuracy_metric[split].compute()
            t.set_description(f'ML {split} (loss={np.mean(total_emo_loss):.5f}) (acc={mean_acc:.5f})')
        avg_val_emo_loss = np.mean(total_emo_loss)
        
        print("")
        print(f"  {split} loss:      {avg_val_emo_loss:.5f}")
        print(f"  {split} Accuracy:  {self.accuracy_metric[split].compute():.5f}")
        print(f"  {split} eval took: {format_time(time.time() - t0)}")
        return avg_val_emo_loss, self.accuracy_metric[split].compute()

    def evaluate_one_step(self, batch, split):
        '''
        Method to evaluate one step of the model.
        '''            
        input_ids = batch[0].to(self.device)
        input_mask = batch[1].to(self.device)
        labels = batch[2].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, 
                                token_type_ids=None, 
                                attention_mask=input_mask) 
            emo_loss = self.criterion(outputs.logits, labels, reduction = 'mean')
        self.accuracy_metric[split].update((outputs.logits, labels.argmax(dim=1)))       
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
        for step, batch in zip(t, self.dataloaders['train']):
            step_loss = self.train_one_step(batch)
            total_emo_loss.append(step_loss)
            mean_acc = self.accuracy_metric['train'].compute()
            t.set_description(f'ML train (loss={np.mean(total_emo_loss):.5f}) (acc={mean_acc:.5f})')
        avg_train_emo_loss = np.mean(total_emo_loss)
        
        print("")
        print("  Average training loss: {0:.5f}".format(avg_train_emo_loss))
        print(f"  Train Accuracy: {self.accuracy_metric['train'].compute():.5f}")
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))
    
    def train_one_step(self, batch):
        '''
        Method to train one step of the model.
        '''            
        input_ids = batch[0].to(self.device)
        input_mask = batch[1].to(self.device)
        labels = batch[2].to(self.device)
    
        self.optimizer.zero_grad()  
        outputs = self.model(input_ids, 
                             token_type_ids=None, 
                             attention_mask=input_mask) 
        emo_loss = self.criterion(outputs.logits, labels, reduction = 'mean')
        emo_loss.backward()
        
        self.accuracy_metric['train'].update((outputs.logits, labels.argmax(dim=1)))

        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return emo_loss.item()