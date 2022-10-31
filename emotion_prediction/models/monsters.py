import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import torch
import os
import numpy as np

ARTEMIS_EMOTIONS = ['amusement', 'awe', 'contentment', 'excitement',
                'anger', 'disgust',  'fear', 'sadness', 'something else']
EMOTION_ID = {e: i for i, e in enumerate(ARTEMIS_EMOTIONS)}
ID_EMOTION = {EMOTION_ID[e]: e for e in EMOTION_ID}

class ThreeHeadedMonster(nn.Module):
    """
    A class to create a 3 headed monster
    The model consists of a backbone that convert captions to a shared embedding space
    then 3 heads use the representations in the shared space to independantly predict
    the emotion. Each head corresponds to a different language.
    
    ...
    
    Attributes:
    -----------
    bert_version: str
        The version of the bert backbone to use.
    xlm_model: nn.Module
        The xlm backbone to use.
    dropout: nn.Dropout
        The dropout layer to use.
    lang: list(str)
        The languages supported by the model.
    heads: nn.ModuleDict(nn.Linear)
        a dictionary containing the heads for each language.
        
    Methods:
    --------
    create_xlm_backbone(bert_version) -> nn.Module
        A method to create the xlm backbone.
    forward(input_ids, token_type_ids, attention_mask, language) -> torch.Tensor
        A method to predict the emotion of a given sentence-language pair.
    save_pretrained(path) -> None
        A method to save the model to a given path.
    load_pretrained(path) -> None
        A method to load the model from a given path.
    """
    
    def __init__(self, bert_version, num_emo_classes):
        '''
        Construct the necessary components for the model as well as the model itself.
        
        Inputs:
        -------
        bert_version: str
            The version of the bert backbone to use.
        num_emo_classes: int
            The number of emotion classes to predict.
        '''
        super(ThreeHeadedMonster, self).__init__()
        # creating the xlm backbone
        self.bert_version = bert_version
        self.xlm_model = self.create_xlm_backbone(bert_version)
        self.dropout = nn.Dropout(p=0.1)
        # creating the 3 heads corresponding to the 3 languages
        self.langs = ['english', 'arabic', 'chinese']
        self.heads = nn.ModuleDict({
            'english': nn.Linear(self.xlm_model.config.hidden_size, num_emo_classes),
            'arabic': nn.Linear(self.xlm_model.config.hidden_size, num_emo_classes),
            'chinese': nn.Linear(self.xlm_model.config.hidden_size, num_emo_classes),
        })
        
    def create_xlm_backbone(self, bert_version):
        '''
        Construct the backbone xlm model.
        
        Inputs:
        -------
        bert_version: str
            The version of the bert backbone to use.
            
        Returns:
        --------
        xlm_model: nn.Module
            The xlm backbone to use.
        '''
        xlm_model = AutoModel.from_pretrained(bert_version)
        xlm_model.pooler.activation = nn.Identity()
        return xlm_model
    
    def forward(self, input_ids, token_type_ids, attention_mask, language):
        '''
        predict the emotion of a given sentence-language pair.
        
        Inputs:
        -------
        input_ids: torch.tensor
            The version of the bert backbone to use.
        token_type_ids: None
            unused input to the huggingface model.
        attention_mask: torch.tensor
            tensor of masks for each sentence.
        language: str
            the language of the head to use for prediction.
        Returns:
        --------
        emotion_logits: torch.tensor
            the logits of the emotion prediction.
        '''
        assert language in self.langs, f'Language {language} is not supported. Only {self.langs} are supported'
        outputs = self.xlm_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        emotion_logits = self.heads[language](self.dropout(outputs.pooler_output))
        return emotion_logits

    def predict(self, *args, **kwargs):
        logits = self.forward(*args, **kwargs)
        # converting logits to probs
        probs = F.softmax(logits, dim=1)
        labels = torch.argmax(probs, dim=1)
        emotion = [kwargs['language']+ '_' + ID_EMOTION[label.item()] for label in labels]
        return (emotion, probs) if kwargs.get('return_probs', False) else emotion

    def predict_all(self, *args, **kwargs):
        emotions = []
        for lang in self.langs:
            kwargs['language'] = lang
            emo = self.predict(*args, **kwargs)
            emotions.append([EMOTION_ID[e.split('_')[1]] for e in emo])
        return np.array(emotions).T
    
    def save_pretrained(self, path):
        '''
        Save the model to a given path.
        
        Inputs:
        -------
        path: str
            The path to save the model to.
        '''
        os.makedirs(path, exist_ok=True)
        self.xlm_model.save_pretrained(path)
        torch.save(self.heads.state_dict(), path+'/heads.pth')
        
    @classmethod
    def load_pretrained(cls, path, num_emo_classes):
        '''
        Load the model from a given path.
        
        Inputs:
        -------
        path: str
            The path to load the model from.
        num_emo_classes: int
            The number of emotion classes to predict.
        
        Returns:
        --------
        class_instance: ThreeHeadedMonster
            The loaded 3 headed monster with the learnt weights, ready to use for inference.
        '''
        classifier = cls(path, num_emo_classes) # propoerly intialize the xlm_model but the heads are randomly initialized
        # updating the heads  weights via the weights in path 
        classifier.heads.load_state_dict(torch.load(path+'/heads.pth'))
        classifier.eval()
        return classifier