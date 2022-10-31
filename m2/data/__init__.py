from .field import RawField, Merge, ArtEmisDetectionsField, TextField, EmotionField, LanguageField
from .dataset import ArtEmis
from torch.utils.data import DataLoader as TorchDataLoader
from .utils import HuggingfaceVocab

class DataLoader(TorchDataLoader):
    def __init__(self, dataset, *args, **kwargs):
        super(DataLoader, self).__init__(dataset, *args, collate_fn=dataset.collate_fn(), **kwargs)
