import os
import yaml
import torch
from torch.utils.data import DataLoader, Dataset
from dataset_utils.vision_language_tsv import VisionLanguageTSVYamlDataset
import pytorch_lightning as pl

class MyDataset(Dataset):
    def __init__(self, args, yaml_file):
        self.dataset = VisionLanguageTSVYamlDataset(args, yaml_file)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):

        img_ID, caption, visual_features = self.dataset[idx]
        return img_ID, caption, visual_features


class MyDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_yaml = args.train_yaml
        self.val_yaml = args.val_yaml
        self.test_yaml = args.test_yaml
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = MyDataset(self.args, self.train_yaml)
            self.val_dataset = MyDataset(self.args, self.val_yaml)
        
        if stage == 'test' or stage is None:
            self.test_dataset = MyDataset(self.test_yaml)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
