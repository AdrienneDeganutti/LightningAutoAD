import os
import yaml
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

class MyDataset(Dataset):
    def __init__(self, data_dir):
        with open(data_dir, 'r') as file:
            data_config = yaml.safe_load(file)
        
        self.directory = data_config['data_dir']
        self.visual_features_dir = os.path.join(self.directory, data_config['img'])
        self.labels_dir = os.path.join(self.directory, data_config['label'])

        self.visual_feature_files = sorted(os.listdir(self.visual_features_dir))
        self.label_files = sorted(os.listdir(self.labels_dir))
    
    def __len__(self):
        return len(self.visual_feature_files)
    
    def __getitem__(self, idx):
        visual_feature_path = os.path.join(self.visual_features_dir, self.visual_feature_files[idx])
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        
        # Load visual features and labels
        visual_features = torch.load(visual_feature_path)
        labels = torch.load(label_path)
        
        return visual_features, labels


class MyDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.train_yaml = args.train_yaml
        self.val_yaml = args.val_yaml
        self.test_yaml = args.test_yaml
        self.batch_size = args.batch_size

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = MyDataset(self.train_yaml)
            self.val_dataset = MyDataset(self.val_yaml)
        
        if stage == 'test' or stage is None:
            self.test_dataset = MyDataset(self.test_yaml)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
