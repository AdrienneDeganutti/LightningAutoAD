import os
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from dataset_utils.vision_language_tsv import VisionLanguageTSVYamlDataset
import pytorch_lightning as pl


class MyDataset(Dataset):
    def __init__(self, args, yaml_file, precomputed_file):
        with open(precomputed_file, 'rb') as f:
            self.data = torch.load(f)
        print(f"Loaded precomputed dataset with {len(self.data)} samples from {precomputed_file}")

        directory_path = os.path.dirname(yaml_file)
        args.train_gt_labels = os.path.join(directory_path, 'train.caption_coco_format.json')
        args.val_gt_labels = os.path.join(directory_path, 'test.caption_coco_format.json')


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class MyDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_yaml = args.train_yaml
        self.val_yaml = args.val_yaml
        self.test_yaml = args.test_yaml
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.train_precomputed_file = self.save_precomputed_dataset_train()
        self.val_precomputed_file = self.save_precomputed_dataset_val()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = MyDataset(self.args, self.train_yaml, self.train_precomputed_file)
            self.val_dataset = MyDataset(self.args, self.val_yaml, self.val_precomputed_file)
        
        if stage == 'test' or stage is None:
            self.test_dataset = MyDataset(self.test_yaml)
    
    def train_dataloader(self):
        train_sampler = DistributedSampler(self.train_dataset) if self.args.num_devices>1 else None
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=(train_sampler is None), sampler=train_sampler)
    
    def val_dataloader(self):
        val_sampler = DistributedSampler(self.val_dataset) if self.args.num_devices>1 else None
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=val_sampler)
    
    #def test_dataloader(self):
    #    test_sampler = DistributedSampler(self.test_dataset) if self.trainer.use_ddp or self.trainer.use_ddp2 else None
    #    return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=test_sampler)

    def save_precomputed_dataset_val(self):
        dataset = VisionLanguageTSVYamlDataset(self.args, self.val_yaml)
        data = [dataset[i] for i in range(len(dataset))]  # Precompute the entire dataset
        val_precomputed_file = 'precomputed_val_dataset.pt'
        torch.save(data, val_precomputed_file)
        print(f"Precomputed dataset saved to {val_precomputed_file}")
        return val_precomputed_file
    
    def save_precomputed_dataset_train(self):
        dataset = VisionLanguageTSVYamlDataset(self.args, self.train_yaml)
        data = [dataset[i] for i in range(len(dataset))]  # Precompute the entire dataset
        train_precomputed_file = 'precomputed_train_dataset.pt'
        torch.save(data, train_precomputed_file)
        print(f"Precomputed dataset saved to {train_precomputed_file}")
        return train_precomputed_file