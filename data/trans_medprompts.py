from torch.utils.data import Dataset
import pandas as pd
from simple_torch.modules import SimpleTorchDataModule
from transformers import AutoTokenizer
import numpy as np

class TabularPASADataset(Dataset):
    def __init__(self, data_x_path, data_y_path, hparams):
        super().__init__()

        feats = np.load(data_x_path)

        if hparams.ablation == "None":
            self.feats = feats
        else:
            raise Exception("Incorrect input for ablation!")
        
        self.labels = np.load(data_y_path)[:, 0]


    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # extract the original row by index
        x = self.feats[idx]
        
        label = self.labels[idx]

        return x, label.astype("float32") 


class TabularMIMICDataset(Dataset):
    def __init__(self, data_x_path, data_y_path, hparams):
        super().__init__()

        feats = np.load(data_x_path)

        if hparams.ablation == "None":
            self.feats = feats
        elif hparams.ablation == "wo_text":
            self.feats = feats[:, :-4, :]
        else:
            raise Exception("Incorrect input for ablation!")
        
        self.labels = np.load(data_y_path)[:, -3]
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # extract the original row by index
        x = self.feats[idx]
        
        label = self.labels[idx]

        return x, label.astype("float32") 

class TransMedPromptsSTDM(SimpleTorchDataModule):
    def __init__(self, hparams):
        super().__init__(hparams)

        if hparams.dataset in ["mimic_mortality"]:
            self.train_dataset = TabularMIMICDataset(hparams.train_x_dir, hparams.train_y_dir, hparams)
            self.val_dataset = TabularMIMICDataset(hparams.val_x_dir, hparams.val_y_dir, hparams)
            self.test_dataset = TabularMIMICDataset(hparams.test_x_dir, hparams.test_y_dir, hparams)
        else:
            raise Exception("No dataset available.")
        


