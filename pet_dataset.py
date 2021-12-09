import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torch import as_tensor 
import torch

class PetDataset(Dataset):
    def __init__(self, root_dir, csv, num_bins = 10, augment_fn=None, as_tensor=False):
        self.root_dir = root_dir
        if isinstance(csv, str):
            self.df = pd.read_csv(csv)
        elif isinstance(csv, pd.DataFrame):
            self.df = csv
        self.augment = augment_fn
        self.num_bins = num_bins
        self.as_tensor = as_tensor
        
    def __len__(self):
        return len(self.df)
    
    def bin_label(self, label):
        # data = np.random.randint(1, 101, size=(100))
        bins = np.linspace(0, 100, self.num_bins+1)
        digitized = np.digitize(label, bins, right=True)
        # -1 for 0 indexing
        return digitized - 1
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["Id"]
        img_path = os.path.join(self.root_dir, img_id) + ".jpg"
        img = Image.open(img_path)
        width, height = map(int, img.size)
        img = np.asarray(img)
        
        assert img.dtype == np.uint8
        if self.augment:
            img = self.augment(image=img)["image"]
        img = np.clip(img, 0.0, 1.0)
        
        # load labels if train.csv is the data source, else -1 for label missing
        try:
            raw_label = row["Pawpularity"]
            label = self.bin_label(raw_label)
        except KeyError:
            raw_label = -1
            label = -1

        
        if self.as_tensor:
            img = as_tensor(img).float()
            img = img.permute(2, 0, 1)
            label = as_tensor(label)
            
        sample = {
            "image": img,
            "label": label,
            "raw_label": raw_label,
            "width": width,
            "height": height,
        }
        
        return sample
