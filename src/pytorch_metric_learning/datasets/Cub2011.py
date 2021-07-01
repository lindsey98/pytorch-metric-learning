# Modified https://github.com/TDeVries/cub2011_dataset/blob/master/cub2011.py

import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from PIL import Image
import torch


class Cub2011(torch.utils.data.Dataset):

    def __init__(self, root, train=True, transform=None):
        
        self.root = root
        self.transform = transform
        self.train = train

        if self.train == True:
            images = [x.strip() for x in open(os.path.join(self.root, 'lists', 'train.txt')).readlines()]
        else:
            images = [x.strip() for x in open(os.path.join(self.root, 'lists', 'test.txt')).readlines()]
        
        self.data = images
        self.labels = [int(os.path.basename(os.path.dirname(x)).split('.')[0]) for x in images]

    def __getitem__(self, idx):
        sample, label = self.data[idx], self.labels[idx]
        path = os.path.join(self.root, 'images', sample)
        label = label - 1  # Targets start at 1 by default, so shift to 0
        img = Image.open(path)
        
        if self.transform is not None:
            img = self.transform(img)

        return img, label
    
    
    def __len__(self):
        return len(self.data)