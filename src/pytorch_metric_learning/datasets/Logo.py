import pickle
from PIL import Image, ImageOps
from typing import List, Union, Callable
import torch

class GetLoader(torch.utils.data.Dataset):
    '''Define customized dataset
    Args:
        data_root:
            Path to directory holding the images to load.
        data_list:
            Path to txt file which map images to labels.
        label_dict:
            Dict which converts label in plain text to label in int.
        transform:
            Transformations to apply.
        grayscale:
            Grayscale model/RGB model, default is RGB
    
    '''
    def __init__(self, data_root, data_list, label_dict, transform=None, grayscale=False):
        
        self.transform = transform
        self.data_root = data_root
        self.grayscale = grayscale
        data_list = [x.strip('\n') for x in open(data_list).readlines()]

        with open(label_dict, 'rb') as handle:
            self.label_dict = pickle.load(handle)

        self.classes = list(self.label_dict.keys())

        self.n_data = len(data_list)

        self.img_paths = []
        self.labels = []
        self.targets = []

        for data in data_list:
            image_path = data
            label = image_path.split('/')[0]
            self.img_paths.append(image_path)
            self.labels.append(label)
            self.targets.append(self.label_dict[label])

    def __getitem__(self, index):

        img_path, label= self.img_paths[index], self.labels[index]
        img_path_full = os.path.join(self.data_root, img_path)
        if self.grayscale:
            img = Image.open(img_path_full).convert('L').convert('RGB')
        else:
            img = Image.open(img_path_full).convert('RGB')

        img = ImageOps.expand(img, (
            (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2,
            (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2), fill=(255, 255, 255))

        label = self.label_dict[label]
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return self.n_data