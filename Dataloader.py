import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn import metrics
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Define Dataset Class
class HumidityDataset(Dataset):
    def __init__(self, data_dir, categories, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform
        
        for category in tqdm(categories):
            path = os.path.join(data_dir, category)
            class_num = categories.index(category)
            
            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                    image_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                    image_resized = cv2.resize(image_rgb, (224, 224))
                    
                    if self.transform:
                        image_resized = self.transform(image_resized)
                    
                    self.data.append(image_resized)
                    self.labels.append(class_num)
                except Exception as e:
                    pass
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]