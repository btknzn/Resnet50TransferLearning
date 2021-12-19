import os
import csv
import numpy as np
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt


from torch.utils.tensorboard import SummaryWriter

# Module for Importing Images
from PIL import Image 

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from efficientnet_pytorch import EfficientNet


class Model(nn.Module):
    def __init__(self, feat_dim = 2048, output_dim =10):
        super(Model, self).__init__()

        self.feat_dim = feat_dim
        self.output_dim = output_dim
    
        self.backbone = torchvision.models.resnet50(pretrained=True)
        for param in self.parameters():
            param.requires_grad = False
            #if i == 142:
            #    break
        #self.dropout = nn.Dropout(p=0.2)
        self.dropout = nn.Dropout(p=0.2)
        #self.backbone.fc = nn.Linear(feat_dim, output_dim)    
        self.backbone.fc = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
               nn.Dropout(0.4),
               nn.Linear(256, 10),
              nn.LogSoftmax(dim=1))
        
        
    def forward(self, img):
        out = self.backbone(img) 
        return out
# ### Create a model and i