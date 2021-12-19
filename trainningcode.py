#!/usr/bin/env python
# coding: utf-8

# ### Import Packages
# 

# ### Import your drive's contents!

# In[ ]:


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


print(torch.__version__)


# ### Let's define some path, and our PokeMon dataset
# - Put the "pokemon" folder to somewhere of your Google Drive, and define the train/test path to "train_path" and "test_path"
# - To 'model_dir', put the drive's directory path that you want to save your model

# In[ ]:


train_path = './train' 
test_path = './test'
model_dir = './models'    #./drive/MyDrive/Path/To/Save/Your/Model
classes = ['bug', 'electric', 'fighting', 'fire', 'flying', 'grass', 'ground', 'phychic', 'poison', 'water']


# In[ ]:


class PokemonDataset(Dataset):
    def __init__(self, data_path, classes):
        self.data_path = data_path
        self.classes = classes

        # organize path information for __len__ and __getitem__
        self.img_path_label = list()
        for c in self.classes:
            img_list = os.listdir(os.path.join(self.data_path, c))
            for fp in img_list:
                full_fp = os.path.join(self.data_path, c, fp)
                self.img_path_label.append((full_fp, c, self.classes.index(c)))

        # Add some tranforms for data augmentation.
        self.tensor_transform = torchvision.transforms.ToTensor()
        self.normalize_transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])
        self.random_crop = torchvision.transforms.RandomResizedCrop( size  = 224)
        self.random_hor_flip  = torchvision.transforms.RandomHorizontalFlip( p = 0.5)
        self.random_ver_flip = torchvision.transforms.RandomVerticalFlip( p = 0.5 )
        self.resize = torchvision.transforms.Resize(size =224)
        self.rotation = torchvision.transforms.RandomRotation(degrees = 20)
        self.train_transform = torchvision.transforms.Compose([self.resize,
                                                               self.random_ver_flip,
                                                               self.random_hor_flip,
                                                               self.rotation,
                                                               self.normalize_transform])
        

    def __len__(self):
        return len(self.img_path_label)

    def __getitem__(self, idx):
        (fp, class_name, class_label) = self.img_path_label[idx]
        img = Image.open(fp)
        original_img = self.tensor_transform(img)
        input = self.normalize_transform(original_img)

        sample = dict()
        sample['input'] = input
        sample['original_img'] = original_img
        sample['target'] = class_label
        sample['class_name'] = class_name

        return sample


# In[ ]:


batch_size = 16

train_dataset = PokemonDataset(train_path, classes)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = PokemonDataset(test_path, classes)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_classes = len(classes)


# ### Set DataSet and DataLoader

# ### Take a sample and try to look at the one

# In[ ]:





# ### Choose your device - use GPU or not?

# In[ ]:


# device = 'cpu'
device = 'cuda'
print('Current Device : {}'.format(device))


# ### Define the model with the pre-trained ResNet

# In[ ]:

class Model(nn.Module):
    def __init__(self, feat_dim = 2048, output_dim =num_classes):
        super(Model, self).__init__()

        self.feat_dim = feat_dim
        self.output_dim = output_dim
    
        self.backbone = torchvision.models.resnet50(pretrained=True)
        for param in self.parameters():
            param.requires_grad = False

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
# ### Create a model and its optimizer
# 

# In[ ]:

model = Model()
model = model.to(device)
best_model = model
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)


# In[ ]:




# ### Define functions for train/test


# In[ ]:

def train(model, optimizer,sample):
    model.train()

    criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad()

    input = sample['input'].float().to(device)
    target = sample['target'].long().to(device) 
    
    pred = model(input)
    pred_loss = criterion(pred, target)
    
    top_val, top_idx = torch.topk(pred, 1)

    num_correct = torch.sum(top_idx == target.view(-1, 1))
    
    pred_loss.backward()
       
    optimizer.step()


    return pred_loss.item(), num_correct.item()


# In[ ]:


def test(model, sample):
    model.eval()

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        input = sample['input'].float().to(device)
        target = sample['target'].long().to(device) 

        pred = model(input)
        pred_loss = criterion(pred, target)

        top_val, top_idx = torch.topk(pred, 1)

        num_correct = torch.sum(top_idx == target.view(-1, 1))

    return pred_loss.item(), num_correct.item()
# ### Run Training

# In[ ]:

max_epoch = 10
save_stride = 5
tmp_path = './checkpoint.pth'
max_accu = -1

epoches = []
train_succes = []
test_succeses = []
train_losses = []
test_losses = []
for epoch in tqdm(range(max_epoch)):        
    ###Train Phase
    # Initialize Loss and Accuracy
    train_loss = 0.0
    train_accu = 0.0
    epoches.append(epoch)
    # Load the saved MODEL AND OPTIMIZER after evaluation.
    if epoch > 0:
        checkpoint = torch.load(tmp_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # how about learning rate scheduler?
        exp_lr_scheduler.step()
    # Iterate over the train_dataloader
    with tqdm(total=len(train_dataloader)) as pbar:
        for idx, sample in enumerate(train_dataloader):
            curr_loss, num_correct = train(model, optimizer, sample)
            train_loss += curr_loss / len(train_dataloader)
            train_accu += num_correct / len(train_dataset)
            pbar.update(1)
    train_losses.append(train_loss)
    train_succes.append(train_accu)

    # save the model and optimizer's information before the evaulation
    checkpoint = {
        'model' : Model(),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    # Save the checkpoint - you can try to save the "best" model with the validation accuracy/loss
    torch.save(checkpoint, tmp_path)
    if (epoch+1) % save_stride == 0:
        torch.save(checkpoint, os.path.join(model_dir, 'pokemon_r50_{}.pth'.format(epoch+1)))
    torch.save(checkpoint, os.path.join(model_dir, 'pokemon_r50_recent.pth'))
    
    ### Test Phase
    # Initialize Loss and Accuracy
    test_loss = 0.0
    test_accu = 0.0

    # Iterate over the test_dataloader
    with tqdm(total=len(test_dataloader)) as pbar:
        for idx, sample in enumerate(test_dataloader):
            curr_loss, num_correct = test(model, sample)
            test_loss += curr_loss / len(test_dataloader)
            test_accu += num_correct / len(test_dataset)
            pbar.update(1)

    test_losses.append(test_loss)
    test_succeses.append(test_accu)
    max_accu = max(test_accu, max_accu)
    if max_accu == test_accu:
        # Save your best model to the checkpoint
        torch.save(checkpoint, os.path.join(model_dir, 'pokemon_r50_best.pth'))
        best_model = model

    # These Lines would make you update your Google Drive after the saving.


    print(train_accu, test_accu)
    # These Lines would make you update your Google Drive after the saving.

    
    
plt.figure(figsize=(9, 3))
plt.plot(epoches, train_losses)
plt.title("train_losses")
plt.show()
plt.figure(figsize=(9, 3))
plt.plot(epoches,test_losses)
plt.title("test_loses")
plt.show()
plt.figure(figsize=(9, 3))
plt.plot(epoches,train_succes)
plt.title("train_succes")
plt.show()
plt.figure(figsize=(9, 3))
plt.plot(epoches,test_succeses)
plt.title("test_succes")
plt.show()
  
  

# In[



print(max_accu)



# In[ ]:
# Fine Tunning Part:Model upload
i=0
for param in model.parameters():
    #i = i+1
    param.requires_grad = True

model = best_model
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-8)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
# In[ ]:
## Fine Tunning Part: Model Trainning

max_epoch = 10
save_stride = 5
tmp_path = './checkpoint.pth'
max_accu = -1

epoches = []
train_succes = []
test_succeses = []
train_losses = []
test_losses = []
for epoch in tqdm(range(max_epoch)):        
    ###Train Phase
    # Initialize Loss and Accuracy
    train_loss = 0.0
    train_accu = 0.0
    epoches.append(epoch)
    # Load the saved MODEL AND OPTIMIZER after evaluation.
    if epoch > 0:
        checkpoint = torch.load(tmp_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # how about learning rate scheduler?
        exp_lr_scheduler.step()
    # Iterate over the train_dataloader
    with tqdm(total=len(train_dataloader)) as pbar:
        for idx, sample in enumerate(train_dataloader):
            curr_loss, num_correct = train(model, optimizer, sample)
            train_loss += curr_loss / len(train_dataloader)
            train_accu += num_correct / len(train_dataset)
            pbar.update(1)
    train_losses.append(train_loss)
    train_succes.append(train_accu)

    # save the model and optimizer's information before the evaulation
    checkpoint = {
        'model' : Model(),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    # Save the checkpoint - you can try to save the "best" model with the validation accuracy/loss
    torch.save(checkpoint, tmp_path)
    if (epoch+1) % save_stride == 0:
        torch.save(checkpoint, os.path.join(model_dir, 'pokemon_r50_{}.pth'.format(epoch+1)))
    torch.save(checkpoint, os.path.join(model_dir, 'pokemon_r50_recent.pth'))
    
    ### Test Phase
    # Initialize Loss and Accuracy
    test_loss = 0.0
    test_accu = 0.0

    # Iterate over the test_dataloader
    with tqdm(total=len(test_dataloader)) as pbar:
        for idx, sample in enumerate(test_dataloader):
            curr_loss, num_correct = test(model, sample)
            test_loss += curr_loss / len(test_dataloader)
            test_accu += num_correct / len(test_dataset)
            pbar.update(1)

    test_losses.append(test_loss)
    test_succeses.append(test_accu)
    max_accu = max(test_accu, max_accu)
    if max_accu == test_accu:
        # Save your best model to the checkpoint
        torch.save(checkpoint, os.path.join(model_dir, 'pokemon_r50_best.pth'))

    # These Lines would make you update your Google Drive after the saving.


    print(train_accu, test_accu)
    # These Lines would make you update your Google Drive after the saving.


    print(train_accu, test_accu)
    
    
plt.figure(figsize=(9, 3))
plt.plot(epoches, train_losses)
plt.title("train_losses")
plt.show()
plt.figure(figsize=(9, 3))
plt.plot(epoches,test_losses)
plt.title("test_loses")
plt.show()
plt.figure(figsize=(9, 3))
plt.plot(epoches,train_succes)
plt.title("train_succes")
plt.show()
plt.figure(figsize=(9, 3))
plt.plot(epoches,test_succeses)
plt.title("test_succes")
plt.show()
  
  

print(max_accu)

# In[ ]:



