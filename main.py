#!/usr/bin/env python3

import sys
sys.path.insert(0, "/home/eragon/Documents/scripts/efficientdet-pytorch")
import torch
import os
from datetime import datetime
import time
import random
import cv2
import pandas as pd
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from glob import glob
import warnings
warnings.filterwarnings("ignore")

# Custom codes
from utils import *
from data import *
from trainer import *

# Debug utils
import pysnooper
import torchsnooper

# Performance + Seed
SEED = 42
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Get data
data_path = "/home/eragon/Desktop/Datasets/Wheat/"

marking = pd.read_csv(data_path+"train.csv")
bboxs = np.stack(marking['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep = ',')))
# with pysnooper.snoop():
for i, column in enumerate(['x', 'y' , 'w', 'h']):
    marking[column] = bboxs[:, i]
marking.drop(columns = ['bbox'], inplace = True)

# Stratified Splits
skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = SEED)

df_folds = marking[['image_id']].copy()
df_folds.loc[:, 'bbox_count'] = 1
df_folds = df_folds.groupby('image_id').count()
df_folds.loc[:, 'source'] = marking[['image_id', 'source']].groupby('image_id').min()['source']
# print(df_folds)
df_folds.loc[:, 'stratify_group'] = np.char.add(
    df_folds['source'].values.astype(str),
    df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
)
df_folds.loc[:,'fold'] = 0

for fold_number, (train_index, val_index) in enumerate(skf.split(X = df_folds.index, y = df_folds['stratify_group'])):
    df_folds.loc[df_folds.iloc[val_index].index , 'fold'] = fold_number

print(df_folds)

# Load data using dataloaders

fold_number = 0

train_dataset = DatasetRetriever(
    image_ids = df_folds[df_folds['fold'] != fold_number].index.values,
    marking = marking,
    transforms=get_train_transforms(),
    test = False,
)

validation_dataset = DatasetRetriever(
    image_ids = df_folds[df_folds['fold'] != fold_number].index.values,
    marking = marking,
    transforms=get_valid_transforms(),
    test = True,
)

# Display one image as a test

image, target, image_id = train_dataset[1]
boxes = target['boxes'].cpu().numpy().astype(np.int32)

numpy_image = image.permute(1,2,0).cpu().numpy()

fig, ax = plt.subplots(1, 1, figsize = (16, 8))

for box in boxes:
    cv2.rectangle(numpy_image, (box[1], box[0]), (box[3], box[2]), (0, 1, 0), 2 )

ax.set_axis_off()
ax.imshow(numpy_image)
plt.savefig("initialImage.png")

# Function to run training

def run_training(net):
    device = torch.device('cuda:0')
    net.to(device)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        sampler=RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=TrainGlobalConfig.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        validation_dataset, 
        batch_size=TrainGlobalConfig.batch_size,
        num_workers=TrainGlobalConfig.num_workers,
        shuffle=False,
        sampler=SequentialSampler(validation_dataset),
        pin_memory=False,
        collate_fn=collate_fn,
    )

    fitter = Fitter(model = net, device = device, config = TrainGlobalConfig)
    fitter.fit(train_loader, val_loader)

# Main training part

net = get_net()
run_training(net)
    
   

