import numpy as np
from sklearn.model_selection import StratifiedKFold
from create_dataloader import CassavaDataset
from torchvision import transforms
import pandas as pd 
import os

import torch
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
import torch.optim as optim
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from utils import fit, validate
import timm
import os
from sklearn.model_selection import train_test_split
from create_dataloader import CassavaDataset
from adamW import AdamW
from sklearn.model_selection import StratifiedKFold

data_root = '/home/member/Workspace/haimd/hai_dataset/cassava_leaf_disease/train_images'
csv_path = '/home/member/Workspace/haimd/classfication_pytorch/train.csv'

transform_train = transforms.Compose([
    transforms.Resize((320,320)),
    # transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

data = pd.read_csv(csv_path)
target = data.loc[:, 'label']
skf = StratifiedKFold(n_splits=5)
fold_no = 0
for train_index, test_index in skf.split(data, target):
    train = data.loc[train_index, :]
    test = data.loc[test_index, :]
    # print('Fold_{} | Class ratio: {}'.format(fold_no, test['label']/len(test['label'])))
    fold_no += 1
    
    train_data = CassavaDataset(data_root=data_root, data=train, transformation=transform_train)
    val_data = CassavaDataset(data_root=data_root, data=test, transformation=transform_train)
    print('#'*20)
    print(f'{len(train)} | {len(test)} | Fold{fold_no}')

    # Learning parameters
    batch_size = 32
    epochs = 50
    lr = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    skf = StratifiedKFold(n_splits=3)


    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # model_ft = models.resnet18(pretrained=True)
    # num_infeature = model_ft.fc.in_features
    # model_ft.fc = nn.Linear(num_infeature, 5)
    # model_ft = nn.DataParallel(model_ft)
    model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=5)
    # model = EffNetb3(model_name='efficientnet_b3', pretrained=True)
    model = nn.DataParallel(model)
    model_ft = model.to(device)

    
    # optimizer
    # optimizer = optim.Adam(model_ft.parameters(), lr=lr)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [3, 6, 9], gamma=0.1)
    optimizer = AdamW(model_ft.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay = 0.1)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [15, 30, 50], gamma=0.1)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer)
    # criterion
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter()

    for epoch in range(epochs):
        print(f'Epoch {epoch+1} of {epochs}')
        train_epoch_loss, train_epoch_accuracy = fit(model_ft, train_loader, optimizer, scheduler, criterion, train_data)
        writer.add_scalar('Loss/train', train_epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', train_epoch_accuracy, epoch)
        print(f'Loss: {train_epoch_loss}, Acc: {train_epoch_accuracy}')
        valid_epoch_loss, valid_epoch_accuracy = validate(model_ft, valid_loader, criterion, val_data)
        print(f'Loss: {valid_epoch_loss}, Acc: {valid_epoch_accuracy}')
        writer.add_scalar('Loss/valid', valid_epoch_loss, epoch)
        writer.add_scalar('Accuracy/valid', valid_epoch_accuracy, epoch)
        print('-'*20)
        if not os.path.exists('./outputs'):
            os.makedirs('./outputs')
        # save checkpoint
        if epoch % 5 == 0:    
            torch.save({
                'epoch': epochs,
                'model_state_dict': model_ft.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
            }, './outputs/model_epoch_{}_fold_{}.pth'.format(epoch, fold_no)
    )
    