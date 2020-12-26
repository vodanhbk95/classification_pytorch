import numpy as np
from sklearn.model_selection import StratifiedKFold
from create_dataloader import CassavaDataset
from torchvision import transforms
import pandas as pd 
import os

data_root = '/home/member/Workspace/haimd/hai_dataset/cassava_leaf_disease/train_images'
csv_path = '/home/member/Workspace/haimd/classfication_pytorch/train.csv'

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

data = pd.read_csv(csv_path)
target = data.loc[:, 'label']
skf = StratifiedKFold(n_splits=2)
fold_no = 1
for train_index, test_index in skf.split(data, target):
    train = data.loc[train_index, :]
    test = data.loc[test_index, :]
    import ipdb; ipdb.set_trace()
    dataset = CassavaDataset(data_root=data_root, data=train, transformation=transform_train)
    # import ipdb; ipdb.set_trace()
    data = iter(dataset)
    cnt = 0
    for i in range(len(dataset)):
        image, label = next(data)
        cnt += 1
        print(image.shape, label, cnt)
    

