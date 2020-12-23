import torch
import os
from PIL import Image
import pandas as pd
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader

class CassavaDataset(Dataset):
    def __init__(self, data_root, csv_file, transformation=None):
        self.data_path = pd.read_csv(csv_file)
        self.transform = transformation
        self.data_root = data_root
    
    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.data_root, self.data_path.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.data_path.iloc[idx, 1]
        # import ipdb; ipdb.set_trace()
        if self.transform:
            image = self.transform(image)        
        
        return image, label


# if __name__ == '__main__':
    
#     transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     # transforms.Normalize((0.4914, 0.4822, 0.4465),
#     #                      (0.2023, 0.1994, 0.2010)),
# ])

#     transform_val = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465),
#                          (0.2023, 0.1994, 0.2010)),
# ])
#     # print(data)
#     # train_loader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True, num_workers=0, pin_memory=True)
#     # print(train_loader)
#     dataset = CassavaDataset(data_root='/home/member/Workspace/haimd/hai_dataset/cassava_leaf_disease/train_images', csv_file='/home/member/Workspace/haimd/hai_dataset/cassava_leaf_disease/train.csv',transformation=transform_train)
#     data = iter(dataset)
#     print(len(dataset))
#     for i in range(len(dataset)):
#         image, label = next(data)
#         print(image.shape, label, type(label))


        








