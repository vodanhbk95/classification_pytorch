from torchvision import datasets
from torchvision.transforms import transforms
from create_dataloader import CassavaDataset
import torch


from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)

#define transforms 
transform_train = transforms.Compose([
    transforms.RandomCrop((320, 320)),
    transforms.Resize((320,320)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

transform_val = transforms.Compose([
    transforms.Resize((320,320)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])


train_loader = torch.utils.data.DataLoader(train_data, batch_size=20, shuffle=True, num_workers=4, pin_memory=True)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=20, shuffle=False, num_workers=4, pin_memory=True)
# dataset = iter(train_data)
# for i in range(len(train_data)):
#     image, label = next(dataset)
#     print(image, label)
