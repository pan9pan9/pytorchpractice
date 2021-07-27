import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose

## torchvision, torch audio, torchtext등으로 사용 가능 ##

training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor(),
)
# 공개 데이터 셋에서 학습 데이터 내려받기

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data,  batch_size=batch_size)

for x, y in test_dataloader:
    print("Shape of x [N, C, H, W]: ", x.shape)
    print("Shape of y:", y.shape, y.dtype)
    break