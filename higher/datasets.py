import torch
from torch.utils.data import SubsetRandomSampler
import torchvision

BATCH_SIZE = 300
#TEST_SIZE = 300
TEST_SIZE = 10000

#ATTENTION : Dataug (Kornia) Expect image in the range of [0, 1]
#transform_train = torchvision.transforms.Compose([
#    torchvision.transforms.RandomHorizontalFlip(),
#    torchvision.transforms.ToTensor(),
    #torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #CIFAR10
#])
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    #torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #CIFAR10
])
'''
data_train = torchvision.datasets.MNIST(
    "./data", train=True, download=True, 
    transform=torchvision.transforms.Compose([
            #torchvision.transforms.RandomAffine(degrees=180, translate=None, scale=None, shear=None, resample=False, fillcolor=0),
            torchvision.transforms.ToTensor()
        ])
)
data_test = torchvision.datasets.MNIST(
    "./data", train=False, download=True, transform=torchvision.transforms.ToTensor()
)
'''
data_train = torchvision.datasets.CIFAR10(
    "./data", train=True, download=True, transform=transform
)
#data_val = torchvision.datasets.CIFAR10(
#    "./data", train=True, download=True, transform=transform
#)
data_test = torchvision.datasets.CIFAR10(
    "./data", train=False, download=True, transform=transform
)
#'''
train_subset_indices=range(int(len(data_train)/2))
val_subset_indices=range(int(len(data_train)/2),len(data_train))
#train_subset_indices=range(BATCH_SIZE*10)
#val_subset_indices=range(BATCH_SIZE*10, BATCH_SIZE*20)

dl_train = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=False, sampler=SubsetRandomSampler(train_subset_indices))
dl_val = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=False, sampler=SubsetRandomSampler(val_subset_indices))
dl_test = torch.utils.data.DataLoader(data_test, batch_size=TEST_SIZE, shuffle=False)