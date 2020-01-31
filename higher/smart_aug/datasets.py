""" Dataset definition.

    MNIST / CIFAR10
"""
import torch
from torch.utils.data import SubsetRandomSampler
import torchvision

#Train/Validation batch size.
BATCH_SIZE = 300
#Test batch size.
TEST_SIZE = BATCH_SIZE 
#TEST_SIZE = 10000 #legerement +Rapide / + Consomation memoire !

#Wether to download data.
download_data=False
#Number of worker to use.
num_workers=2 #4
#Pin GPU memory
pin_memory=False #True :+ GPU memory / + Lent

#ATTENTION : Dataug (Kornia) Expect image in the range of [0, 1]
#transform_train = torchvision.transforms.Compose([
#    torchvision.transforms.RandomHorizontalFlip(),
#    torchvision.transforms.ToTensor(),
#    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #CIFAR10
#])
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
#    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #CIFAR10
])

transform_train = torchvision.transforms.Compose([
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomVerticalFlip(),
    torchvision.transforms.ToTensor(),
])
#from RandAugment import RandAugment
# Add RandAugment with N, M(hyperparameter)
#transform_train.transforms.insert(0, RandAugment(n=2, m=30))

### Classic Dataset ###
#MNIST
#data_train = torchvision.datasets.MNIST("../data", train=True, download=True, transform=transform_train)
#data_val = torchvision.datasets.MNIST("../data", train=True, download=True, transform=transform)
#data_test = torchvision.datasets.MNIST("../data", train=False, download=True, transform=transform)
#CIFAR
data_train = torchvision.datasets.CIFAR10("../data", train=True, download=download_data, transform=transform_train)
data_val = torchvision.datasets.CIFAR10("../data", train=True, download=download_data, transform=transform)
data_test = torchvision.datasets.CIFAR10("../data", train=False, download=download_data, transform=transform)

train_subset_indices=range(int(len(data_train)/2))
val_subset_indices=range(int(len(data_train)/2),len(data_train))
#train_subset_indices=range(BATCH_SIZE*10)
#val_subset_indices=range(BATCH_SIZE*10, BATCH_SIZE*20)

dl_train = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=False, sampler=SubsetRandomSampler(train_subset_indices), num_workers=num_workers, pin_memory=pin_memory)
dl_val = torch.utils.data.DataLoader(data_val, batch_size=BATCH_SIZE, shuffle=False, sampler=SubsetRandomSampler(val_subset_indices), num_workers=num_workers, pin_memory=pin_memory)
dl_test = torch.utils.data.DataLoader(data_test, batch_size=TEST_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
