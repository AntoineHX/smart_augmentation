""" Dataset definition.

    MNIST / CIFAR10
"""
import torch
from torch.utils.data import SubsetRandomSampler
from torch.utils.data.dataset import ConcatDataset
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
dataroot="../data"

#MNIST
#data_train = torchvision.datasets.MNIST(dataroot, train=True, download=True, transform=transform_train)
#data_val = torchvision.datasets.MNIST(dataroot, train=True, download=True, transform=transform)
#data_test = torchvision.datasets.MNIST(dataroot, train=False, download=True, transform=transform)

#CIFAR
data_train = torchvision.datasets.CIFAR10(dataroot, train=True, download=download_data, transform=transform_train)
#data_val = torchvision.datasets.CIFAR10(dataroot, train=True, download=download_data, transform=transform)
data_test = torchvision.datasets.CIFAR10(dataroot, train=False, download=download_data, transform=transform)

#data_train = torchvision.datasets.CIFAR100(dataroot, train=True, download=download_data, transform=transform_train)
#data_val = torchvision.datasets.CIFAR100(dataroot, train=True, download=download_data, transform=transform)
#data_test = torchvision.datasets.CIFAR100(dataroot, train=False, download=download_data, transform=transform)

#SVHN
#trainset = torchvision.datasets.SVHN(root=dataroot, split='train', download=download_data, transform=transform_train)
#extraset = torchvision.datasets.SVHN(root=dataroot, split='extra', download=download_data, transform=transform_train)
#data_train = ConcatDataset([trainset, extraset])
#data_test = torchvision.datasets.SVHN(dataroot, split='test', download=download_data, transform=transform)

#ImageNet
#Necessite SciPy
# Probleme ? : https://github.com/ildoonet/pytorch-randaugment/blob/48b8f509c4bbda93bbe733d98b3fd052b6e4c8ae/RandAugment/imagenet.py#L28
#data_train = torchvision.datasets.ImageNet(root=os.path.join(dataroot, 'imagenet-pytorch'), split='train', transform=transform_train)
#data_test = torchvision.datasets.ImageNet(root=os.path.join(dataroot, 'imagenet-pytorch'), split='val', transform=transform_test)


train_subset_indices=range(int(len(data_train)/2))
val_subset_indices=range(int(len(data_train)/2),len(data_train))
#train_subset_indices=range(BATCH_SIZE*10)
#val_subset_indices=range(BATCH_SIZE*10, BATCH_SIZE*20)

dl_train = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=False, sampler=SubsetRandomSampler(train_subset_indices), num_workers=num_workers, pin_memory=pin_memory)
dl_val = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=False, sampler=SubsetRandomSampler(val_subset_indices), num_workers=num_workers, pin_memory=pin_memory)
dl_test = torch.utils.data.DataLoader(data_test, batch_size=TEST_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
