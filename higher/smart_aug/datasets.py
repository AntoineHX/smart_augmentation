""" Dataset definition.

    MNIST / CIFAR10 / CIFAR100 / SVHN / ImageNet
"""
import torch
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
#Data storage folder
dataroot="../data"

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

## RandAugment ##
from RandAugment import RandAugment
# Add RandAugment with N, M(hyperparameter)
rand_aug={'N': 2, 'M': 1}
#rand_aug={'N': 2, 'M': 9./30} #RN-ImageNet
#rand_aug={'N': 3, 'M': 5./30} #WRN-CIFAR10
#rand_aug={'N': 2, 'M': 14./30} #WRN-CIFAR100
#rand_aug={'N': 3, 'M': 7./30} #WRN-SVHN
transform_train.transforms.insert(0, RandAugment(n=rand_aug['N'], m=rand_aug['M']))

### Classic Dataset ###

#MNIST
#data_train = torchvision.datasets.MNIST(dataroot, train=True, download=True, transform=transform_train)
#data_val = torchvision.datasets.MNIST(dataroot, train=True, download=True, transform=transform)
#data_test = torchvision.datasets.MNIST(dataroot, train=False, download=True, transform=transform)

#CIFAR
data_train = torchvision.datasets.CIFAR10(dataroot, train=True, download=download_data, transform=transform_train)
data_val = torchvision.datasets.CIFAR10(dataroot, train=True, download=download_data, transform=transform)
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


#Validation set size [0, 1]
valid_size=0.1
train_subset_indices=range(int(len(data_train)*(1-valid_size)))
val_subset_indices=range(int(len(data_train)*(1-valid_size)),len(data_train))
#train_subset_indices=range(BATCH_SIZE*10)
#val_subset_indices=range(BATCH_SIZE*10, BATCH_SIZE*20)

from torch.utils.data import SubsetRandomSampler
dl_train = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=False, sampler=SubsetRandomSampler(train_subset_indices), num_workers=num_workers, pin_memory=pin_memory)
dl_val = torch.utils.data.DataLoader(data_val, batch_size=BATCH_SIZE, shuffle=False, sampler=SubsetRandomSampler(val_subset_indices), num_workers=num_workers, pin_memory=pin_memory)
dl_test = torch.utils.data.DataLoader(data_test, batch_size=TEST_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

#Cross Validation
'''
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
class CVSplit(object):
    """Class that perform train/valid split on a dataset.

        Inspired from : https://skorch.readthedocs.io/en/latest/user/dataset.html

        Attributes:
            _stratified (bool): Wether the split should be stratified. Recommended to be True for unbalanced dataset.
            _val_size (float, int): If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the validation split. 
                If int, represents the absolute number of validation samples.
            _data (Dataset): Dataset to split.
            _targets (np.array): Targets of the dataset used if _stratified is set to True.
            _cv (BaseShuffleSplit) : Scikit learn object used to split.

    """
    def __init__(self, data, val_size=0.1, stratified=True):
        """ Intialize CVSplit.

            Args:
                data (Dataset): Dataset to split.
                val_size (float, int): If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the validation split. 
                If int, represents the absolute number of validation samples. (Default: 0.1)
                stratified (bool): Wether the split should be stratified. Recommended to be True for unbalanced dataset.
        """
        self._stratified=stratified
        self._val_size=val_size

        self._data=data
        if self._stratified:
            cv_cls = StratifiedShuffleSplit
            self._targets= np.array(data_train.targets)
        else:
            cv_cls = ShuffleSplit

        self._cv= cv_cls(test_size=val_size, random_state=0) #Random state w/ fixed seed

    def next_split(self):
        """ Get next cross-validation split.

            Returns:
                Train DataLoader, Validation DataLoader
        """
        args=(np.arange(len(self._data)),)
        if self._stratified:
            args = args + (self._targets,)
            
        idx_train, idx_valid = next(iter(self._cv.split(*args)))

        train_subset = torch.utils.data.Subset(self._data, idx_train)
        val_subset = torch.utils.data.Subset(self._data, idx_valid)

        dl_train = torch.utils.data.DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        dl_val = torch.utils.data.DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

        return dl_train, dl_val

cvs = CVSplit(data_train, val_size=valid_size)
dl_train, dl_val = cvs.next_split()
'''

'''
from skorch.dataset import CVSplit
import numpy as np
cvs = CVSplit(cv=valid_size, stratified=True) #Stratified =True for unbalanced dataset #ShuffleSplit

def next_CVSplit():

    train_subset, val_subset = cvs(data_train, y=np.array(data_train.targets))
    dl_train = torch.utils.data.DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    dl_val = torch.utils.data.DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    return dl_train, dl_val

dl_train, dl_val = next_CVSplit()
'''