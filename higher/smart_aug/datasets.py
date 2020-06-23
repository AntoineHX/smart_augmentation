""" Dataset definition.

    MNIST / CIFAR10 / CIFAR100 / SVHN / ImageNet
"""
import os
import torch
from torch.utils.data.dataset import ConcatDataset
import torchvision
from arg_parser import *

args = parser.parse_args()

#Wether to download data.
download_data=False
#Pin GPU memory
pin_memory=False #True :+ GPU memory / + Lent
#Data storage folder
dataroot=args.dataroot

# if args.dtype == 'FP32':
#     def_type=torch.float32
# elif args.dtype == 'FP16':
#     # def_type=torch.float16 #Default : float32
#     def_type=torch.bfloat16
# else:
#     raise Exception('dtype not supported :', args.dtype)

#ATTENTION : Dataug (Kornia) Expect image in the range of [0, 1]
transform = [
    #torchvision.transforms.Grayscale(3), #MNIST
    #torchvision.transforms.Resize((224,224), interpolation=2)#VGG
    torchvision.transforms.ToTensor(),
    #torchvision.transforms.Normalize(MEAN, STD), #CIFAR10
    # torchvision.transforms.Lambda(lambda tensor: tensor.to(def_type)),
]

transform_train = [
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomVerticalFlip(),
    #torchvision.transforms.Grayscale(3), #MNIST
    #torchvision.transforms.Resize((224,224), interpolation=2)
    torchvision.transforms.ToTensor(),
    #torchvision.transforms.Normalize(MEAN, STD), #CIFAR10
    # torchvision.transforms.Lambda(lambda tensor: tensor.to(def_type)),
]

## RandAugment ##
#from RandAugment import RandAugment
# Add RandAugment with N, M(hyperparameter)
#rand_aug={'N': 2, 'M': 1}
#rand_aug={'N': 2, 'M': 9./30} #RN-ImageNet
#rand_aug={'N': 3, 'M': 5./30} #WRN-CIFAR10
#rand_aug={'N': 2, 'M': 14./30} #WRN-CIFAR100
#rand_aug={'N': 3, 'M': 7./30} #WRN-SVHN
#transform_train.transforms.insert(0, RandAugment(n=rand_aug['N'], m=rand_aug['M']))

### Classic Dataset ###
BATCH_SIZE = args.batch_size
TEST_SIZE = BATCH_SIZE
# Load Dataset
if args.dataset == 'MNIST':
    transform_train.insert(0, torchvision.transforms.Grayscale(3))
    transform.insert(0, torchvision.transforms.Grayscale(3))

    val_set=False
    data_train = torchvision.datasets.MNIST(dataroot, train=True, download=True, transform=torchvision.transforms.Compose(transform_train))
    data_val = torchvision.datasets.MNIST(dataroot, train=True, download=True, transform=torchvision.transforms.Compose(transform))
    data_test = torchvision.datasets.MNIST(dataroot, train=False, download=True, transform=torchvision.transforms.Compose(transform))
elif args.dataset == 'CIFAR10': #(32x32 RGB)
    val_set=False
    MEAN=(0.4914, 0.4822, 0.4465)
    STD=(0.2023, 0.1994, 0.2010)
    data_train = torchvision.datasets.CIFAR10(dataroot, train=True, download=download_data, transform=torchvision.transforms.Compose(transform_train))
    data_val = torchvision.datasets.CIFAR10(dataroot, train=True, download=download_data, transform=torchvision.transforms.Compose(transform))
    data_test = torchvision.datasets.CIFAR10(dataroot, train=False, download=download_data, transform=torchvision.transforms.Compose(transform))
elif args.dataset == 'CIFAR100': #(32x32 RGB)
    val_set=False
    MEAN=(0.4914, 0.4822, 0.4465)
    STD=(0.2023, 0.1994, 0.2010)
    data_train = torchvision.datasets.CIFAR100(dataroot, train=True, download=download_data, transform=torchvision.transforms.Compose(transform_train))
    data_val = torchvision.datasets.CIFAR100(dataroot, train=True, download=download_data, transform=torchvision.transforms.Compose(transform))
    data_test = torchvision.datasets.CIFAR100(dataroot, train=False, download=download_data, transform=torchvision.transforms.Compose(transform))
elif args.dataset == 'TinyImageNet': #(Train:100k, Val:5k, Test:5k) (64x64 RGB)
    image_size=64 #128 / 224
    print('Using image size', image_size)
    transform_train=[torchvision.transforms.Resize(image_size), torchvision.transforms.CenterCrop(image_size)]+transform_train
    transform=[torchvision.transforms.Resize(image_size), torchvision.transforms.CenterCrop(image_size)]+transform
    
    val_set=True
    MEAN=(0.485, 0.456, 0.406)
    STD=(0.229, 0.224, 0.225)
    data_train = torchvision.datasets.ImageFolder(os.path.join(dataroot, 'tiny-imagenet-200/train'), transform=torchvision.transforms.Compose(transform_train))
    data_val = torchvision.datasets.ImageFolder(os.path.join(dataroot, 'tiny-imagenet-200/val'), transform=torchvision.transforms.Compose(transform))
    data_test = torchvision.datasets.ImageFolder(os.path.join(dataroot, 'tiny-imagenet-200/test'), transform=torchvision.transforms.Compose(transform))
elif args.dataset == 'ImageNet': #
    image_size=128 #224
    print('Using image size', image_size)
    transform_train=[torchvision.transforms.Resize(image_size), torchvision.transforms.CenterCrop(image_size)]+transform_train
    transform=[torchvision.transforms.Resize(image_size), torchvision.transforms.CenterCrop(image_size)]+transform
    
    val_set=False
    MEAN=(0.485, 0.456, 0.406)
    STD=(0.229, 0.224, 0.225)
    data_train = torchvision.datasets.ImageFolder(root=os.path.join(dataroot, 'ImageNet/train'), transform=torchvision.transforms.Compose(transform_train))
    data_val = torchvision.datasets.ImageFolder(root=os.path.join(dataroot, 'ImageNet/train'), transform=torchvision.transforms.Compose(transform))
    data_test = torchvision.datasets.ImageFolder(root=os.path.join(dataroot, 'ImageNet/validation'), transform=torchvision.transforms.Compose(transform))

else:
    raise Exception('Unknown dataset')

# Ready dataloader
if not val_set : #Split Training set into Train/Val
    #Validation set size [0, 1]
    valid_size=0.1
    train_subset_indices=range(int(len(data_train)*(1-valid_size)))
    val_subset_indices=range(int(len(data_train)*(1-valid_size)),len(data_train))
    #train_subset_indices=range(BATCH_SIZE*10)
    #val_subset_indices=range(BATCH_SIZE*10, BATCH_SIZE*20)

    from torch.utils.data import SubsetRandomSampler
    dl_train = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=False, sampler=SubsetRandomSampler(train_subset_indices), num_workers=args.workers, pin_memory=pin_memory)
    dl_val = torch.utils.data.DataLoader(data_val, batch_size=BATCH_SIZE, shuffle=False, sampler=SubsetRandomSampler(val_subset_indices), num_workers=args.workers, pin_memory=pin_memory)
    dl_test = torch.utils.data.DataLoader(data_test, batch_size=TEST_SIZE, shuffle=False, num_workers=args.workers, pin_memory=pin_memory)
else:
    dl_train = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.workers, pin_memory=pin_memory)
    dl_val = torch.utils.data.DataLoader(data_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.workers, pin_memory=pin_memory)
    dl_test = torch.utils.data.DataLoader(data_test, batch_size=TEST_SIZE, shuffle=False, num_workers=args.workers, pin_memory=pin_memory)


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