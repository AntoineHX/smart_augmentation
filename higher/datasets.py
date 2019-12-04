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

from torchvision.datasets.vision import VisionDataset
from PIL import Image
import augmentation_transforms
import numpy as np

class AugmentedDataset(VisionDataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, subset=None):

        super(AugmentedDataset, self).__init__(root, transform=transform, target_transform=target_transform)

        supervised_dataset = torchvision.datasets.CIFAR10(root, train=train, download=download, transform=transform)

        self.sup_data = supervised_dataset.data if not subset else supervised_dataset.data[subset[0]:subset[1]]
        self.sup_targets = supervised_dataset.targets if not subset else supervised_dataset.targets[subset[0]:subset[1]]
        assert len(self.sup_data)==len(self.sup_targets)

        for idx, img in enumerate(self.sup_data):
            self.sup_data[idx]= Image.fromarray(img) #to PIL Image

        self.unsup_data=[]
        self.unsup_targets=[]

        self.data= self.sup_data
        self.targets= self.sup_targets

        self.dataset_info= {
            'name': 'CIFAR10',
            'sup': len(self.sup_data),
            'unsup': len(self.unsup_data),
            'length': len(self.sup_data)+len(self.unsup_data),
        }


        self._TF = [
        'Invert', 'Cutout', 'Sharpness', 'AutoContrast', 'Posterize',
        'ShearX', 'TranslateX', 'TranslateY', 'ShearY', 'Rotate',
        'Equalize', 'Contrast', 'Color', 'Solarize', 'Brightness'
        ]
        self._op_list =[]
        self.prob=0.5
        for tf in self._TF:
            for mag in range(1, 10):
                self._op_list+=[(tf, self.prob, mag)]
        self._nb_op = len(self._op_list)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def augement_data(self, aug_copy=1):

        policies = []
        for op_1 in self._op_list:
            for op_2 in self._op_list:
                policies += [[op_1, op_2]]

        for idx, image in enumerate(self.sup_data):
            if (idx/self.dataset_info['sup'])%0.2==0: print("Augmenting data... ", idx,"/", self.dataset_info['sup'])

            for _ in range(aug_copy):
                chosen_policy = policies[np.random.choice(len(policies))]
                aug_image = augmentation_transforms.apply_policy(chosen_policy, image)
                #aug_image = augmentation_transforms.cutout_numpy(aug_image)

                self.unsup_data+=[aug_image]
                self.unsup_targets+=[self.sup_targets[idx]]

        self.unsup_data=np.array(self.unsup_data).astype(self.sup_data.dtype)
        self.data= np.concatenate((self.sup_data, self.unsup_data), axis=0)
        self.targets= np.concatenate((self.sup_targets, self.unsup_targets), axis=0)

        assert len(self.unsup_data)==len(self.unsup_targets)
        assert len(self.data)==len(self.targets)
        self.dataset_info['unsup']=len(self.unsup_data)
        self.dataset_info['length']=self.dataset_info['sup']+self.dataset_info['unsup']

    def len_supervised(self):
        return self.dataset_info['sup']

    def len_unsupervised(self):
        return self.dataset_info['unsup']

    def __len__(self):
        return self.dataset_info['length']

    def __str__(self):
        return "CIFAR10(Sup:{}-Unsup:{})".format(self.dataset_info['sup'], self.dataset_info['unsup'])

### Classic Dataset ###
data_train = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform)
#data_val = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform)
data_test = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform)


train_subset_indices=range(int(len(data_train)/2))
val_subset_indices=range(int(len(data_train)/2),len(data_train))
#train_subset_indices=range(BATCH_SIZE*10)
#val_subset_indices=range(BATCH_SIZE*10, BATCH_SIZE*20)

dl_train = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=False, sampler=SubsetRandomSampler(train_subset_indices))

### Augmented Dataset ###
data_train_aug = AugmentedDataset("./data", train=True, download=True, transform=transform, subset=(0,int(len(data_train)/2)))
#data_train_aug.augement_data(aug_copy=1)
print(data_train_aug)

dl_train = torch.utils.data.DataLoader(data_train_aug, batch_size=BATCH_SIZE, shuffle=True)


dl_val = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=False, sampler=SubsetRandomSampler(val_subset_indices))
dl_test = torch.utils.data.DataLoader(data_test, batch_size=TEST_SIZE, shuffle=False)
