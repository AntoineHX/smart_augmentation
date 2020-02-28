import datetime
import os
import time
import sys

import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
from PIL import ImageEnhance
import random

import utils
from fastprogress import master_bar, progress_bar
import numpy as np

## DATA AUG ##
import higher
from dataug import *
from dataug_utils import *
tf_names = [
    ## Geometric TF ##
    'Identity',
    'FlipUD',
    'FlipLR',
    'Rotate',
    'TranslateX',
    'TranslateY',
    'ShearX',
    'ShearY',

    ## Color TF (Expect image in the range of [0, 1]) ##
    #'Contrast',
    #'Color',
    #'Brightness',
    #'Sharpness',
    #'Posterize',
    #'Solarize', #=>Image entre [0,1] #Pas opti pour des batch
]

class Lambda(nn.Module):
    "Create a layer that simply calls `func` with `x`"
    def __init__(self, func): 
        super().__init__()
        self.func=func
    def forward(self, x): return self.func(x)

class SubsetSampler(torch.utils.data.SubsetRandomSampler):
    def __init__(self, indices):
        super().__init__(indices)

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

def sharpness(img, factor):
    sharpness_factor = random.uniform(1, factor)
    sharp = ImageEnhance.Sharpness(img)
    sharped = sharp.enhance(sharpness_factor)
    return sharped

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, master_bar, Kldiv=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter=" ")
    confmat = utils.ConfusionMatrix(num_classes=len(data_loader.dataset.classes))
    header = 'Epoch: {}'.format(epoch)
    for _, (image, target) in metric_logger.log_every(data_loader, header=header, parent=master_bar):

        image, target = image.to(device), target.to(device)

        if not Kldiv :
            output = model(image)
            #output = F.log_softmax(output, dim=1)
            loss = criterion(output, target) #Pas de softmax ?

        else : #Consume x2 memory
            model.augment(mode=False)
            output = model(image)
            model.augment(mode=True)
            log_sup=F.log_softmax(output, dim=1)
            sup_loss = F.cross_entropy(log_sup, target)

            aug_output = model(image)
            log_aug=F.log_softmax(aug_output, dim=1)
            aug_loss=F.cross_entropy(log_aug, target)

            #KL div w/ logits - Similarite predictions (distributions)
            KL_loss = F.softmax(output, dim=1)*(log_sup-log_aug)
            KL_loss = KL_loss.sum(dim=-1)
            #KL_loss = F.kl_div(aug_logits, sup_logits, reduction='none')
            KL_loss = KL_loss.mean()

            unsupp_coeff = 1
            loss = sup_loss + (aug_loss + KL_loss) * unsupp_coeff
            #print(sup_loss.item(), (aug_loss + KL_loss).item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1 = utils.accuracy(output, target)[0]
        batch_size = image.shape[0]
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.update(loss=loss.item())

        confmat.update(target.flatten(), output.argmax(1).flatten())


    return metric_logger.loss.global_avg, confmat


def evaluate(model, criterion, data_loader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    confmat = utils.ConfusionMatrix(num_classes=len(data_loader.dataset.classes))
    header = 'Test:'
    missed = []
    with torch.no_grad():
        for i, (image, target) in metric_logger.log_every(data_loader, leave=False, header=header, parent=None):
            image, target = image.to(device), target.to(device)
            output = model(image)
            loss = criterion(output, target)
            if target.item() != output.topk(1)[1].item():
                missed.append(data_loader.dataset.imgs[data_loader.sampler.indices[i]])

            confmat.update(target.flatten(), output.argmax(1).flatten())

            acc1 = utils.accuracy(output, target)[0]
            batch_size = image.shape[0]
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.update(loss=loss.item())


    return metric_logger.loss.global_avg, missed, confmat

def get_train_valid_loader(args, augment, random_seed, valid_size=0.1, shuffle=True, num_workers=4, pin_memory=True):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    # normalize = transforms.Normalize(
    #     mean=[0.4914, 0.4822, 0.4465],
    #     std=[0.2023, 0.1994, 0.2010],
    # )

    # define transforms
    if augment:
        train_transform = transforms.Compose([
            # transforms.ColorJitter(brightness=0.3),
            # transforms.Lambda(lambda img: sharpness(img, 5)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize,
        ])

        valid_transform = transforms.Compose([
                # transforms.ColorJitter(brightness=0.3),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            # normalize,
        ])

        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            # normalize,
        ])


    # load the dataset
    train_dataset = torchvision.datasets.ImageFolder(
        root=args.data_path, transform=train_transform
    )

    valid_dataset = torchvision.datasets.ImageFolder(
        root=args.data_path, transform=valid_transform
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx) if not args.test_only else SubsetSampler(train_idx)
    valid_sampler = SubsetSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size if not args.test_only else 1, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    imgs = np.asarray(train_dataset.imgs)

    # print('Train')
    # print(imgs[train_idx])
    #print('Valid')
    #print(imgs[valid_idx])

    tgt = [0,0]
    for _, targets in train_loader: 
        for target in targets:
            tgt[target]+=1
    print("Train targets :", tgt)

    tgt = [0,0]
    for _, targets in valid_loader:
        for target in targets:
            tgt[target]+=1
    print("Valid targets :", tgt)

    return (train_loader, valid_loader)

def main(args):
    print(args)

    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True


    #augment = True if not args.test_only else False

    if not args.test_only and args.augment=='flip' : augment = True
    else : augment = False

    print("Augment", augment)
    data_loader, data_loader_test = get_train_valid_loader(args=args, pin_memory=True, augment=augment,
                                                            num_workers=args.workers, valid_size=0.3, random_seed=999)

    print("Creating model")
    model = torchvision.models.__dict__[args.model](pretrained=True)
    flat = list(model.children())

    body, head = nn.Sequential(*flat[:-2]), nn.Sequential(flat[-2], Lambda(func=lambda x: torch.flatten(x, 1)), nn.Linear(flat[-1].in_features, len(data_loader.dataset.classes))) 
    model = nn.Sequential(body, head)

    Kldiv=False
    if not args.test_only and (args.augment=='Rand' or args.augment=='RandKL'):
        tf_dict = {k: TF.TF_dict[k] for k in tf_names}
        model = Augmented_model(RandAug(TF_dict=tf_dict, N_TF=2), model).to(device)

        if args.augment=='RandKL': Kldiv=True

        model['data_aug']['mag'].data = model['data_aug']['mag'].data * args.magnitude
        print("Augmodel")
    
    # model.fc = nn.Linear(model.fc.in_features, 2)
    # import ipdb; ipdb.set_trace()

    criterion = nn.CrossEntropyLoss().to(device)

    # optimizer = torch.optim.SGD(
    #     model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    es = utils.EarlyStopping() if not (args.augment=='Rand' or args.augment=='RandKL') else utils.EarlyStopping(augmented_model=True)

    if args.test_only:
        model.load_state_dict(torch.load('checkpoint.pt', map_location=lambda storage, loc: storage))
        model = model.to(device)
        print('TEST')
        _, missed, _ = evaluate(model, criterion, data_loader_test, device=device)
        print(missed)
        print('TRAIN')
        _, missed, _ = evaluate(model, criterion, data_loader, device=device)
        print(missed)
        return

    model = model.to(device)

    print("Start training")
    start_time = time.time()
    mb = master_bar(range(args.epochs))

    for epoch in mb:
        _, train_confmat = train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, mb, Kldiv)
        lr_scheduler.step( (epoch+1)*len(data_loader) )
        val_loss, _, valid_confmat = evaluate(model, criterion, data_loader_test, device=device)
        es(val_loss, model)

        # print('Valid Missed')
        # print(valid_missed)

        # print('Train')
        # print(train_confmat)
        #print('Valid')
        #print(valid_confmat)

        # if es.early_stop:
        #     break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--data-path', default='/github/smart_augmentation/salvador/data', help='dataset')
    parser.add_argument('--model', default='resnet18', help='model') #'resnet18'
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=3, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=4e-5, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    parser.add_argument('-a', '--augment', default='None', type=str,
                        metavar='N', help='Data augment',
                        dest='augment')
    parser.add_argument('-m', '--magnitude', default=1.0, type=float,
                        metavar='N', help='Augmentation magnitude',
                        dest='magnitude')


    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)