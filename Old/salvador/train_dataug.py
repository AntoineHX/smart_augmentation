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
    'Contrast',
    'Color',
    'Brightness',
    'Sharpness',
    'Posterize',
    'Solarize', #=>Image entre [0,1] #Pas opti pour des batch
]

def compute_vaLoss(model, dl_it, dl):
    device = next(model.parameters()).device
    try:
        xs, ys = next(dl_it)
    except StopIteration: #Fin epoch val
        dl_it = iter(dl)
        xs, ys = next(dl_it)
    xs, ys = xs.to(device), ys.to(device)

    model.eval() #Validation sans transfornations !

    return F.cross_entropy(model(xs), ys)

def model_copy(src,dst, patch_copy=True, copy_grad=True):
    #model=copy.deepcopy(fmodel) #Pas approprie, on ne souhaite que les poids/grad (pas tout fmodel et ses etats)

    dst.load_state_dict(src.state_dict()) #Do not copy gradient ! 

    if patch_copy:
        dst['model'].load_state_dict(src['model'].state_dict()) #Copie donnee manquante ?
        dst['data_aug'].load_state_dict(src['data_aug'].state_dict())

    #Copie des gradients
    if copy_grad:
        for paramName, paramValue, in src.named_parameters():
          for netCopyName, netCopyValue, in dst.named_parameters():
            if paramName == netCopyName:
              netCopyValue.grad = paramValue.grad
              #netCopyValue=copy.deepcopy(paramValue)

    try: #Data_augV4
        dst['data_aug']._input_info = src['data_aug']._input_info 
        dst['data_aug']._TF_matrix = src['data_aug']._TF_matrix
    except:
        pass

def optim_copy(dopt, opt):

    #inner_opt.load_state_dict(diffopt.state_dict()) #Besoin sauver etat otpim (momentum, etc.) => Ne copie pas le state...
    #opt_param=higher.optim.get_trainable_opt_params(diffopt)

    for group_idx, group in enumerate(opt.param_groups):
       # print('gp idx',group_idx)
        for p_idx, p in enumerate(group['params']):
            opt.state[p]=dopt.state[group_idx][p_idx]


#############

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

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, master_bar):
    model.train()
    metric_logger = utils.MetricLogger(delimiter=" ")
    confmat = utils.ConfusionMatrix(num_classes=len(data_loader.dataset.classes))
    header = 'Epoch: {}'.format(epoch)
    for _, (image, target) in metric_logger.log_every(data_loader, header=header, parent=master_bar):

        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

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

def get_train_valid_loader(args, augment, random_seed, train_size=0.5, test_size=0.1, shuffle=True, num_workers=4, pin_memory=True):
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
    error_msg = "[!] test_size should be in the range [0, 1]."
    assert ((test_size >= 0) and (test_size <= 1)), error_msg

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

    test_dataset = torchvision.datasets.ImageFolder(
        root=args.data_path, transform=valid_transform
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(test_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, test_idx = indices[split:], indices[:split]
    train_idx, valid_idx = train_idx[:int(len(train_idx)*train_size)], train_idx[int(len(train_idx)*train_size):]
    print("\nTrain", len(train_idx), "\nValid", len(valid_idx), "\nTest", len(test_idx))
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx) if not args.test_only else SubsetSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx) if not args.test_only else SubsetSampler(valid_idx)
    test_sampler = SubsetSampler(test_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size if not args.test_only else 1, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size if not args.test_only else 1, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, sampler=test_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    imgs = np.asarray(train_dataset.imgs)

    # print('Train')
    # print(imgs[train_idx])
    #print('Valid')
    #print(imgs[valid_idx])

    return (train_loader, valid_loader, test_loader)

def main(args):
    print(args)

    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True

    #augment = True if not args.test_only else False
    augment = False

    data_loader, dl_val, data_loader_test = get_train_valid_loader(args=args, pin_memory=True, augment=augment,
                                                            num_workers=args.workers, train_size=0.99, test_size=0.2, random_seed=999)

    print("Creating model")
    model = torchvision.models.__dict__[args.model](pretrained=True)
    flat = list(model.children())

    body, head = nn.Sequential(*flat[:-2]), nn.Sequential(flat[-2], Lambda(func=lambda x: torch.flatten(x, 1)), nn.Linear(flat[-1].in_features, len(data_loader.dataset.classes))) 
    model = nn.Sequential(body, head)

    # model.fc = nn.Linear(model.fc.in_features, 2)
    # import ipdb; ipdb.set_trace()

    criterion = nn.CrossEntropyLoss().to(device)

    # optimizer = torch.optim.SGD(
    #     model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    '''
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)
    '''
    es = utils.EarlyStopping()
    
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
    """
    for epoch in mb:
        _, train_confmat = train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, mb)
        lr_scheduler.step( (epoch+1)*len(data_loader) )
        val_loss, _, valid_confmat = evaluate(model, criterion, data_loader_test, device=device)
        es(val_loss, model)

        # print('Valid Missed')
        # print(valid_missed)


        # print('Train')
        # print(train_confmat)
        print('Valid')
        print(valid_confmat)

        # if es.early_stop:
        #     break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    """

    #######

    inner_it = args.inner_it
    dataug_epoch_start=0
    print_freq=1
    KLdiv=False

    tf_dict = {k: TF.TF_dict[k] for k in tf_names}
    model = Augmented_model(Data_augV5(TF_dict=tf_dict, N_TF=3, mix_dist=0.0, fixed_prob=False, fixed_mag=False, shared_mag=False), model).to(device)
    #model = Augmented_model(RandAug(TF_dict=tf_dict, N_TF=2), model).to(device)

    val_loss=torch.tensor(0) #Necessaire si pas de metastep sur une epoch
    dl_val_it = iter(dl_val)
    countcopy=0

    #if inner_it!=0: 
    meta_opt = torch.optim.Adam(model['data_aug'].parameters(), lr=args.lr) #lr=1e-2
    #inner_opt = torch.optim.SGD(model['model'].parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay) #lr=1e-2 / momentum=0.9
    inner_opt = torch.optim.Adam(model['model'].parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        inner_opt,
        lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    high_grad_track = True
    if inner_it == 0:
        high_grad_track=False

    model.train()
    model.augment(mode=False)
    
    fmodel = higher.patch.monkeypatch(model, device=None, copy_initial_weights=True)
    diffopt = higher.optim.get_diff_optim(inner_opt, model.parameters(),fmodel=fmodel,track_higher_grads=high_grad_track)

    i=0

    for epoch in mb:

        metric_logger = utils.MetricLogger(delimiter=" ")
        confmat = utils.ConfusionMatrix(num_classes=len(data_loader.dataset.classes))
        header = 'Epoch: {}'.format(epoch)

        t0 = time.process_time()
        for _, (image, target) in metric_logger.log_every(data_loader, header=header, parent=mb):
        #for i, (xs, ys) in enumerate(dl_train):
            #print_torch_mem("it"+str(i))
            i+=1
            image, target = image.to(device), target.to(device)

            if(not KLdiv):
            #Methode uniforme
                logits = fmodel(image)  # modified `params` can also be passed as a kwarg
                output = F.log_softmax(logits, dim=1)
                loss = F.cross_entropy(output, target, reduction='none')  # no need to call loss.backwards()

                if fmodel._data_augmentation: #Weight loss
                    w_loss = fmodel['data_aug'].loss_weight()#.to(device)
                    loss = loss * w_loss
                loss = loss.mean()
            
            else:
            #Methode KL div
                fmodel.augment(mode=False)
                sup_logits = fmodel(xs)
                log_sup=F.log_softmax(sup_logits, dim=1)
                fmodel.augment(mode=True)
                loss = F.cross_entropy(log_sup, ys)

                if fmodel._data_augmentation:
                    aug_logits = fmodel(xs)
                    log_aug=F.log_softmax(aug_logits, dim=1)
                    aug_loss=0
                    if epoch>50: #debut differe ?
                        #KL div w/ logits - Similarite predictions (distributions)
                        aug_loss = F.softmax(sup_logits, dim=1)*(log_sup-log_aug)
                        aug_loss=aug_loss.sum(dim=-1)
                        #aug_loss = F.kl_div(aug_logits, sup_logits, reduction='none')
                        w_loss = fmodel['data_aug'].loss_weight() #Weight loss
                        aug_loss = (w_loss * aug_loss).mean()

                    aug_loss += (F.cross_entropy(log_aug, ys , reduction='none') * w_loss).mean()
                    #print(aug_loss)
                    unsupp_coeff = 1
                    loss += aug_loss * unsupp_coeff

            diffopt.step(loss) #(opt.zero_grad, loss.backward, opt.step)

            if(high_grad_track and i%inner_it==0): #Perform Meta step
                #print("meta")
                #Peu utile si high_grad_track = False
                val_loss = compute_vaLoss(model=fmodel, dl_it=dl_val_it, dl=dl_val) + fmodel['data_aug'].reg_loss()
                #print_graph(val_loss)

                val_loss.backward()

                countcopy+=1
                model_copy(src=fmodel, dst=model)
                optim_copy(dopt=diffopt, opt=inner_opt)

                #if epoch>50:
                meta_opt.step()
                model['data_aug'].adjust_param(soft=False) #Contrainte sum(proba)=1
                #model['data_aug'].next_TF_set()

                fmodel = higher.patch.monkeypatch(model, device=None, copy_initial_weights=True)
                diffopt = higher.optim.get_diff_optim(inner_opt, model.parameters(),fmodel=fmodel, track_higher_grads=high_grad_track)


            acc1 = utils.accuracy(output, target)[0]
            batch_size = image.shape[0]
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.update(loss=loss.item())

            confmat.update(target.flatten(), output.argmax(1).flatten())

            if(not high_grad_track and (torch.cuda.memory_cached()/1024.0**2)>20000): 
                countcopy+=1
                print_torch_mem("copy")
                model_copy(src=fmodel, dst=model)
                optim_copy(dopt=diffopt, opt=inner_opt)
                val_loss = compute_vaLoss(model=fmodel, dl_it=dl_val_it, dl=dl_val)

                #Necessaire pour reset higher (Accumule les fast_param meme avec track_higher_grads = False)
                fmodel = higher.patch.monkeypatch(model, device=None, copy_initial_weights=True)
                diffopt = higher.optim.get_diff_optim(inner_opt, model.parameters(),fmodel=fmodel, track_higher_grads=high_grad_track)
                print_torch_mem("copy")

        if(not high_grad_track): 
                countcopy+=1
                print_torch_mem("end copy")
                model_copy(src=fmodel, dst=model)
                optim_copy(dopt=diffopt, opt=inner_opt)
                val_loss = compute_vaLoss(model=fmodel, dl_it=dl_val_it, dl=dl_val)

                #Necessaire pour reset higher (Accumule les fast_param meme avec track_higher_grads = False)
                fmodel = higher.patch.monkeypatch(model, device=None, copy_initial_weights=True)
                diffopt = higher.optim.get_diff_optim(inner_opt, model.parameters(),fmodel=fmodel, track_higher_grads=high_grad_track)
                print_torch_mem("end copy")


        tf = time.process_time()


        #### Print ####
        if(print_freq and epoch%print_freq==0):
            print('-'*9)
            print('Epoch : %d'%(epoch))
            print('Time : %.00f'%(tf - t0))
            print('Train loss :',loss.item(), '/ val loss', val_loss.item())
            print('Data Augmention : {} (Epoch {})'.format(model._data_augmentation, dataug_epoch_start))
            print('TF Proba :', model['data_aug']['prob'].data)
            #print('proba grad',model['data_aug']['prob'].grad)
            print('TF Mag :', model['data_aug']['mag'].data)
            #print('Mag grad',model['data_aug']['mag'].grad)
            #print('Reg loss:', model['data_aug'].reg_loss().item())
            #print('Aug loss', aug_loss.item())
        #############
        #### Log ####
        #print(type(model['data_aug']) is dataug.Data_augV5)
        '''
        param = [{'p': p.item(), 'm':model['data_aug']['mag'].item()} for p in model['data_aug']['prob']] if model['data_aug']._shared_mag else [{'p': p.item(), 'm': m.item()} for p, m in zip(model['data_aug']['prob'], model['data_aug']['mag'])]
        data={
            "epoch": epoch,
            "train_loss": loss.item(),
            "val_loss": val_loss.item(),
            "acc": accuracy,
            "time": tf - t0,

            "param": param #if isinstance(model['data_aug'], Data_augV5) 
            #else [p.item() for p in model['data_aug']['prob']],
        }
        log.append(data)
        '''
        #############

        train_confmat=confmat
        lr_scheduler.step( (epoch+1)*len(data_loader) )

        test_loss, _, test_confmat = evaluate(model, criterion, data_loader_test, device=device)
        es(test_loss, model)

        # print('Valid Missed')
        # print(valid_missed)


        # print('Train')
        # print(train_confmat)
        print('Test')
        print(test_confmat)

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

    parser.add_argument('--in_it', '--inner_it', default=0, type=int,
                        metavar='N', help='higher inner_it',
                        dest='inner_it')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)