import numpy as np
import json, math, time, os

from torch.utils.data import SubsetRandomSampler
import torch.optim as optim
import higher
from model import *

import copy

BATCH_SIZE = 300
TEST_SIZE = 300

mnist_train = torchvision.datasets.MNIST(
    "./data", train=True, download=True, 
    transform=torchvision.transforms.Compose([
            #torchvision.transforms.RandomAffine(degrees=180, translate=None, scale=None, shear=None, resample=False, fillcolor=0),
            torchvision.transforms.ToTensor()
        ])
)

mnist_test = torchvision.datasets.MNIST(
    "./data", train=False, download=True, transform=torchvision.transforms.ToTensor()
)

#train_subset_indices=range(int(len(mnist_train)/2))
train_subset_indices=range(BATCH_SIZE)
val_subset_indices=range(int(len(mnist_train)/2),len(mnist_train))

dl_train = torch.utils.data.DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=False, sampler=SubsetRandomSampler(train_subset_indices))
dl_val = torch.utils.data.DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=False, sampler=SubsetRandomSampler(val_subset_indices))
dl_test = torch.utils.data.DataLoader(mnist_test, batch_size=TEST_SIZE, shuffle=False)


def test(model):
    model.eval()
    for i, (features, labels) in enumerate(dl_test):
        pred = model.forward(features)
        return pred.argmax(dim=1).eq(labels).sum().item() / TEST_SIZE * 100

def train_classic(model, optim, epochs=1):
    model.train()
    log = []
    for epoch in range(epochs):
        t0 = time.process_time()
        for i, (features, labels) in enumerate(dl_train):

            optim.zero_grad()
            pred = model.forward(features)
            loss = F.cross_entropy(pred,labels)
            loss.backward()
            optim.step()

        #### Log ####
        tf = time.process_time()
        data={
            "time": tf - t0,
        }
        log.append(data)

    times = [x["time"] for x in log]
    print("Vanilla : acc", test(model), "in (ms):", np.mean(times), "+/-", np.std(times))
##########################################
if __name__ == "__main__":

    device = torch.device('cpu')

    model = LeNet(1,10)
    opt_param = {
        "lr": torch.tensor(1e-2).requires_grad_(),
        "momentum": torch.tensor(0.9).requires_grad_()
        }
    n_inner_iter = 1
    dl_train_it = iter(dl_train)
    dl_val_it = iter(dl_val)
    epoch = 0
    epochs = 10

    ####
    train_classic(model=model, optim=torch.optim.Adam(model.parameters(), lr=0.001), epochs=epochs)
    model = LeNet(1,10)

    meta_opt = torch.optim.Adam(opt_param.values(), lr=1e-2)
    inner_opt = torch.optim.SGD(model.parameters(), lr=opt_param['lr'], momentum=opt_param['momentum'])
    #for xs_val, ys_val in dl_val:
    while epoch < epochs:
        #print(data_aug.params["mag"], data_aug.params["mag"].grad)
        meta_opt.zero_grad()
        model.train()
        with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=True, track_higher_grads=True) as (fmodel, diffopt): #effet copy_initial_weight pas clair...
            
            for param_group in diffopt.param_groups:
                param_group['lr'] = opt_param['lr']
                param_group['momentum'] = opt_param['momentum']

            for i in range(n_inner_iter):
                try:
                    xs, ys = next(dl_train_it)
                except StopIteration: #Fin epoch train
                    epoch +=1
                    dl_train_it = iter(dl_train)
                    xs, ys = next(dl_train_it)

                    print('Epoch', epoch)
                    print('train loss',loss.item(), '/ val loss', val_loss.item())
                    print('acc', test(model))
                    print('opt : lr', opt_param['lr'].item(), 'momentum', opt_param['momentum'].item())
                    print('-'*9)
                    model.train()


                logits = fmodel(xs)  # modified `params` can also be passed as a kwarg
                loss = F.cross_entropy(logits, ys)  # no need to call loss.backwards()
                #print('loss',loss.item())
                diffopt.step(loss)  # note that `step` must take `loss` as an argument!
                # The line above gets P[t+1] from P[t] and loss[t]. `step` also returns
                # these new parameters, as an alternative to getting them from
                # `fmodel.fast_params` or `fmodel.parameters()` after calling
                # `diffopt.step`.

                # At this point, or at any point in the iteration, you can take the
                # gradient of `fmodel.parameters()` (or equivalently
                # `fmodel.fast_params`) w.r.t. `fmodel.parameters(time=0)` (equivalently
                # `fmodel.init_fast_params`). i.e. `fast_params` will always have
                # `grad_fn` as an attribute, and be part of the gradient tape.
            
            # At the end of your inner loop you can obtain these e.g. ...
            #grad_of_grads = torch.autograd.grad(
            #    meta_loss_fn(fmodel.parameters()), fmodel.parameters(time=0))
            try:
                xs_val, ys_val = next(dl_val_it)
            except StopIteration: #Fin epoch val
                dl_val_it = iter(dl_val_it)
                xs_val, ys_val = next(dl_val_it)

            val_logits = fmodel(xs_val)
            val_loss = F.cross_entropy(val_logits, ys_val)
            #print('val_loss',val_loss.item())

            val_loss.backward()
            #meta_grads = torch.autograd.grad(val_loss, opt_lr, allow_unused=True)
            #print(meta_grads)
            for param_group in diffopt.param_groups:
                    print(param_group['lr'], '/',param_group['lr'].grad)
                    print(param_group['momentum'], '/',param_group['momentum'].grad)

            #model=copy.deepcopy(fmodel)
            model.load_state_dict(fmodel.state_dict())

            meta_opt.step()
