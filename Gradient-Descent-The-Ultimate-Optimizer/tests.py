import numpy as np
import json, math, time, os
from data_aug import *
#from data_aug_v2 import *
import gc

import matplotlib.pyplot as plt
from torchviz import make_dot, make_dot_from_trace

from torch.utils.data import SubsetRandomSampler

BATCH_SIZE = 300
#TEST_SIZE = 10000
TEST_SIZE = 300
DATA_LIMIT = 10

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
    "./data", train=True, download=True, 
    transform=torchvision.transforms.Compose([
            #torchvision.transforms.RandomAffine(degrees=180, translate=None, scale=None, shear=None, resample=False, fillcolor=0),
            torchvision.transforms.ToTensor()
        ])
)

data_test = torchvision.datasets.CIFAR10(
    "./data", train=False, download=True, transform=torchvision.transforms.ToTensor()
)

train_subset_indices=range(int(len(data_train)/2))
val_subset_indices=range(int(len(data_train)/2),len(data_train))

dl_train = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=False, sampler=SubsetRandomSampler(train_subset_indices))
dl_val = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=False, sampler=SubsetRandomSampler(val_subset_indices))
dl_test = torch.utils.data.DataLoader(data_test, batch_size=TEST_SIZE, shuffle=False)

def test(model, reshape_in=True, device = torch.device('cuda')):
    for i, (features_, labels_) in enumerate(dl_test):
        if reshape_in :
            features, labels = torch.reshape(features_, (TEST_SIZE, 28 * 28)), labels_
        else:
            features, labels =features_, labels_

        features, labels = features.to(device), labels.to(device)

        pred = model.forward(features)
        return pred.argmax(dim=1).eq(labels).sum().item() / TEST_SIZE * 100

def train_one_epoch(model, optimizer, epoch=0, reshape_in=True, device = torch.device('cuda'), train_data=True):
    if train_data: dl = dl_train
    else: dl = dl_val
    for i, (features_, labels_) in enumerate(dl):
        if i > DATA_LIMIT : break
        #t0 = time.process_time()

        if reshape_in :
            features, labels = torch.reshape(features_, (BATCH_SIZE, 28 * 28)), labels_
        else:
            features, labels =features_, labels_

        features, labels = features.to(device), labels.to(device)

        #optimizer.begin()
        #optimizer.zero_grad()
        model.begin()
        model.zero_grad()
        pred = model.forward(features)

        #loss = F.nll_loss(pred, labels)
        loss = F.cross_entropy(pred,labels)

        #model.print_grad_fn()
        #optimizer.print_grad_fn()
        #print('-'*50)

        loss.backward(create_graph=True)

        #optimizer.step()
        if train_data: model.adjust()
        else: model.adjust_val()
        
        #tf = time.process_time()
        #data = {
        #    "time": tf - t0,
        #    "iter": epoch * len(dl_train) + i,
        #    "loss": loss.item(),
        #    "params": {
        #        k: v.item()
        #        for k, v in model.optimizer.parameters.items()
        #        if "." not in k
        #    },
        #}
        #stats.append(data)

        #print_torch_mem(i)
    return loss.item()

def train_v2(model, optimizer, epochs=3, reshape_in=True, device = torch.device('cuda')):
    log = []
    for epoch in range(epochs):

        #dl_train.dataset.transform=torchvision.transforms.Compose([
        #    torchvision.transforms.RandomAffine(degrees=model.param('mag'), translate=None, scale=None, shear=None, resample=False, fillcolor=0),
        #    torchvision.transforms.ToTensor()
        #])
        viz_data(fig_name='res/data_sample')
        t0 = time.process_time()
        loss = train_one_epoch(model=model, optimizer=optimizer, epoch=epoch, reshape_in=reshape_in, device=device)
        train_one_epoch(model=model, optimizer=optimizer, epoch=epoch, reshape_in=reshape_in, device=device,train_data=False)

        #acc = test(model=model, reshape_in=reshape_in, device=device)
        acc = 0

        
        tf = time.process_time()
        data = {
            "time": tf - t0,
            "epoch": epoch,
            "loss": loss,
            "acc": acc,
            "params": {
                k: v.item()
                for k, v in model.optimizer.parameters.items()
                #for k, v in model.mods.data_aug.params.named_parameters()
                if "." not in k

            },
        }
        log.append(data)


        print("Epoch :",epoch+1, "/",epochs, "- Loss :",log[-1]["loss"])
        param = [p for p in model.param_grad() if p.grad is not None]
        if(len(param)!=0):
            print(param[-2],' / ', param[-2].grad)
            print(param[-1],' / ', param[-1].grad)
    return log

def train(model, epochs=3, height=1, reshape_in=True, device = torch.device('cuda')):
    stats = []
    for epoch in range(epochs):
        for i, (features_, labels_) in enumerate(dl_train):
            t0 = time.process_time()
            model.begin()
            if reshape_in :
                features, labels = torch.reshape(features_, (BATCH_SIZE, 28 * 28)), labels_
            else:
            	features, labels =features_, labels_

            features, labels = features.to(device), labels.to(device)
            
            pred = model.forward(
                features
            )  # typo in https://www.groundai.com/project/gradient-descent-the-ultimate-optimizer/
            #loss = F.nll_loss(pred, labels)
            loss = F.cross_entropy(pred,labels)

            #print('-'*50)
            #param = [p for p in model.param_grad() if p.grad is not None]
            #if(len(param)!=0):
            #	print(param[-2],' / ', param[-2].grad)
            #	print(param[-1],' / ', param[-1].grad)

            model.zero_grad()
            loss.backward(create_graph=True)
            model.adjust()
            tf = time.process_time()
            data = {
                "time": tf - t0,
                "iter": epoch * len(dl_train) + i,
                "loss": loss.item(),
                "params": {
                    k: v.item()
                    for k, v in model.optimizer.parameters.items()
                    if "." not in k
                },
            }
            stats.append(data)

        print('-'*50)
        i=0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)) and len(obj.size())>1:
                    print(i, type(obj), obj.size())
                    i+=1
            except:
                pass
        print("Epoch :",epoch+1, "/",epochs, "- Loss :",stats[-1]["loss"])
        param = [p for p in model.param_grad() if p.grad is not None]
        if(len(param)!=0):
            print(param[-2],' / ', param[-2].grad)
            print(param[-1],' / ', param[-1].grad)
    return stats

def run(opt, name="out", usr={}, epochs=10, height=1, cnn=True, device = torch.device('cuda')):
    torch.manual_seed(0x42)
    if not cnn:
        reshape_in = True
        #model = MNIST_FullyConnected(28 * 28, 128, 10, opt)
        model = MNIST_FullyConnected_Augmented(28 * 28, 128, 10, opt, device=device)
        
    else:
        reshape_in = False
        #model = LeNet(1, 10,opt, device)
        #model = LeNet_v2(1, 10,opt, device).to(device=device)
        model = LeNet_v2(3, 10,opt, device).to(device=device)
        optimizer=None
        '''
        m = LeNet_v3(1, 10)
        a = Data_aug()
        model = Augmented_model(model=m, 
                                data_augmenter=a,
                                optimizer=opt).to(device) #deux fois le meme optimizer ?...
        '''
        '''
        m = LeNet_v3(1, 10)
        a = Data_aug()
        model = Augmented_model(model=m, data_augmenter=a).to(device)
        #optimizer = SGD(model.parameters())
        optimizer = SGD(model.parameters(), lr=0.01, height=1)
        '''
        
        
    #for idx, m in enumerate(model.modules()):
    #    print(idx, '->', m)
    print("Running...", str(model))
    model.initialize()
    #print_model(model)
    #model.data_augmentation(False)
    #model.eval()

    log = train_v2(model=model, optimizer=optimizer, epochs=epochs, reshape_in=reshape_in, device=device)
    model.eval()
    acc = test(model, reshape_in, device=device)

    
    #param = [p for p in model.param_grad() if p.grad is not None]
    #if(len(param)!=0):
    #    print(param[-2],' / ', param[-2].grad)
    #    print(param[-1],' / ', param[-1].grad)
	
    out = {"acc": acc, "log": log, "usr": usr}
    with open("log/%s.json" % name, "w+") as f:
        json.dump(out, f, indent=True)
    times = [x["time"] for x in log]
    print("Times (ms):", np.mean(times), "+/-", np.std(times))
    print("Final accuracy:", acc)

    #plot_res(log, fig_name='res/'+name)

    return out

def make_adam_stack(height, top=0.0000001, device = torch.device('cuda')):
    #print(height,device)
    if height == 0:
        return Adam(alpha=top, device=device)
    return Adam(alpha=top, optimizer=make_adam_stack(height - 1, top, device=device), device=device)

def plot_res(log, fig_name='res'):
    
    fig, ax = plt.subplots(ncols=3, figsize=(15, 3))
    ax[0].set_title('Loss')
    ax[0].plot([x["loss"] for x in log])
        
    ax[1].set_title('Acc')
    ax[1].plot([x["acc"] for x in log]) 

    ax[2].set_title('mag')
    ax[2].plot([x["data_aug"] for x in log]) 

    plt.savefig(fig_name)

def print_torch_mem(add_info=''):

    nb=0
    max_size=0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)): # and len(obj.size())>1:
                #print(i, type(obj), obj.size())
                size = np.sum(obj.size())
                if(size>max_size): max_size=size
                nb+=1
        except:
            pass
    print(add_info, "-Pytroch tensor nb:",nb," / Max dim:", max_size)

def print_model(model, fig_name='graph/graph'): #Semble ne pas marcher pour les models en fonctionnel
    x = torch.randn(1,1,28,28, device=device)
    dot=make_dot(model(x), params=dict(model.named_parameters()))
    dot.format = 'svg' #https://graphviz.readthedocs.io/en/stable/manual.html#formats
    dot.render(fig_name)
    print("Model graph generated !")

def viz_data(fig_name='data_sample'):

    features_, labels_ = next(iter(dl_train))
    plt.figure(figsize=(10,10))
    #for i, (features_, labels_) in enumerate(dl_train):
    for i in range(25):
        if i==25: break
        #print(features_.size(), labels_.size())

        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        img = features_[i,0,:,:]
        
        #print('im shape',img.shape)
        plt.imshow(img, cmap=plt.cm.binary)
        plt.xlabel(labels_[i].item())

    plt.savefig(fig_name)

##########################################
if __name__ == "__main__":
    try:
        os.mkdir("log")
    except:
        print("log/ exists already")

    device = torch.device('cuda')

    run(make_adam_stack(height=1, top=0.001, device=device), 
        "Augmented_MNIST", 
        epochs=100, 
        cnn=True, 
        device = device)
    print()