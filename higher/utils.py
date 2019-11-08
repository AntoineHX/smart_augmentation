import numpy as np
import json, math, time, os
import matplotlib.pyplot as plt
import copy
import gc

from torchviz import make_dot 

import torch
import torch.nn.functional as F


def print_graph(PyTorch_obj, fig_name='graph'):
    graph=make_dot(PyTorch_obj) #Loss give the whole graph
    graph.format = 'svg' #https://graphviz.readthedocs.io/en/stable/manual.html#formats
    graph.render(fig_name)

def plot_res(log, fig_name='res'):

    epochs = [x["epoch"] for x in log]

    fig, ax = plt.subplots(ncols=3, figsize=(15, 3))

    ax[0].set_title('Loss')
    ax[0].plot(epochs,[x["train_loss"] for x in log], label='Train')
    ax[0].plot(epochs,[x["val_loss"] for x in log], label='Val')
    ax[0].legend()
        
    ax[1].set_title('Acc')
    ax[1].plot(epochs,[x["acc"] for x in log]) 

    if log[0]["param"]!= None:
        if isinstance(log[0]["param"],float):
            ax[2].set_title('Mag')
            ax[2].plot(epochs,[x["param"] for x in log], label='Mag')
            ax[2].legend()
        else :
            ax[2].set_title('Prob')
            for idx, _ in enumerate(log[0]["param"]):
                ax[2].plot(epochs,[x["param"][idx] for x in log], label='P'+str(idx))
            ax[2].legend() 
            #ax[2].legend(('P-0', 'P-45', 'P-180'))

    fig_name = fig_name.replace('.',',')
    plt.savefig(fig_name)

def plot_compare(filenames, fig_name='res'):

    all_data=[]
    legend=""
    for idx, file in enumerate(filenames):
        legend+=str(idx)+'-'+file+'\n'
        with open(file) as json_file:
            data = json.load(json_file)
            all_data.append(data)

    fig, ax = plt.subplots(ncols=3, figsize=(30, 8))

    for data_idx, log in enumerate(all_data):            
        log=log['Log']
        epochs = [x["epoch"] for x in log]

        ax[0].plot(epochs,[x["train_loss"] for x in log], label=str(data_idx)+'-Train')
        ax[0].plot(epochs,[x["val_loss"] for x in log], label=str(data_idx)+'-Val')
            
        ax[1].plot(epochs,[x["acc"] for x in log], label=str(data_idx)) 
        #ax[1].text(x=0.5,y=0,s=str(data_idx)+'-'+filenames[data_idx], transform=ax[1].transAxes)

        if log[0]["param"]!= None:
            if isinstance(log[0]["param"],float):
                ax[2].plot(epochs,[x["param"] for x in log], label=str(data_idx)+'-Mag')
                
            else :
                for idx, _ in enumerate(log[0]["param"]):
                    ax[2].plot(epochs,[x["param"][idx] for x in log], label=str(data_idx)+'-P'+str(idx))

    fig.suptitle(legend)
    ax[0].set_title('Loss')
    ax[1].set_title('Acc')
    ax[2].set_title('Param')
    for a in ax: a.legend()
    fig_name = fig_name.replace('.',',')

    plt.savefig(fig_name, bbox_inches='tight')

def viz_sample_data(imgs, labels, fig_name='data_sample'):

    sample = imgs[0:25,].permute(0, 2, 3, 1).squeeze().cpu()

    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(sample[i,], cmap=plt.cm.binary)
        plt.xlabel(labels[i].item())

    plt.savefig(fig_name)

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

def print_torch_mem(add_info=''):

    nb=0
    max_size=0
    for obj in gc.get_objects():
        #print(type(obj))
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)): # and len(obj.size())>1:
                #print(i, type(obj), obj.size())
                size = np.sum(obj.size())
                if(size>max_size): max_size=size
                nb+=1
        except:
            pass
    print(add_info, "-Pytroch tensor nb:",nb," / Max dim:", max_size)

    #print(add_info, "-Garbage size :",len(gc.garbage))

class loss_monitor(): #Voir https://github.com/pytorch/ignite
    def __init__(self, patience, end_train=1):
        self.patience = patience
        self.end_train = end_train
        self.counter = 0
        self.best_score = None
        self.reached_limit = 0

    def register(self, loss):
        if self.best_score is None:
            self.best_score = loss
        elif loss > self.best_score:
            self.counter += 1
            #if not self.reached_limit: 
            print("loss no improve counter", self.counter, self.reached_limit)
        else:
            self.best_score = loss
            self.counter = 0
    def limit_reached(self):
        if self.counter >= self.patience:
            self.counter = 0
            self.reached_limit +=1
            self.best_score = None
        return self.reached_limit

    def end_training(self):
        if self.limit_reached() >= self.end_train:
            return True
        else:
            return False

    def reset(self):
        self.__init__(self.patience, self.end_train)