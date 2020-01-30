""" Utilties function.

"""
import numpy as np
import json, math, time, os
import matplotlib.pyplot as plt
import copy
import gc

from torchviz import make_dot 

import torch
import torch.nn.functional as F

import time

def print_graph(PyTorch_obj, fig_name='graph'):
    """Save the computational graph.

        Args:
            PyTorch_obj (Tensor): End of the graph. Commonly, the loss tensor to get the whole graph.
            fig_name (string): Relative path where to save the graph. (default: graph)
    """
    graph=make_dot(PyTorch_obj)
    graph.format = 'pdf' #https://graphviz.readthedocs.io/en/stable/manual.html#formats
    graph.render(fig_name)

def plot_resV2(log, fig_name='res', param_names=None):
    """Save a visual graph of the logs.

        Args:
            log (dict): Logs of the training generated by most of train_utils.
            fig_name (string): Relative path where to save the graph. (default: res)
            param_names (list): Labels for the parameters. (default: None)
    """
    epochs = [x["epoch"] for x in log]

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(30, 15))

    ax[0, 0].set_title('Loss')
    ax[0, 0].plot(epochs,[x["train_loss"] for x in log], label='Train')
    ax[0, 0].plot(epochs,[x["val_loss"] for x in log], label='Val')
    ax[0, 0].legend()
        
    ax[1, 0].set_title('Acc')
    ax[1, 0].plot(epochs,[x["acc"] for x in log]) 

    if log[0]["param"]!= None:
        if not param_names : param_names = ['P'+str(idx) for idx, _ in enumerate(log[0]["param"])]
        #proba=[[x["param"][idx] for x in log] for idx, _ in enumerate(log[0]["param"])]
        proba=[[x["param"][idx]['p'] for x in log] for idx, _ in enumerate(log[0]["param"])]
        mag=[[x["param"][idx]['m'] for x in log] for idx, _ in enumerate(log[0]["param"])]

        ax[0, 1].set_title('Prob =f(epoch)')
        ax[0, 1].stackplot(epochs, proba, labels=param_names)
        #ax[0, 1].legend(param_names, loc='center left', bbox_to_anchor=(1, 0.5)) 

        ax[1, 1].set_title('Prob =f(TF)')
        mean = np.mean(proba, axis=1)
        std = np.std(proba, axis=1)
        ax[1, 1].bar(param_names, mean, yerr=std)
        plt.sca(ax[1, 1]), plt.xticks(rotation=90)

        ax[0, 2].set_title('Mag =f(epoch)')
        ax[0, 2].stackplot(epochs, mag, labels=param_names)
        #ax[0, 2].plot(epochs, np.array(mag).T, label=param_names)
        ax[0, 2].legend(param_names, loc='center left', bbox_to_anchor=(1, 0.5)) 

        ax[1, 2].set_title('Mag =f(TF)')
        mean = np.mean(mag, axis=1)
        std = np.std(mag, axis=1)
        ax[1, 2].bar(param_names, mean, yerr=std)
        plt.sca(ax[1, 2]), plt.xticks(rotation=90)
            

    fig_name = fig_name.replace('.',',')
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close()

def plot_compare(filenames, fig_name='res'):
    """Save a visual graph comparing trainings stats.

        Args:
            filenames (list[Strings]): Relative paths to the logs (JSON files).
            fig_name (string): Relative path where to save the graph. (default: res)
    """
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
    plt.close()

def viz_sample_data(imgs, labels, fig_name='data_sample', weight_labels=None):
    """Save data samples.

        Args:
            imgs (Tensor): Batch of image to sample from. Intended to contain at least 25 images.
            labels (Tensor): Labels of the images.
            fig_name (string): Relative path where to save the graph. (default: data_sample)
            weight_labels (Tensor): Weights associated to each labels. (default: None)
    """

    sample = imgs[0:25,].permute(0, 2, 3, 1).squeeze().cpu()

    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1) #Trop de figure cree ?
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(sample[i,].detach().numpy(), cmap=plt.cm.binary)
        label = str(labels[i].item())
        if weight_labels is not None : label+= (" - p %.2f" % weight_labels[i].item())
        plt.xlabel(label)

    plt.savefig(fig_name)
    print("Sample saved :", fig_name)
    plt.close('all')

def print_torch_mem(add_info=''):
    """Print informations on PyTorch memory usage.

        Args:
            add_info (string): Prefix added before the print. (default: None)
    """
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

    """Simple GPU memory report."""

    mega_bytes = 1024.0 * 1024.0
    string = add_info + ' memory (MB)'
    string += ' | allocated: {}'.format(
        torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | cached: {}'.format(torch.cuda.memory_cached() / mega_bytes)
    string += ' | max cached: {}'.format(
        torch.cuda.max_memory_cached()/ mega_bytes)
    print(string)

'''
def plot_TF_influence(log, fig_name='TF_influence', param_names=None):
    proba=[[x["param"][idx]['p'] for x in log] for idx, _ in enumerate(log[0]["param"])]
    mag=[[x["param"][idx]['m'] for x in log] for idx, _ in enumerate(log[0]["param"])]

    plt.figure()

    mean = np.mean(proba, axis=1)*np.mean(mag, axis=1) #Pourrait etre interessant de multiplier avant le mean
    std = np.std(proba, axis=1)*np.std(mag, axis=1)
    plt.bar(param_names, mean, yerr=std)

    plt.xticks(rotation=90)
    fig_name = fig_name.replace('.',',')
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close()
'''

from torch._six import inf
def clip_norm(tensors, max_norm, norm_type=2):
    """Clips norm of passed tensors.
        The norm is computed over all tensors together, as if they were
        concatenated into a single vector. Clipped tensors are returned.
        
        See: https://github.com/facebookresearch/higher/issues/18

        Args:
            tensors (Iterable[Tensor]): an iterable of Tensors or a
                single Tensor to be normalized.
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.
        Returns:
          Clipped (List[Tensor]) tensors.
    """
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    tensors = list(tensors)
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(t.abs().max() for t in tensors)
    else:
        total_norm = 0
        for t in tensors:
            param_norm = t.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef >= 1:
        return tensors
    return [t.mul(clip_coef) for t in tensors]