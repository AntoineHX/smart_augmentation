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

import transformations as TF
class TF_loader(object):
    """ Transformations builder.

        See 'config' folder for pre-defined config files.

        Attributes:
            _filename (str): Path to config file (JSON) used.
            _TF_dict (dict): Transformations dictionnary built from config file.
            _TF_ignore_mag (set): Ensemble of transformations names for which magnitude should be ignored.
            _TF_names (list): List of transformations names/keys.
    """
    def __init__(self):
        """ Initialize TF_loader.

        """
        self._filename=''
        self._TF_dict={}
        self._TF_ignore_mag=set()
        self._TF_names=[]

    def load_TF_dict(self, filename):
        """ Build a TF dictionnary.

            Args:
                filename (str): Path to config file (JSON) defining the transformations.
            Returns:
                (dict, set) : TF dicttionnary built and ensemble of TF names for which mag should be ignored.
        """
        self._filename=filename
        self._TF_names=[]
        self._TF_dict={}
        self._TF_ignore_mag=set()

        with open(filename) as json_file:
            TF_params = json.load(json_file)

        for tf in TF_params:
            self._TF_names.append(tf['name'])
            if tf['function'] in TF.TF_ignore_mag:
                self._TF_ignore_mag.add(tf['name'])

            if tf['function'] == 'identity':
                self._TF_dict[tf['name']]=(lambda x, mag: x)

            elif tf['function'] == 'flip':
                #Inverser axes ?
                if tf['param']['axis'] == 'X':
                    self._TF_dict[tf['name']]=(lambda x, mag: TF.flipLR(x))
                elif tf['param']['axis'] == 'Y':
                    self._TF_dict[tf['name']]=(lambda x, mag: TF.flipUD(x))
                else:
                    raise Exception("Unknown TF axis : %s in %s"%(tf['function'], self._filename))

            elif tf['function'] in {'translate', 'shear'}:
                rand_fct= 'invScale_rand_floats' if tf['param']['invScale'] else 'rand_floats'
                self._TF_dict[tf['name']]=self.build_lambda(tf['function'], rand_fct, tf['param']['min'], tf['param']['max'], tf['param']['absolute'], tf['param']['axis'])
            
            else:
                rand_fct= 'invScale_rand_floats' if tf['param']['invScale'] else 'rand_floats'
                self._TF_dict[tf['name']]=self.build_lambda(tf['function'], rand_fct, tf['param']['min'], tf['param']['max'])

        return self._TF_dict, self._TF_ignore_mag

    def build_lambda(self, fct_name, rand_fct_name, minval, maxval, absolute=True, axis=None):
        """ Build a lambda function performing transformations.

            Args:
                fct_name (str): Name of the transformations to use (see transformations.py).
                rand_fct_name (str): Name of the random mapping function to use (see transformations.py).
                minval (float): minimum magnitude value of the TF.
                maxval (float): maximum magnitude value of the TF.
                absolute (bool): Wether the maxval should be relative (absolute=False) to the image size. (default: True)
                axis (str): Axis ('X' / 'Y') of the TF, if relevant. Should be used for (flip)/translate/shear functions. (default: None)

            Returns:
                (function) transformations function : Tensor=f(Tensor, magnitude)
        """
        if absolute:
            max_val_fct=(lambda x: maxval)
        else: #Relative to img size
            max_val_fct=(lambda x: x*maxval)

        if axis is None:
            return (lambda x, mag: 
                        getattr(TF, fct_name)(
                            x, 
                            getattr(TF, rand_fct_name)(
                                size=x.shape[0], 
                                mag=mag, 
                                minval=minval, 
                                maxval=maxval)))
        elif axis =='X':
            return (lambda x, mag: 
                getattr(TF, fct_name)(
                    x, 
                    TF.zero_stack(
                        getattr(TF, rand_fct_name)(
                            size=(x.shape[0],), 
                            mag=mag, 
                            minval=minval, 
                            maxval=max_val_fct(x.shape[2])),
                        zero_pos=0)))
        elif axis == 'Y':
            return (lambda x, mag: 
                getattr(TF, fct_name)(
                    x, 
                    TF.zero_stack(
                        getattr(TF, rand_fct_name)(
                            size=(x.shape[0],), 
                            mag=mag, 
                            minval=minval, 
                            maxval=max_val_fct(x.shape[3])),
                        zero_pos=1)))
        else:
            raise Exception("Unknown TF axis : %s in %s"%(fct_name, self._filename))

        def get_TF_names(self):
            return self._TF_names
        def get_TF_dict(self):
            return self._TF_dict

class ConfusionMatrix(object):
    """ Confusion matrix.

        Helps computing the confusion matrix and F1 scores.
        
        Example use ::
            confmat = ConfusionMatrix(...)

            confmat.reset()
            for data in dataset:
                ...
                confmat.update(...)

            confmat.f1_metric(...)

        Attributes:
            num_classes (int): Number of classes.
            mat (Tensor): Confusion matrix. Filled by update method.
    """
    def __init__(self, num_classes):
        """ Initialize ConfusionMatrix.

            Args:
                num_classes (int): Number of classes.
        """
        self.num_classes = num_classes
        self.mat = None

    def update(self, target, pred):
        """ Update the confusion matrix.

            Args:
                target (Tensor): Target labels.
                pred (Tensor): Prediction.
        """
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=target.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        """ Reset the Confusion matrix.

        """
        if self.mat is not None:
            self.mat.zero_()

    def f1_metric(self, average=None):
        """ Compute the F1 score.

            Inspired from : 
                https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
                https://discuss.pytorch.org/t/how-to-get-the-sensitivity-and-specificity-of-a-dataset/39373/6

            Args:
                average (str): Type of averaging performed on the data. (Default: None)
                    ``None``:
                        The scores for each class are returned.
                    ``'micro'``:
                        Calculate metrics globally by counting the total true positives,
                        false negatives and false positives.
                    ``'macro'``:
                        Calculate metrics for each label, and find their unweighted
                        mean.  This does not take label imbalance into account.
            Return:
                Tensor containing the F1 score. It's shape is either 1, if there was averaging, or (num_classes).
        """

        h = self.mat.float()
        TP = torch.diag(h)
        TN = []
        FP = []
        FN = []
        for c in range(self.num_classes):
            idx = torch.ones(self.num_classes).bool()
            idx[c] = 0
            # all non-class samples classified as non-class
            TN.append(self.mat[idx.nonzero()[:, None], idx.nonzero()].sum()) #conf_matrix[idx[:, None], idx].sum() - conf_matrix[idx, c].sum()
            # all non-class samples classified as class
            FP.append(self.mat[idx, c].sum())
            # all class samples not classified as class
            FN.append(self.mat[c, idx].sum())

            #print('Class {}\nTP {}, TN {}, FP {}, FN {}'.format(c, TP[c], TN[c], FP[c], FN[c]))

        tp = (TP/h.sum(1))#.sum()
        tn = (torch.tensor(TN, device=h.device, dtype=torch.float)/h.sum(1))#.sum()
        fp = (torch.tensor(FP, device=h.device, dtype=torch.float)/h.sum(1))#.sum()
        fn = (torch.tensor(FN, device=h.device, dtype=torch.float)/h.sum(1))#.sum()

        if average=="micro":
            tp, tn, fp, fn = tp.sum(), tn.sum(), fp.sum(), fn.sum()

        epsilon = 1e-7
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        
        f1 = 2* (precision*recall) / (precision + recall + epsilon)

        if average=="macro":
            f1=f1.mean()
        return f1

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
        
    ax[1, 0].set_title('Test')
    ax[1, 0].plot(epochs,[x["acc"] for x in log], label='Acc') 
    
    if "f1" in log[0].keys():
        #ax[1, 0].plot(epochs,[x["f1"]*100 for x in log], label='F1')
        #'''
        #print(log[0]["f1"])
        if isinstance(log[0]["f1"], list):
            for c in range(len(log[0]["f1"])):
                ax[1, 0].plot(epochs,[x["f1"][c]*100 for x in log], label='F1-'+str(c), ls='--')
        else:
            ax[1, 0].plot(epochs,[x["f1"]*100 for x in log], label='F1', ls='--')
        #'''

    ax[1, 0].legend()

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
            

    fig_name = fig_name.replace('.',',').replace(',,/','../')
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