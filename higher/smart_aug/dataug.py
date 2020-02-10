""" Data augmentation modules.

    Features a custom implementaiton of RandAugment (RandAug), as well as a data augmentation modules allowing gradient propagation.

    Typical usage:

        aug_model = Augmented_model(Data_AugV5, model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import *

#import kornia
#import random
import numpy as np
import copy

import transformations as TF

import higher
import higher_patch

from utils import clip_norm 
from train_utils import compute_vaLoss

### Data augmenter ###
class Data_augV5(nn.Module): #Optimisation jointe (mag, proba)
    """Data augmentation module with learnable parameters.

        Applies transformations (TF) to batch of data.
        Each TF is defined by a (name, probability of application, magnitude of distorsion) tuple which can be learned. For the full definiton of the TF, see transformations.py.
        The TF probabilities defines a distribution from which we sample the TF applied.

        Be warry, that the order of sequential application of TF is not taken into account. See Data_augV7.

        Attributes:
            _data_augmentation (bool): Wether TF will be applied during forward pass.
            _TF_dict (dict) : A dictionnary containing the data transformations (TF) to be applied.
            _TF (list) : List of TF names.
            _nb_tf (int) : Number of TF used.
            _N_seqTF (int) : Number of TF to be applied sequentially to each inputs
            _shared_mag (bool) : Wether to share a single magnitude parameters for all TF. Beware using shared mag with basic color TF as their lowest magnitude is at PARAMETER_MAX/2.
            _fixed_mag (bool): Wether to lock the TF magnitudes.
            _fixed_prob (bool): Wether to lock the TF probabilies.
            _samples (list): Sampled TF index during last forward pass.
            _mix_dist (bool): Wether we use a mix of an uniform distribution and the real distribution (TF probabilites). If False, only a uniform distribution is used.
            _fixed_mix (bool): Wether we lock the mix distribution factor.
            _params (nn.ParameterDict): Learnable parameters.
            _reg_tgt (Tensor): Target for the magnitude regularisation. Only used when _fixed_mag is set to false (ie. we learn the magnitudes).
            _reg_mask (list): Mask selecting the TF considered for the regularisation.
    """
    def __init__(self, TF_dict=TF.TF_dict, N_TF=1, mix_dist=0.0, fixed_prob=False, fixed_mag=True, shared_mag=True):
        """Init Data_augv5.

            Args:
                TF_dict (dict): A dictionnary containing the data transformations (TF) to be applied. (default: use all available TF from transformations.py)
                N_TF (int): Number of TF to be applied sequentially to each inputs. (default: 1)
                mix_dist (float): Proportion [0.0, 1.0] of the real distribution used for sampling/selection of the TF. Distribution = (1-mix_dist)*Uniform_distribution + mix_dist*Real_distribution. If None is given, try to learn this parameter. (default: 0)
                fixed_prob (bool): Wether to lock the TF probabilies. (default: False)
                fixed_mag (bool): Wether to lock the TF magnitudes. (default: True)
                shared_mag (bool): Wether to share a single magnitude parameters for all TF. (default: True)
        """
        super(Data_augV5, self).__init__()
        assert len(TF_dict)>0
        assert N_TF>=0
        
        self._data_augmentation = True

        #TF
        self._TF_dict = TF_dict
        self._TF= list(self._TF_dict.keys())
        self._nb_tf= len(self._TF)
        self._N_seqTF = N_TF

        #Mag
        self._shared_mag = shared_mag
        self._fixed_mag = fixed_mag
        if not self._fixed_mag and len([tf for tf in self._TF if tf not in TF.TF_ignore_mag])==0:
            print("WARNING: Mag would be fixed as current TF doesn't allow gradient propagation:",self._TF)
            self._fixed_mag=True

        #Distribution
        self._fixed_prob=fixed_prob
        self._samples = []

        self._mix_dist = False
        if mix_dist != 0.0: #Mix dist
            self._mix_dist = True

        self._fixed_mix=True
        if mix_dist is None: #Learn Mix dist
            self._fixed_mix = False
            mix_dist=0.5
        
        #Params
        init_mag = float(TF.PARAMETER_MAX) if self._fixed_mag else float(TF.PARAMETER_MAX)/2
        self._params = nn.ParameterDict({
            "prob": nn.Parameter(torch.ones(self._nb_tf)/self._nb_tf), #Distribution prob uniforme
            "mag" : nn.Parameter(torch.tensor(init_mag) if self._shared_mag
                            else torch.tensor(init_mag).repeat(self._nb_tf)), #[0, PARAMETER_MAX]
            "mix_dist": nn.Parameter(torch.tensor(mix_dist).clamp(min=0.0,max=0.999))
        })

        #for tf in TF.TF_no_grad :
        #    if tf in self._TF: self._params['mag'].data[self._TF.index(tf)]=float(TF.PARAMETER_MAX) #TF fixe a max parameter
        #for t in TF.TF_no_mag: self._params['mag'][self._TF.index(t)].data-=self._params['mag'][self._TF.index(t)].data #Mag inutile pour les TF ignore_mag

        #Mag regularisation
        if not self._fixed_mag:
            if  self._shared_mag :
                self._reg_tgt = torch.tensor(TF.PARAMETER_MAX, dtype=torch.float) #Encourage amplitude max
            else:
                self._reg_mask=[self._TF.index(t) for t in self._TF if t not in TF.TF_ignore_mag]
                self._reg_tgt=torch.full(size=(len(self._reg_mask),), fill_value=TF.PARAMETER_MAX) #Encourage amplitude max

    def forward(self, x):
        """ Main method of the Data augmentation module.

            Args:
                x (Tensor): Batch of data.

            Returns:
                Tensor : Batch of tranformed data.
        """
        self._samples = []
        if self._data_augmentation:# and TF.random.random() < 0.5:
            device = x.device
            batch_size, h, w = x.shape[0], x.shape[2], x.shape[3]

            x = copy.deepcopy(x) #Evite de modifier les echantillons par reference (Problematique pour des utilisations paralleles)
            
            ## Echantillonage ##
            uniforme_dist = torch.ones(1,self._nb_tf,device=device).softmax(dim=1)

            if not self._mix_dist:
                self._distrib = uniforme_dist        
            else:
                prob = self._params["prob"].detach() if self._fixed_prob else self._params["prob"]
                mix_dist = self._params["mix_dist"].detach() if self._fixed_mix else self._params["mix_dist"]
                self._distrib = (mix_dist*prob+(1-mix_dist)*uniforme_dist)#.softmax(dim=1) #Mix distrib reel / uniforme avec mix_factor

            cat_distrib= Categorical(probs=torch.ones((batch_size, self._nb_tf), device=device)*self._distrib)
            
            for _ in range(self._N_seqTF):
                
                sample = cat_distrib.sample()
                self._samples.append(sample)

                ## Transformations ##
                x = self.apply_TF(x, sample)
        return x

    def apply_TF(self, x, sampled_TF):
        """ Applies the sampled transformations.

            Args:
                x (Tensor): Batch of data.
                sampled_TF (Tensor): Indexes of the TF to be applied to each element of data.

            Returns:
                Tensor: Batch of tranformed data.
        """
        device = x.device
        batch_size, channels, h, w = x.shape
        smps_x=[]
        
        for tf_idx in range(self._nb_tf):
            mask = sampled_TF==tf_idx #Create selection mask
            smp_x = x[mask] #torch.masked_select() ? (Necessite d'expand le mask au meme dim)

            if smp_x.shape[0]!=0: #if there's data to TF
                magnitude=self._params["mag"] if self._shared_mag else self._params["mag"][tf_idx]
                if self._fixed_mag: magnitude=magnitude.detach() #Fmodel tente systematiquement de tracker les gradient de tout les param

                tf=self._TF[tf_idx]

                #In place
                #x[mask]=self._TF_dict[tf](x=smp_x, mag=magnitude)

                #Out of place
                smp_x = self._TF_dict[tf](x=smp_x, mag=magnitude)
                idx= mask.nonzero()
                idx= idx.expand(-1,channels).unsqueeze(dim=2).expand(-1,channels, h).unsqueeze(dim=3).expand(-1,channels, h, w) #Il y a forcement plus simple ...
                x=x.scatter(dim=0, index=idx, src=smp_x)
                                
        return x

    def adjust_param(self, soft=False): #Detach from gradient ?
        """ Enforce limitations to the learned parameters.

            Ensure that the parameters value stays in the right intevals. This should be called after each update of those parameters.

            Args:
                soft (bool): Wether to use a softmax function for TF probabilites. Tends to lock the probabilities if the learning rate is low, preventing them to be learned. (default: False)
        """
        if not self._fixed_prob:
            if soft :
                self._params['prob'].data=F.softmax(self._params['prob'].data, dim=0)
            else:
                self._params['prob'].data = self._params['prob'].data.clamp(min=1/(self._nb_tf*100),max=1.0)
                self._params['prob'].data = self._params['prob']/sum(self._params['prob']) #Contrainte sum(p)=1

        if not self._fixed_mag:
            self._params['mag'].data = self._params['mag'].data.clamp(min=TF.PARAMETER_MIN, max=TF.PARAMETER_MAX)

        if not self._fixed_mix:
            self._params['mix_dist'].data = self._params['mix_dist'].data.clamp(min=0.0, max=0.999)

    def loss_weight(self):
        """ Weights for the loss.
            Compute the weights for the loss of each inputs depending on wich TF was applied to them.
            Should be applied to the loss before reduction.
            
            Do not take into account the order of application of the TF. See Data_augV7.

            Returns:
                Tensor : Loss weights.
        """
        if len(self._samples)==0 : return torch.tensor(1, device=self._params["prob"].device) #Pas d'echantillon = pas de ponderation

        prob = self._params["prob"].detach() if self._fixed_prob else self._params["prob"]
        
        #Plusieurs TF sequentielles (Attention ne prend pas en compte ordre !)
        w_loss = torch.zeros((self._samples[0].shape[0],self._nb_tf), device=self._samples[0].device)
        for sample in self._samples:
            tmp_w = torch.zeros(w_loss.size(),device=w_loss.device)
            tmp_w.scatter_(dim=1, index=sample.view(-1,1), value=1/self._N_seqTF)
            w_loss += tmp_w

        w_loss = w_loss * prob/self._distrib #Ponderation par les proba (divisee par la distrib pour pas diminuer la loss)
        w_loss = torch.sum(w_loss,dim=1)
        return w_loss

    def reg_loss(self, reg_factor=0.005):
        """ Regularisation term used to learn the magnitudes.
            Use an L2 loss to encourage high magnitudes TF.

            Args:
                reg_factor (float): Factor by wich the regularisation loss is multiplied. (default: 0.005)
            Returns:
                Tensor containing the regularisation loss value.
        """
        if self._fixed_mag:
            return torch.tensor(0)
        else:
            #return reg_factor * F.l1_loss(self._params['mag'][self._reg_mask], target=self._reg_tgt, reduction='mean') 
            mags = self._params['mag'] if self._params['mag'].shape==torch.Size([]) else self._params['mag'][self._reg_mask]
            max_mag_reg = reg_factor * F.mse_loss(mags, target=self._reg_tgt.to(mags.device), reduction='mean')
            return max_mag_reg

    def train(self, mode=True):
        """ Set the module training mode.

            Args:
                mode (bool): Wether to learn the parameter of the module. None would not change mode. (default: None)
        """
        #if mode is None :
        #    mode=self._data_augmentation
        self.augment(mode=mode) #Inutile si mode=None
        super(Data_augV5, self).train(mode)
        return self

    def eval(self):
        """ Set the module to evaluation mode.
        """
        return self.train(mode=False)

    def augment(self, mode=True):
        """ Set the augmentation mode.

            Args:
                mode (bool): Wether to perform data augmentation on the forward pass. (default: True)
        """
        self._data_augmentation=mode

    def is_augmenting(self):
        """ Return wether data augmentation is applied.

            Returns:
                bool : True if data augmentation is applied.
        """
        return self._data_augmentation

    def __getitem__(self, key):
        """Access to the learnable parameters
        Args:
            key (string): Name of the learnable parameter to access.

        Returns:
            nn.Parameter.
        """
        return self._params[key]

    def __str__(self):
        """Name of the module

            Returns:
                String containing the name of the module as well as the higher levels parameters.
        """
        dist_param=''
        if self._fixed_prob: dist_param+='Fx'
        mag_param='Mag'
        if self._fixed_mag: mag_param+= 'Fx'
        if self._shared_mag: mag_param+= 'Sh'
        if not self._mix_dist:
            return "Data_augV5(Uniform%s-%dTFx%d-%s)" % (dist_param, self._nb_tf, self._N_seqTF, mag_param)
        elif self._fixed_mix:
            return "Data_augV5(Mix%.1f%s-%dTFx%d-%s)" % (self._params['mix_dist'].item(),dist_param, self._nb_tf, self._N_seqTF, mag_param)
        else:
            return "Data_augV5(Mix%s-%dTFx%d-%s)" % (dist_param, self._nb_tf, self._N_seqTF, mag_param)

class Data_augV7(nn.Module): #Proba sequentielles
    """Data augmentation module with learnable parameters.

        Applies transformations (TF) to batch of data.
        Each TF is defined by a (name, probability of application, magnitude of distorsion) tuple which can be learned. For the full definiton of the TF, see transformations.py.
        The TF probabilities defines a distribution from which we sample the TF applied.

        Replace the use of TF by TF sets which are combinaisons of classic TF.

        Attributes:
            _data_augmentation (bool): Wether TF will be applied during forward pass.
            _TF_dict (dict) : A dictionnary containing the data transformations (TF) to be applied.
            _TF (list) : List of TF names.
            _nb_tf (int) : Number of TF used.
            _N_seqTF (int) : Number of TF to be applied sequentially to each inputs
            _shared_mag (bool) : Wether to share a single magnitude parameters for all TF. Beware using shared mag with basic color TF as their lowest magnitude is at PARAMETER_MAX/2.
            _fixed_mag (bool): Wether to lock the TF magnitudes.
            _fixed_prob (bool): Wether to lock the TF probabilies.
            _samples (list): Sampled TF index during last forward pass.
            _mix_dist (bool): Wether we use a mix of an uniform distribution and the real distribution (TF probabilites). If False, only a uniform distribution is used.
            _fixed_mix (bool): Wether we lock the mix distribution factor.
            _params (nn.ParameterDict): Learnable parameters.
            _reg_tgt (Tensor): Target for the magnitude regularisation. Only used when _fixed_mag is set to false (ie. we learn the magnitudes).
            _reg_mask (list): Mask selecting the TF considered for the regularisation.
    """
    def __init__(self, TF_dict=TF.TF_dict, N_TF=2, mix_dist=0.0, fixed_prob=False, fixed_mag=True, shared_mag=True):
        """Init Data_augv7.

            Args:
                TF_dict (dict): A dictionnary containing the data transformations (TF) to be applied. (default: use all available TF from transformations.py)
                N_TF (int): Number of TF to be applied sequentially to each inputs. Minimum 2, otherwise prefer using Data_augV5. (default: 2)
                mix_dist (float): Proportion [0.0, 1.0] of the real distribution used for sampling/selection of the TF. Distribution = (1-mix_dist)*Uniform_distribution + mix_dist*Real_distribution. If None is given, try to learn this parameter. (default: 0)
                fixed_prob (bool): Wether to lock the TF probabilies. (default: False)
                fixed_mag (bool): Wether to lock the TF magnitudes. (default: True)
                shared_mag (bool): Wether to share a single magnitude parameters for all TF. (default: True)
        """
        super(Data_augV7, self).__init__()
        assert len(TF_dict)>0
        assert N_TF>=0

        if N_TF<2:
            print("WARNING: Data_augv7 isn't designed to use less than 2 sequentials TF. Please use Data_augv5 instead.")
        
        self._data_augmentation = True

        #TF
        self._TF_dict = TF_dict
        self._TF= list(self._TF_dict.keys())
        self._nb_tf= len(self._TF)
        self._N_seqTF = N_TF

        #Mag
        self._shared_mag = shared_mag
        self._fixed_mag = fixed_mag
        if not self._fixed_mag and len([tf for tf in self._TF if tf not in TF.TF_ignore_mag])==0:
            print("WARNING: Mag would be fixed as current TF doesn't allow gradient propagation:",self._TF)
            self._fixed_mag=True

        #Distribution
        self._fixed_prob=fixed_prob
        self._samples = []

        self._mix_dist = False
        if mix_dist != 0.0: #Mix dist
            self._mix_dist = True

        self._fixed_mix=True
        if mix_dist is None: #Learn Mix dist
            self._fixed_mix = False
            mix_dist=0.5
        
        #TF sets
        #import itertools
        #itertools.product(range(self._nb_tf), repeat=self._N_seqTF)

        #no_consecutive={idx for idx, t in enumerate(self._TF) if t in {'FlipUD', 'FlipLR'}} #Specific No consecutive ops
        no_consecutive={idx for idx, t in enumerate(self._TF) if t not in {'Identity'}} #No consecutive same ops (except Identity)
        cons_test = (lambda i, idxs:  i in no_consecutive and len(idxs)!=0 and i==idxs[-1]) #Exclude selected consecutive
        def generate_TF_sets(n_TF, set_size, idx_prefix=[]): #Generate every arrangement (with reuse) of TF (exclude cons_test arrangement)
            TF_sets=[]
            if set_size>1:
                for i in range(n_TF):
                    if not cons_test(i, idx_prefix):
                        TF_sets += generate_TF_sets(n_TF, set_size=set_size-1, idx_prefix=idx_prefix+[i])
            else:
                TF_sets+=[[idx_prefix+[i]] for i in range(n_TF) if not cons_test(i, idx_prefix)]
            return TF_sets

        self._TF_sets=torch.ByteTensor(generate_TF_sets(self._nb_tf, self._N_seqTF)).squeeze()
        self._nb_TF_sets=len(self._TF_sets)
        print("Number of TF sets:",self._nb_TF_sets)
        #print(self._TF_sets)
        self._prob_mem=torch.zeros(self._nb_TF_sets)

        #Params
        init_mag = float(TF.PARAMETER_MAX) if self._fixed_mag else float(TF.PARAMETER_MAX)/2
        self._params = nn.ParameterDict({
            "prob": nn.Parameter(torch.ones(self._nb_TF_sets)/self._nb_TF_sets), #Distribution prob uniforme
            "mag" : nn.Parameter(torch.tensor(init_mag) if self._shared_mag
                            else torch.tensor(init_mag).repeat(self._nb_tf)), #[0, PARAMETER_MAX]
            "mix_dist": nn.Parameter(torch.tensor(mix_dist).clamp(min=0.0,max=0.999))
        })

        #for tf in TF.TF_no_grad :
        #    if tf in self._TF: self._params['mag'].data[self._TF.index(tf)]=float(TF.PARAMETER_MAX) #TF fixe a max parameter
        #for t in TF.TF_no_mag: self._params['mag'][self._TF.index(t)].data-=self._params['mag'][self._TF.index(t)].data #Mag inutile pour les TF ignore_mag

        #Mag regularisation
        if not self._fixed_mag:
            if  self._shared_mag :
                self._reg_tgt = torch.FloatTensor(TF.PARAMETER_MAX) #Encourage amplitude max
            else:
                self._reg_mask=[idx for idx,t in enumerate(self._TF) if t not in TF.TF_ignore_mag]
                self._reg_tgt=torch.full(size=(len(self._reg_mask),), fill_value=TF.PARAMETER_MAX) #Encourage amplitude max

    def forward(self, x):
        """ Main method of the Data augmentation module.

            Args:
                x (Tensor): Batch of data.

            Returns:
                Tensor : Batch of tranformed data.
        """
        self._samples = None
        if self._data_augmentation:# and TF.random.random() < 0.5:
            device = x.device
            batch_size, h, w = x.shape[0], x.shape[2], x.shape[3]

            x = copy.deepcopy(x) #Evite de modifier les echantillons par reference (Problematique pour des utilisations paralleles)
            

            ## Echantillonage ##
            uniforme_dist = torch.ones(1,self._nb_TF_sets,device=device).softmax(dim=1)

            if not self._mix_dist:
                self._distrib = uniforme_dist        
            else:
                prob = self._params["prob"].detach() if self._fixed_prob else self._params["prob"]
                mix_dist = self._params["mix_dist"].detach() if self._fixed_mix else self._params["mix_dist"]
                self._distrib = (mix_dist*prob+(1-mix_dist)*uniforme_dist)#.softmax(dim=1) #Mix distrib reel / uniforme avec mix_factor

            cat_distrib= Categorical(probs=torch.ones((batch_size, self._nb_TF_sets), device=device)*self._distrib)
            sample = cat_distrib.sample()
            
            self._samples=sample
            TF_samples=self._TF_sets[sample,:].to(device) #[Batch_size, TFseq]

            for i in range(self._N_seqTF):
                ## Transformations ##
                x = self.apply_TF(x, TF_samples[:,i])
        return x

    def apply_TF(self, x, sampled_TF):
        """ Applies the sampled transformations.

            Args:
                x (Tensor): Batch of data.
                sampled_TF (Tensor): Indexes of the TF to be applied to each element of data.

            Returns:
                Tensor: Batch of tranformed data.
        """
        device = x.device
        batch_size, channels, h, w = x.shape
        smps_x=[]
        
        for tf_idx in range(self._nb_tf):
            mask = sampled_TF==tf_idx #Create selection mask
            smp_x = x[mask] #torch.masked_select() ? (Necessite d'expand le mask au meme dim)

            if smp_x.shape[0]!=0: #if there's data to TF
                magnitude=self._params["mag"] if self._shared_mag else self._params["mag"][tf_idx]
                if self._fixed_mag: magnitude=magnitude.detach() #Fmodel tente systematiquement de tracker les gradient de tout les param

                tf=self._TF[tf_idx]

                #In place
                #x[mask]=self._TF_dict[tf](x=smp_x, mag=magnitude)

                #Out of place
                smp_x = self._TF_dict[tf](x=smp_x, mag=magnitude)
                idx= mask.nonzero()
                idx= idx.expand(-1,channels).unsqueeze(dim=2).expand(-1,channels, h).unsqueeze(dim=3).expand(-1,channels, h, w) #Il y a forcement plus simple ...
                x=x.scatter(dim=0, index=idx, src=smp_x)
                                
        return x

    def adjust_param(self, soft=False): #Detach from gradient ?
        """ Enforce limitations to the learned parameters.

            Ensure that the parameters value stays in the right intevals. This should be called after each update of those parameters.

            Args:
                soft (bool): Wether to use a softmax function for TF probabilites. Not Recommended as it tends to lock the probabilities, preventing them to be learned. (default: False)
        """
        if not self._fixed_prob:
            if soft :
                self._params['prob'].data=F.softmax(self._params['prob'].data, dim=0) #Trop 'soft', bloque en dist uniforme si lr trop faible
            else:
                self._params['prob'].data = self._params['prob'].data.clamp(min=1/(self._nb_tf*100),max=1.0)
                self._params['prob'].data = self._params['prob']/sum(self._params['prob']) #Contrainte sum(p)=1

        if not self._fixed_mag:
            self._params['mag'].data = self._params['mag'].data.clamp(min=TF.PARAMETER_MIN, max=TF.PARAMETER_MAX)

        if not self._fixed_mix:
            self._params['mix_dist'].data = self._params['mix_dist'].data.clamp(min=0.0, max=0.999)

    def loss_weight(self):
        """ Weights for the loss.
            Compute the weights for the loss of each inputs depending on wich TF was applied to them.
            Should be applied to the loss before reduction.

            Returns:
                Tensor : Loss weights.
        """
        if self._samples is None : return 1 #Pas d'echantillon = pas de ponderation

        prob = self._params["prob"].detach() if self._fixed_prob else self._params["prob"]
        
        w_loss = torch.zeros((self._samples.shape[0],self._nb_TF_sets), device=self._samples.device)
        w_loss.scatter_(1, self._samples.view(-1,1), 1)
        w_loss = w_loss * prob/self._distrib #Ponderation par les proba (divisee par la distrib pour pas diminuer la loss)
        w_loss = torch.sum(w_loss,dim=1)
        return w_loss

    def reg_loss(self, reg_factor=0.005):
        """ Regularisation term used to learn the magnitudes.
            Use an L2 loss to encourage high magnitudes TF.

            Args:
                reg_factor (float): Factor by wich the regularisation loss is multiplied. (default: 0.005)
            Returns:
                Tensor containing the regularisation loss value.
        """
        if self._fixed_mag:
            return torch.tensor(0)
        else:
            #return reg_factor * F.l1_loss(self._params['mag'][self._reg_mask], target=self._reg_tgt, reduction='mean') 
            mags = self._params['mag'] if self._params['mag'].shape==torch.Size([]) else self._params['mag'][self._reg_mask]
            max_mag_reg = reg_factor * F.mse_loss(mags, target=self._reg_tgt.to(mags.device), reduction='mean')
            return max_mag_reg

    def TF_prob(self):
        """ Gives an estimation of the individual TF probabilities.

            Be warry that the probability returned isn't exact. The TF distribution isn't fully represented by those.
            Each probability should be taken individualy. They only represent the chance for a specific TF to be picked at least once.

            Returms:
                Tensor containing the single TF probabilities of applications. 
        """
        if torch.all(self._params['prob']!=self._prob_mem.to(self._params['prob'].device)): #Prevent recompute if originial prob didn't changed
            self._prob_mem=self._params['prob'].data.detach_()
            self._single_TF_prob=torch.zeros(self._nb_tf)
            for idx_tf in range(self._nb_tf):
                for i, t_set in enumerate(self._TF_sets):
                    #uni, count = np.unique(t_set, return_counts=True)
                    #if idx_tf in uni:
                    #    res[idx_tf]+=self._params['prob'][i]*int(count[np.where(uni==idx_tf)])
                    if idx_tf in t_set:
                        self._single_TF_prob[idx_tf]+=self._params['prob'][i]

        return self._single_TF_prob

    def train(self, mode=True):
        """ Set the module training mode.

            Args:
                mode (bool): Wether to learn the parameter of the module. None would not change mode. (default: None)
        """
        #if mode is None :
        #    mode=self._data_augmentation
        self.augment(mode=mode) #Inutile si mode=None
        super(Data_augV7, self).train(mode)
        return self

    def eval(self):
        """ Set the module to evaluation mode.
        """
        return self.train(mode=False)

    def augment(self, mode=True):
        """ Set the augmentation mode.

            Args:
                mode (bool): Wether to perform data augmentation on the forward pass. (default: True)
        """
        self._data_augmentation=mode

    def is_augmenting(self):
        """ Return wether data augmentation is applied.

            Returns:
                bool : True if data augmentation is applied.
        """
        return self._data_augmentation

    def __getitem__(self, key):
        """Access to the learnable parameters
        Args:
            key (string): Name of the learnable parameter to access.

        Returns:
            nn.Parameter.
        """
        if key == 'prob': #Override prob access
            return self.TF_prob()
        return self._params[key]

    def __str__(self):
        """Name of the module

            Returns:
                String containing the name of the module as well as the higher levels parameters.
        """
        dist_param=''
        if self._fixed_prob: dist_param+='Fx'
        mag_param='Mag'
        if self._fixed_mag: mag_param+= 'Fx'
        if self._shared_mag: mag_param+= 'Sh'
        if not self._mix_dist:
            return "Data_augV7(Uniform%s-%dTFx%d-%s)" % (dist_param, self._nb_tf, self._N_seqTF, mag_param)
        elif self._fixed_mix:
            return "Data_augV7(Mix%.1f%s-%dTFx%d-%s)" % (self._params['mix_dist'].item(),dist_param, self._nb_tf, self._N_seqTF, mag_param)
        else:
            return "Data_augV7(Mix%s-%dTFx%d-%s)" % (dist_param, self._nb_tf, self._N_seqTF, mag_param)

class RandAug(nn.Module): #RandAugment = UniformFx-MagFxSh + rapide
    """RandAugment implementation.

        Applies transformations (TF) to batch of data.
        Each TF is defined by a (name, probability of application, magnitude of distorsion) tuple. For the full definiton of the TF, see transformations.py.
        The TF probabilities are ignored and, instead selected randomly.

        Attributes:
            _data_augmentation (bool): Wether TF will be applied during forward pass.
            _TF_dict (dict) : A dictionnary containing the data transformations (TF) to be applied.
            _TF (list) : List of TF names.
            _nb_tf (int) : Number of TF used.
            _N_seqTF (int) : Number of TF to be applied sequentially to each inputs
            _shared_mag (bool) : Wether to share a single magnitude parameters for all TF. Should be True.
            _fixed_mag (bool): Wether to lock the TF magnitudes. Should be True.
            _params (nn.ParameterDict): Data augmentation parameters.
    """
    def __init__(self, TF_dict=TF.TF_dict, N_TF=1, mag=TF.PARAMETER_MAX):
        """Init RandAug.

            Args:
            TF_dict (dict): A dictionnary containing the data transformations (TF) to be applied. (default: use all available TF from transformations.py)
            N_TF (int): Number of TF to be applied sequentially to each inputs. (default: 1)
            mag (float): Magnitude of the TF. Should be between [PARAMETER_MIN, PARAMETER_MAX] defined in transformations.py. (default: PARAMETER_MAX)
        """
        super(RandAug, self).__init__()

        self._data_augmentation = True

        self._TF_dict = TF_dict
        self._TF= list(self._TF_dict.keys())
        self._nb_tf= len(self._TF)
        self._N_seqTF = N_TF

        self.mag=nn.Parameter(torch.tensor(float(mag)))
        self._params = nn.ParameterDict({
            "prob": nn.Parameter(torch.ones(self._nb_tf)/self._nb_tf), #Ignored
            "mag" : nn.Parameter(torch.tensor(float(mag))),
        })
        self._shared_mag = True
        self._fixed_mag = True
        self._fixed_prob=True
        self._fixed_mix=True

        self._params['mag'].data = self._params['mag'].data.clamp(min=TF.PARAMETER_MIN, max=TF.PARAMETER_MAX)

    def forward(self, x):
        """ Main method of the Data augmentation module.

            Args:
                x (Tensor): Batch of data.

            Returns:
                Tensor : Batch of tranformed data.
        """
        if self._data_augmentation:# and TF.random.random() < 0.5:
            device = x.device
            batch_size, h, w = x.shape[0], x.shape[2], x.shape[3]

            x = copy.deepcopy(x) #Evite de modifier les echantillons par reference (Problematique pour des utilisations paralleles)
            
            for _ in range(self._N_seqTF):
                ## Echantillonage ## == sampled_ops = np.random.choice(transforms, N)
                uniforme_dist = torch.ones(1,self._nb_tf,device=device).softmax(dim=1)
                cat_distrib= Categorical(probs=torch.ones((batch_size, self._nb_tf), device=device)*uniforme_dist)
                sample = cat_distrib.sample()

                ## Transformations ##
                x = self.apply_TF(x, sample)
        return x

    def apply_TF(self, x, sampled_TF):
        """ Applies the sampled transformations.

            Args:
                x (Tensor): Batch of data.
                sampled_TF (Tensor): Indexes of the TF to be applied to each element of data.

            Returns:
                Tensor: Batch of tranformed data.
        """
        smps_x=[]
        
        for tf_idx in range(self._nb_tf):
            mask = sampled_TF==tf_idx #Create selection mask
            smp_x = x[mask] #torch.masked_select() ? (NEcessite d'expand le mask au meme dim)

            if smp_x.shape[0]!=0: #if there's data to TF
                magnitude=self._params["mag"].detach()
                
                tf=self._TF[tf_idx]
                #print(magnitude)

                #In place
                x[mask]=self._TF_dict[tf](x=smp_x, mag=magnitude)
   
        return x

    def adjust_param(self, soft=False):
        """Not used
        """
        pass #Pas de parametre a opti

    def loss_weight(self):
        """Not used
        """
        return 1 #Pas d'echantillon = pas de ponderation

    def reg_loss(self, reg_factor=0.005):
        """Not used
        """
        return torch.tensor(0) #Pas de regularisation
    
    def train(self, mode=None):
        """ Set the module training mode.

            Args:
                mode (bool): Wether to learn the parameter of the module. None would not change mode. (default: None)
        """
        if mode is None :
            mode=self._data_augmentation
        self.augment(mode=mode) #Inutile si mode=None
        super(RandAug, self).train(mode)

    def eval(self):
        """ Set the module to evaluation mode.
        """
        self.train(mode=False)

    def augment(self, mode=True):
        """ Set the augmentation mode.

            Args:
                mode (bool): Wether to perform data augmentation on the forward pass. (default: True)
        """
        self._data_augmentation=mode

    def is_augmenting(self):
        """ Return wether data augmentation is applied.

            Returns:
                bool : True if data augmentation is applied.
        """
        return self._data_augmentation

    def __getitem__(self, key):
        """Access to the learnable parameters
        Args:
            key (string): Name of the learnable parameter to access.

        Returns:
            nn.Parameter.
        """
        return self._params[key]

    def __str__(self):
        """Name of the module

            Returns:
                String containing the name of the module as well as the higher levels parameters.
        """
        return "RandAug(%dTFx%d-Mag%d)" % (self._nb_tf, self._N_seqTF, self.mag)

### Models ###
class Higher_model(nn.Module):
    """Model wrapper for higher gradient tracking.

        Keep in memory the orginial model and it's functionnal, higher, version.

        Might not be needed anymore if Higher implement detach for fmodel.

        see : https://github.com/facebookresearch/higher

        TODO: Get rid of the original model if not needed by user.

        Attributes:
            _name (string): Name of the model.
            _mods (nn.ModuleDict): Models (Orginial and Higher version).
    """
    def __init__(self, model, model_name=None):
        """Init Higher_model.
        
            Args:
                model (nn.Module): Network for which higher gradients can be tracked.
                model_name (string): Model name. (Default: Class name of model)
        """
        super(Higher_model, self).__init__()

        self._name = model_name if model_name else model.__class__.__name__ #model.__str__()
        self._mods = nn.ModuleDict({
            'original': model,
            'functional': higher.patch.monkeypatch(model, device=None, copy_initial_weights=True)
            })

    def get_diffopt(self, opt, grad_callback=None, track_higher_grads=True):
        """Get a differentiable version of an Optimizer.

            Higher/Differentiable optimizer required to be used for higher gradient tracking.
            Usage : diffopt.step(loss) == (opt.zero_grad, loss.backward, opt.step)

            Be warry that if track_higher_grads is set to True, a new state of the model would be saved each time diffopt.step() is called.
            Thus increasing memory consumption. The detach_() method should be called to reset the gradient tape and prevent memory saturation.

            Args:
                opt (torch.optim): Optimizer to make differentiable.
                grad_callback (fct(grads)=grads): Function applied to the list of gradients parameters (ex: clipping). (default: None)
                track_higher_grads (bool): Wether higher gradient are tracked. If True, the graph/states will be retained to allow backpropagation. (default: True)

            Returns:
                (Higher.DifferentiableOptimizer): Differentiable version of the optimizer.
        """
        return higher.optim.get_diff_optim(opt, 
            self._mods['original'].parameters(),
            fmodel=self._mods['functional'],
            grad_callback=grad_callback,
            track_higher_grads=track_higher_grads)

    def forward(self, x):
        """ Main method of the model.

            Args:
                x (Tensor): Batch of data.

            Returns:
                Tensor : Output of the network. Should be logits.
        """
        return self._mods['functional'](x)

    def detach_(self):
        """Detach from the graph.

            Needed to limit the number of state kept in memory.
        """
        tmp = self._mods['functional'].fast_params
        self._mods['functional']._fast_params=[]
        self._mods['functional'].update_params(tmp)
        for p in self._mods['functional'].fast_params:
            p.detach_().requires_grad_()

    def state_dict(self):
        """Returns a dictionary containing a whole state of the module.
        """
        return self._mods['functional'].state_dict()

    def __getitem__(self, key):
        """Access to modules
        Args:
            key (string): Name of the module to access.

        Returns:
            nn.Module.
        """
        return self._mods[key]

    def __str__(self):
        """Name of the module

            Returns:
                String containing the name of the module.
        """
        return self._name

class Augmented_model(nn.Module):
    """Wrapper for a Data Augmentation module and a model.

        Attributes:
            _mods (nn.ModuleDict): A dictionary containing the modules.
            _data_augmentation (bool): Wether data augmentation should be used. 
    """
    def __init__(self, data_augmenter, model):
        """Init Augmented Model.
            
            By default, data augmentation will be performed.

            Args:
                data_augmenter (nn.Module): Data augmentation module.
                model (nn.Module): Network.
        """
        super(Augmented_model, self).__init__()

        self._mods = nn.ModuleDict({
            'data_aug': data_augmenter,
            'model': model
            })

        self.augment(mode=True)

    def forward(self, x):
        """ Main method of the Augmented model.

            Perform the forward pass of both modules.

            Args:
                x (Tensor): Batch of data.

            Returns:
                Tensor : Output of the networks. Should be logits.
        """
        return self._mods['model'](self._mods['data_aug'](x))
    
    def augment(self, mode=True):
        """ Set the augmentation mode.

            Args:
                mode (bool): Wether to perform data augmentation on the forward pass. (default: True)
        """
        self._data_augmentation=mode
        self._mods['data_aug'].augment(mode)

    #### Encapsulation Meta Opt ####
    def start_bilevel_opt(self, inner_it, hp_list, opt_param, dl_val):
        """ Set up Augmented Model for bi-level optimisation.

            Create and keep in Augmented Model the necessary objects for meta-optimisation.
            This allow for an almost transparent use by just hiding the bi-level optimisation (see ''run_dist_dataugV3'') by ::

                model.step(loss)
                
            See ''run_simple_smartaug'' for a complete example.

            Args:
                inner_it (int): Number of inner iteration before a meta-step. 0 inner iteration means there's no meta-step.
                hp_list (list): List of hyper-parameters to be learned.
                opt_param (dict): Dictionnary containing optimizers parameters.
                dl_val (DataLoader): Data loader of validation data.
        """

        self._it_count=0
        self._in_it=inner_it

        self._opt_param=opt_param
        #Inner Opt
        inner_opt = torch.optim.SGD(self._mods['model']['original'].parameters(), 
            lr=opt_param['Inner']['lr'], 
            momentum=opt_param['Inner']['momentum'], 
            weight_decay=opt_param['Inner']['decay'], 
            nesterov=opt_param['Inner']['nesterov']) #lr=1e-2 / momentum=0.9

        #Validation data
        self._dl_val=dl_val
        self._dl_val_it=iter(dl_val)
        self._val_loss=0.

        if inner_it==0 or len(hp_list)==0: #No meta-opt
            print("No meta optimization")

            #Inner Opt
            self._diffopt = self._mods['model'].get_diffopt(
                inner_opt, 
                grad_callback=(lambda grads: clip_norm(grads, max_norm=10)),
                track_higher_grads=False)

            self._meta_opt=None

        else: #Bi-level opt
            print("Bi-Level optimization")

            #Inner Opt
            self._diffopt = self._mods['model'].get_diffopt(
                inner_opt, 
                grad_callback=(lambda grads: clip_norm(grads, max_norm=10)),
                track_higher_grads=True)

            #Meta Opt
            self._meta_opt = torch.optim.Adam(hp_list, lr=opt_param['Meta']['lr'])
            self._meta_opt.zero_grad()

    def step(self, loss):
        """ Perform a model update.

            ''start_bilevel_opt'' method needs to be called once before using this method.

            Perform a step of inner optimization and, if needed, a step of meta optimization.
            Replace ::

                opt.zero_grad()
                loss.backward()
                opt.step()

                val_loss=...
                val_loss.backward()
                meta_opt.step()
                adjust_param()
                detach()
                meta_opt.zero_grad()

            By ::

                model.step(loss)

            Args:
                loss (Tensor): the training loss tensor.
        """
        self._it_count+=1
        self._diffopt.step(loss) #(opt.zero_grad, loss.backward, opt.step)
        
        if(self._meta_opt and self._it_count>0 and self._it_count%self._in_it==0): #Perform Meta step
            #print("meta")
            self._val_loss = compute_vaLoss(model=self._mods['model'], dl_it=self._dl_val_it, dl=self._dl_val) + self._mods['data_aug'].reg_loss()
            #print_graph(val_loss) #to visualize computational graph
            self._val_loss.backward()

            torch.nn.utils.clip_grad_norm_(self._mods['data_aug'].parameters(), max_norm=10, norm_type=2) #Prevent exploding grad with RNN

            self._meta_opt.step()

            #Adjust Hyper-parameters
            self._mods['data_aug'].adjust_param(soft=False) #Contrainte sum(proba)=1
            
            #For optimizer parameters, if needed
            #for param_group in self._diffopt.param_groups: 
            #    for param in list(self._opt_param['Inner'].keys())[1:]:
            #        param_group[param].data = param_group[param].data.clamp(min=1e-4)

            #Reset gradients
            self._diffopt.detach_()
            self._mods['model'].detach_()
            self._meta_opt.zero_grad()

            self._it_count=0

    def val_loss(self):
        """ Get the validation loss.

            Compute, if needed, the validation loss and returns it.

            ''start_bilevel_opt'' method needs to be called once before using this method.

            Returns:
                (Tensor) Validation loss on a single batch of data.
        """
        if(self._meta_opt): #Bilevel opti
            return self._val_loss
        else:
            return compute_vaLoss(model=self._mods['model'], dl_it=self._dl_val_it, dl=self._dl_val)

    ##########################

    def train(self, mode=True):
        """ Set the module training mode.

            Args:
                mode (bool): Wether to learn the parameter of the module. (default: None)
        """
        #if mode is None :
        #    mode=self._data_augmentation
        super(Augmented_model, self).train(mode)
        self._mods['data_aug'].augment(mode=self._data_augmentation) #Restart if needed data augmentation
        return self

    def eval(self):
        """ Set the module to evaluation mode.
        """
        #return self.train(mode=False)
        super(Augmented_model, self).train(mode=False)
        self._mods['data_aug'].augment(mode=False)
        return self

    def items(self):
        """Return an iterable of the ModuleDict key/value pairs.
        """
        return self._mods.items()

    def update(self, modules):
        """Update the module dictionnary.

            The new dictionnary should keep the same structure.
        """
        assert(self._mods.keys()==modules.keys())
        self._mods.update(modules)

    def is_augmenting(self):
        """ Return wether data augmentation is applied.

            Returns:
                bool : True if data augmentation is applied.
        """
        return self._data_augmentation

    def TF_names(self):
        """ Get the transformations names used by the data augmentation module.

            Returns:
                list : names of the transformations of the data augmentation module.
        """
        try:
            return self._mods['data_aug']._TF
        except:
            return None

    def __getitem__(self, key):
        """Access to the modules.
        Args:
            key (string): Name of the module to access.

        Returns:
            nn.Module.
        """
        return self._mods[key]

    def __str__(self):
        """Name of the module

            Returns:
                String containing the name of the module as well as the higher levels parameters.
        """
        return "Aug_mod("+str(self._mods['data_aug'])+"-"+str(self._mods['model'])+")"