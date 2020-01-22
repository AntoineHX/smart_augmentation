import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import *

#import kornia
#import random
import numpy as np
import copy

import transformations as TF

class Data_aug(nn.Module): #Rotation parametree
    def __init__(self):
        super(Data_aug, self).__init__()
        self._data_augmentation = True
        self._params = nn.ParameterDict({
            "prob": nn.Parameter(torch.tensor(0.5)),
            "mag": nn.Parameter(torch.tensor(1.0))
        })

        #self.params["mag"].register_hook(print)

    def forward(self, x):

        if self._data_augmentation and random.random() < self._params["prob"]:
            #print('Aug')
            batch_size = x.shape[0]
            # create transformation (rotation)
            alpha = self._params["mag"]*180 # in degrees
            angle = torch.ones(batch_size, device=x.device) * alpha

            # define the rotation center
            center = torch.ones(batch_size, 2, device=x.device)
            center[..., 0] = x.shape[3] / 2  # x
            center[..., 1] = x.shape[2] / 2  # y

            #print(x.shape, center)
            # define the scale factor
            scale = torch.ones(batch_size, device=x.device)

            # compute the transformation matrix
            M = kornia.get_rotation_matrix2d(center, angle, scale)

            # apply the transformation to original image
            x = kornia.warp_affine(x, M, dsize=(x.shape[2], x.shape[3])) #dsize=(h, w)

        return x

    def eval(self):
        self.augment(mode=False)
        nn.Module.eval(self)

    def augment(self, mode=True):
        self._data_augmentation=mode

    def __getitem__(self, key):
        return self._params[key]

    def __str__(self):
        return "Data_aug(Mag-1 TF)"

class Data_augV2(nn.Module): #Methode exacte
    def __init__(self):
        super(Data_augV2, self).__init__()
        self._data_augmentation = True

        self._fixed_transf=[0.0, 45.0, 180.0] #Degree rotation
        #self._fixed_transf=[0.0]
        self._nb_tf= len(self._fixed_transf)

        self._params = nn.ParameterDict({
            "prob": nn.Parameter(torch.ones(self._nb_tf)/self._nb_tf), #Distribution prob uniforme
            #"prob2": nn.Parameter(torch.ones(len(self._fixed_transf)).softmax(dim=0))
        })

        #print(self._params["prob"], self._params["prob2"])

        self.transf_idx=0

    def forward(self, x):

        if self._data_augmentation:
            #print('Aug',self._fixed_transf[self.transf_idx])
            device = x.device
            batch_size = x.shape[0]

            # create transformation (rotation)
            #alpha = 180 # in degrees
            alpha = self._fixed_transf[self.transf_idx]
            angle = torch.ones(batch_size, device=device) * alpha

            x = self.rotate(x,angle)

        return x

    def rotate(self, x, angle):

        device = x.device
        batch_size = x.shape[0]
        # define the rotation center
        center = torch.ones(batch_size, 2, device=device)
        center[..., 0] = x.shape[3] / 2  # x
        center[..., 1] = x.shape[2] / 2  # y

        #print(x.shape, center)
        # define the scale factor
        scale = torch.ones(batch_size, device=device)

        # compute the transformation matrix
        M = kornia.get_rotation_matrix2d(center, angle, scale)

        # apply the transformation to original image
        return kornia.warp_affine(x, M, dsize=(x.shape[2], x.shape[3])) #dsize=(h, w)


    def adjust_param(self): #Detach from gradient ?
        self._params['prob'].data = self._params['prob'].clamp(min=0.0,max=1.0)
        #print('proba',self._params['prob'])
        self._params['prob'].data = self._params['prob']/sum(self._params['prob']) #Contrainte sum(p)=1
        #print('Sum p', sum(self._params['prob']))

    def eval(self):
        self.augment(mode=False)
        nn.Module.eval(self)

    def augment(self, mode=True):
        self._data_augmentation=mode

    def __getitem__(self, key):
        return self._params[key]

    def __str__(self):
        return "Data_augV2(Exact-%d TF)" % self._nb_tf

class Data_augV3(nn.Module): #Echantillonage uniforme/Mixte
    def __init__(self, mix_dist=0.0):
        super(Data_augV3, self).__init__()
        self._data_augmentation = True

        #self._fixed_transf=[0.0, 45.0, 180.0] #Degree rotation
        self._fixed_transf=[0.0, 1.0, -1.0] #Flips (Identity,Horizontal,Vertical)
        #self._fixed_transf=[0.0]
        self._nb_tf= len(self._fixed_transf)

        self._params = nn.ParameterDict({
            "prob": nn.Parameter(torch.ones(self._nb_tf)/self._nb_tf), #Distribution prob uniforme
            #"prob2": nn.Parameter(torch.ones(len(self._fixed_transf)).softmax(dim=0))
        })

        #print(self._params["prob"], self._params["prob2"])
        self._sample = []

        self._mix_dist = False
        if mix_dist != 0.0:
            self._mix_dist = True
            self._mix_factor = max(min(mix_dist, 1.0), 0.0)

    def forward(self, x):

        if self._data_augmentation:
            device = x.device
            batch_size = x.shape[0]

            
            #good_distrib = Uniform(low=torch.zeros(batch_size,1, device=device),high=torch.new_full((batch_size,1),self._params["prob"], device=device))
            #bad_distrib = Uniform(low=torch.zeros(batch_size,1, device=device),high=torch.new_full((batch_size,1), 1-self._params["prob"], device=device))

            #transform_dist = Categorical(probs=torch.tensor([self._params["prob"], 1-self._params["prob"]], device=device))
            #self._sample = transform_dist._sample(sample_shape=torch.Size([batch_size,1]))  

            uniforme_dist = torch.ones(1,self._nb_tf,device=device).softmax(dim=0)

            if not self._mix_dist:
                distrib = uniforme_dist        
            else:
                distrib = (self._mix_factor*self._params["prob"]+(1-self._mix_factor)*uniforme_dist).softmax(dim=0) #Mix distrib reel / uniforme avec mix_factor

            cat_distrib= Categorical(probs=torch.ones((batch_size, self._nb_tf), device=device)*distrib)
            self._sample = cat_distrib.sample()

            TF_param = torch.tensor([self._fixed_transf[x] for x in self._sample], device=device) #Approche de marco peut-etre plus rapide

            #x = self.rotate(x,angle=TF_param)
            x = self.flip(x,flip_mat=TF_param)

        return x

    def rotate(self, x, angle):

        device = x.device
        batch_size = x.shape[0]
        # define the rotation center
        center = torch.ones(batch_size, 2, device=device)
        center[..., 0] = x.shape[3] / 2  # x
        center[..., 1] = x.shape[2] / 2  # y

        #print(x.shape, center)
        # define the scale factor
        scale = torch.ones(batch_size, device=device)

        # compute the transformation matrix
        M = kornia.get_rotation_matrix2d(center, angle, scale)

        # apply the transformation to original image
        return kornia.warp_affine(x, M, dsize=(x.shape[2], x.shape[3])) #dsize=(h, w)

    def flip(self, x, flip_mat):

        #print(flip_mat)
        device = x.device
        batch_size = x.shape[0]

        h, w = x.shape[2], x.shape[3]  # destination size
        #points_src = torch.ones(batch_size, 4, 2, device=device)
        #points_dst = torch.ones(batch_size, 4, 2, device=device)

        #Identity
        iM=torch.tensor(np.eye(3))

        #Horizontal flip
        # the source points are the region to crop corners
        #points_src = torch.FloatTensor([[
        #    [w - 1, 0], [0, 0], [0, h - 1], [w - 1, h - 1],
        #]])
        # the destination points are the image vertexes
        #points_dst = torch.FloatTensor([[
        #    [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1],
        #]])
        # compute perspective transform
        #hM = kornia.get_perspective_transform(points_src, points_dst)
        hM =torch.tensor( [[[-1.,  0., w-1],
                            [ 0.,  1.,  0.],
                            [ 0.,  0.,  1.]]])

        #Vertical flip
        # the source points are the region to crop corners
        #points_src = torch.FloatTensor([[
        #    [0, h - 1], [w - 1, h - 1], [w - 1, 0], [0, 0],
        #]])
        # the destination points are the image vertexes
        #points_dst = torch.FloatTensor([[
        #    [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1],
        #]])
        # compute perspective transform
        #vM = kornia.get_perspective_transform(points_src, points_dst)
        vM =torch.tensor( [[[ 1.,  0.,  0.],
                            [ 0., -1.,  h-1],
                            [ 0.,  0.,  1.]]])
        #print(vM)

        M=torch.ones(batch_size, 3, 3, device=device)

        for i in range(batch_size): # A optimiser
            if flip_mat[i]==0.0:
                M[i,]=iM
            elif flip_mat[i]==1.0:
                M[i,]=hM
            elif flip_mat[i]==-1.0:
                M[i,]=vM

        # warp the original image by the found transform
        return kornia.warp_perspective(x, M, dsize=(h, w))

    def adjust_param(self, soft=False): #Detach from gradient ?
        
        if soft :
            self._params['prob'].data=F.softmax(self._params['prob'].data, dim=0) #Trop 'soft', bloque en dist uniforme si lr trop faible
        else:
            #self._params['prob'].clamp(min=0.0,max=1.0)
            self._params['prob'].data = F.relu(self._params['prob'].data)
            #self._params['prob'].data = self._params['prob'].clamp(min=0.0,max=1.0)
            #print('proba',self._params['prob'])
            self._params['prob'].data = self._params['prob']/sum(self._params['prob']) #Contrainte sum(p)=1
            #print('Sum p', sum(self._params['prob']))

    def loss_weight(self):
        #w_loss = [self._params["prob"][x] for x in self._sample]
        #print(self._sample.view(-1,1).shape)
        #print(self._sample[:10])
        
        w_loss = torch.zeros((self._sample.shape[0],self._nb_tf), device=self._sample.device)
        w_loss.scatter_(1, self._sample.view(-1,1), 1)
        #print(w_loss.shape)
        #print(w_loss[:10,:])
        w_loss = w_loss * self._params["prob"]
        #print(w_loss.shape)
        #print(w_loss[:10,:])
        w_loss = torch.sum(w_loss,dim=1)
        #print(w_loss.shape)
        #print(w_loss[:10])
        return w_loss

    def train(self, mode=None):
        if mode is None :
            mode=self._data_augmentation
        self.augment(mode=mode) #Inutile si mode=None
        super(Data_augV3, self).train(mode)

    def eval(self):
        self.train(mode=False)
        #super(Augmented_model, self).eval()

    def augment(self, mode=True):
        self._data_augmentation=mode

    def __getitem__(self, key):
        return self._params[key]

    def __str__(self):
        if not self._mix_dist:
            return "Data_augV3(Uniform-%d TF)" % self._nb_tf
        else:
            return "Data_augV3(Mix %.1f-%d TF)" % (self._mix_factor, self._nb_tf)

'''
TF_dict={ #Dataugv4
  ## Geometric TF ##
  'Identity' : (lambda x, mag: x),
  'FlipUD' : (lambda x, mag: flipUD(x)),
  'FlipLR' : (lambda x, mag: flipLR(x)),
  'Rotate': (lambda x, mag: rotate(x, angle=torch.tensor([rand_int(mag, maxval=30)for _ in x], device=x.device))),
  'TranslateX': (lambda x, mag: translate(x, translation=torch.tensor([[rand_int(mag, maxval=20), 0] for _ in x], device=x.device))),
  'TranslateY': (lambda x, mag: translate(x, translation=torch.tensor([[0, rand_int(mag, maxval=20)] for _ in x], device=x.device))),
  'ShearX': (lambda x, mag: shear(x, shear=torch.tensor([[rand_float(mag, maxval=0.3), 0] for _ in x], device=x.device))),
  'ShearY': (lambda x, mag: shear(x, shear=torch.tensor([[0, rand_float(mag, maxval=0.3)] for _ in x], device=x.device))),

  ## Color TF (Expect image in the range of [0, 1]) ##
  'Contrast': (lambda x, mag: contrast(x, contrast_factor=torch.tensor([rand_float(mag, minval=0.1, maxval=1.9) for _ in x], device=x.device))),
  'Color':(lambda x, mag: color(x, color_factor=torch.tensor([rand_float(mag, minval=0.1, maxval=1.9) for _ in x], device=x.device))),
  'Brightness':(lambda x, mag: brightness(x, brightness_factor=torch.tensor([rand_float(mag, minval=0.1, maxval=1.9) for _ in x], device=x.device))),
  'Sharpness':(lambda x, mag: sharpeness(x, sharpness_factor=torch.tensor([rand_float(mag, minval=0.1, maxval=1.9) for _ in x], device=x.device))),
  'Posterize': (lambda x, mag: posterize(x, bits=torch.tensor([rand_int(mag, minval=4, maxval=8) for _ in x], device=x.device))),
  'Solarize': (lambda x, mag: solarize(x, thresholds=torch.tensor([rand_int(mag,minval=1, maxval=256)/256. for _ in x], device=x.device))) , #=>Image entre [0,1] #Pas opti pour des batch

  #Non fonctionnel
  #'Auto_Contrast': (lambda mag: None), #Pas opti pour des batch (Super lent)
  #'Equalize': (lambda mag: None),
}
'''
class Data_augV4(nn.Module): #Transformations avec mask
    def __init__(self, TF_dict=TF.TF_dict, N_TF=1, mix_dist=0.0):
        super(Data_augV4, self).__init__()
        assert len(TF_dict)>0
        
        self._data_augmentation = True

        #self._TF_matrix={}
        #self._input_info={'h':0, 'w':0, 'device':None} #Input associe a TF_matrix
        #self._mag_fct = TF_dict
        self._TF_dict = TF_dict
        self._TF= list(self._TF_dict.keys())
        self._nb_tf= len(self._TF)

        self._N_seqTF = N_TF

        self._fixed_mag=5 #[0, PARAMETER_MAX]
        self._params = nn.ParameterDict({
            "prob": nn.Parameter(torch.ones(self._nb_tf)/self._nb_tf), #Distribution prob uniforme
        })

        self._samples = []

        self._mix_dist = False
        if mix_dist != 0.0:
            self._mix_dist = True
            self._mix_factor = max(min(mix_dist, 1.0), 0.0)

    def forward(self, x):
        if self._data_augmentation:
            device = x.device
            batch_size, h, w = x.shape[0], x.shape[2], x.shape[3]

            x = copy.deepcopy(x) #Evite de modifier les echantillons par reference (Problematique pour des utilisations paralleles)
            self._samples = []
            
            for _ in range(self._N_seqTF):
                ## Echantillonage ##
                uniforme_dist = torch.ones(1,self._nb_tf,device=device).softmax(dim=1)

                if not self._mix_dist:
                    self._distrib = uniforme_dist        
                else:
                    self._distrib = (self._mix_factor*self._params["prob"]+(1-self._mix_factor)*uniforme_dist).softmax(dim=1) #Mix distrib reel / uniforme avec mix_factor

                cat_distrib= Categorical(probs=torch.ones((batch_size, self._nb_tf), device=device)*self._distrib)
                sample = cat_distrib.sample()
                self._samples.append(sample)

                ## Transformations ##
                x = self.apply_TF(x, sample)
        return x
    '''
    def compute_TF_matrix(self, magnitude=None, sample_info= None):
        print('Computing TF_matrix...')
        if not magnitude :
            magnitude=self._fixed_mag

        if sample_info:
            self._input_info['h']= sample_info['h']
            self._input_info['w']= sample_info['w']
            self._input_info['device'] = sample_info['device']
        h, w, device= self._input_info['h'], self._input_info['w'], self._input_info['device']

        self._TF_matrix={}
        for tf in self._TF :
            if tf=='Id':
                self._TF_matrix[tf]=torch.tensor([[[ 1.,  0.,  0.],
                                                   [ 0.,  1.,  0.],
                                                   [ 0.,  0.,  1.]]], device=device)
            elif tf=='Rot':
                center = torch.ones(1, 2, device=device)
                center[0, 0] = w / 2  # x
                center[0, 1] = h / 2  # y
                scale = torch.ones(1, device=device)
                angle = self._mag_fct[tf](magnitude) * torch.ones(1, device=device)
                R = kornia.get_rotation_matrix2d(center, angle, scale) #Rotation matrix (1,2,3)
                self._TF_matrix[tf]=torch.cat((R,torch.tensor([[[ 0.,  0.,  1.]]], device=device)), dim=1) #TF matrix (1,3,3)
            elif tf=='FlipLR':
                self._TF_matrix[tf]=torch.tensor([[[-1.,  0., w-1],
                                                   [ 0.,  1.,  0.],
                                                   [ 0.,  0.,  1.]]], device=device)
            elif tf=='FlipUD':
                self._TF_matrix[tf]=torch.tensor([[[ 1.,  0.,  0.],
                                                   [ 0., -1.,  h-1],
                                                   [ 0.,  0.,  1.]]], device=device)
            else:
                raise Exception("Invalid TF requested")
    '''
    def apply_TF(self, x, sampled_TF):
        device = x.device
        smps_x=[]
        masks=[]
        for tf_idx in range(self._nb_tf):
            mask = sampled_TF==tf_idx #Create selection mask
            smp_x = x[mask] #torch.masked_select() ?

            if smp_x.shape[0]!=0: #if there's data to TF
                magnitude=self._fixed_mag
                tf=self._TF[tf_idx]

                '''
                ## Geometric TF ##
                if tf=='Identity':
                    pass
                elif tf=='FlipLR':
                    smp_x = TF.flipLR(smp_x)
                elif tf=='FlipUD':
                    smp_x = TF.flipUD(smp_x)
                elif tf=='Rotate':
                    smp_x = TF.rotate(smp_x, angle=torch.tensor([self._mag_fct[tf](magnitude) for _ in smp_x], device=device))
                elif tf=='TranslateX' or tf=='TranslateY':
                    smp_x = TF.translate(smp_x, translation=torch.tensor([self._mag_fct[tf](magnitude) for _ in smp_x], device=device))
                elif tf=='ShearX' or tf=='ShearY' :
                    smp_x = TF.shear(smp_x, shear=torch.tensor([self._mag_fct[tf](magnitude) for _ in smp_x], device=device))

                ## Color TF (Expect image in the range of [0, 1]) ##
                elif tf=='Contrast':
                    smp_x = TF.contrast(smp_x, contrast_factor=torch.tensor([self._mag_fct[tf](magnitude) for _ in smp_x], device=device))
                elif tf=='Color':
                    smp_x = TF.color(smp_x, color_factor=torch.tensor([self._mag_fct[tf](magnitude) for _ in smp_x], device=device))
                elif tf=='Brightness':
                    smp_x = TF.brightness(smp_x, brightness_factor=torch.tensor([self._mag_fct[tf](magnitude) for _ in smp_x], device=device))
                elif tf=='Sharpness':
                    smp_x = TF.sharpeness(smp_x, sharpness_factor=torch.tensor([self._mag_fct[tf](magnitude) for _ in smp_x], device=device))
                elif tf=='Posterize':
                    smp_x = TF.posterize(smp_x, bits=torch.tensor([1 for _ in smp_x], device=device))
                elif tf=='Solarize':
                    smp_x = TF.solarize(smp_x, thresholds=torch.tensor([self._mag_fct[tf](magnitude) for _ in smp_x], device=device)) 
                elif tf=='Equalize':
                    smp_x = TF.equalize(smp_x)
                elif tf=='Auto_Contrast':
                    smp_x = TF.auto_contrast(smp_x)
                else:
                    raise Exception("Invalid TF requested : ", tf) 

                x[mask]=smp_x # Refusionner eviter x[mask] : in place
                '''
                x[mask]=self._TF_dict[tf](x=smp_x, mag=magnitude) # Refusionner eviter x[mask] : in place
            
            #idx= mask.nonzero()
            #print('-'*8)
            #print(idx[0], tf_idx)
            #print(smp_x[0,])
            #x=x.view(-1,3*32*32)
            #x=x.scatter(dim=0, index=idx, src=smp_x.view(-1,3*32*32)) #Changement des Tensor mais pas visible sur la visualisation...
            #x=x.view(-1,3,32,32)
            #print(x[0,])
            
        '''
        if len(self._TF_matrix)==0 or self._input_info['h']!=h or self._input_info['w']!=w or self._input_info['device']!=device: #Device different:Pas necessaire de tout recalculer
            self.compute_TF_matrix(sample_info={'h': x.shape[2],
                                                'w': x.shape[3], 
                                                'device': x.device})

        TF_matrix = torch.zeros(batch_size, 3, 3, device=device) #All geom TF 

        for tf_idx in range(self._nb_tf):
            mask = self._sample==tf_idx #Create selection mask
            TF_matrix[mask,]=self._TF_matrix[self._TF[tf_idx]]

        x=kornia.warp_perspective(x, TF_matrix, dsize=(h, w))
        '''
        return x

    def adjust_param(self, soft=False): #Detach from gradient ?
        
        if soft :
            self._params['prob'].data=F.softmax(self._params['prob'].data, dim=0) #Trop 'soft', bloque en dist uniforme si lr trop faible
        else:
            #self._params['prob'].clamp(min=0.0,max=1.0)
            self._params['prob'].data = F.relu(self._params['prob'].data)
            #self._params['prob'].data = self._params['prob'].clamp(min=0.0,max=1.0)

            self._params['prob'].data = self._params['prob']/sum(self._params['prob']) #Contrainte sum(p)=1

    def loss_weight(self):
        # 1 seule TF
        #self._sample = self._samples[-1]
        #w_loss = torch.zeros((self._sample.shape[0],self._nb_tf), device=self._sample.device)
        #w_loss.scatter_(dim=1, index=self._sample.view(-1,1), value=1)
        #w_loss = w_loss * self._params["prob"]/self._distrib #Ponderation par les proba (divisee par la distrib pour pas diminuer la loss)
        #w_loss = torch.sum(w_loss,dim=1)
        
        #Plusieurs TF sequentielles
        w_loss = torch.zeros((self._samples[0].shape[0],self._nb_tf), device=self._samples[0].device)
        for sample in self._samples:
            tmp_w = torch.zeros(w_loss.size(),device=w_loss.device)
            tmp_w.scatter_(dim=1, index=sample.view(-1,1), value=1/self._N_seqTF)
            w_loss += tmp_w

        w_loss = w_loss * self._params["prob"]/self._distrib #Ponderation par les proba (divisee par la distrib pour pas diminuer la loss)
        w_loss = torch.sum(w_loss,dim=1)
        return w_loss


    def train(self, mode=None):
        if mode is None :
            mode=self._data_augmentation
        self.augment(mode=mode) #Inutile si mode=None
        super(Data_augV4, self).train(mode)

    def eval(self):
        self.train(mode=False)

    def augment(self, mode=True):
        self._data_augmentation=mode

    def __getitem__(self, key):
        return self._params[key]

    def __str__(self):
        if not self._mix_dist:
            return "Data_augV4(Uniform-%d TF x %d)" % (self._nb_tf, self._N_seqTF)
        else:
            return "Data_augV4(Mix %.1f-%d TF x %d)" % (self._mix_factor, self._nb_tf, self._N_seqTF)


class Data_augV6(nn.Module): #Optimisation sequentielle #Mauvais resultats
    def __init__(self, TF_dict=TF.TF_dict, N_TF=1, mix_dist=0.0, fixed_prob=False, prob_set_size=None, fixed_mag=True, shared_mag=True):
        super(Data_augV6, self).__init__()
        assert len(TF_dict)>0
        
        self._data_augmentation = True

        self._TF_dict = TF_dict
        self._TF= list(self._TF_dict.keys())
        self._nb_tf= len(self._TF)

        self._N_seqTF = N_TF
        self._shared_mag = shared_mag
        self._fixed_mag = fixed_mag
        
        self._TF_set_size = prob_set_size if prob_set_size else self._nb_tf
        
        self._fixed_TF=[0] #Identite
        assert self._TF_set_size>=len(self._fixed_TF)

        if self._TF_set_size>self._nb_tf:
            print("Warning : TF sets size higher than number of TF. Reducing set size to %d"%self._nb_tf)
            self._TF_set_size=self._nb_tf
        
        ## Genenerate TF sets ##
        if self._TF_set_size==len(self._fixed_TF): 
            print("Warning : using only fixed set of TF : ", self._fixed_TF)
            self._TF_sets=torch.tensor([self._fixed_TF])
        else: 
            def generate_TF_sets(n_TF, set_size, idx_prefix=[]):
                TF_sets=[]
                if len(idx_prefix)!=0:
                    if set_size>2:
                        for i in range(idx_prefix[-1]+1, n_TF):
                            TF_sets += generate_TF_sets(n_TF=n_TF, set_size=set_size-1, idx_prefix=idx_prefix+[i])
                    else:
                        #if i not in idx_prefix:
                        TF_sets+=[torch.tensor(idx_prefix+[i]) for i in range(idx_prefix[-1]+1, n_TF)]
                elif set_size>1:
                     for i in range(0, n_TF):
                        TF_sets += generate_TF_sets(n_TF=n_TF, set_size=set_size, idx_prefix=[i])
                else:
                    TF_sets+=[torch.tensor([i]) for i in range(0, n_TF)]
                return TF_sets

            self._TF_sets=generate_TF_sets(self._nb_tf, self._TF_set_size, self._fixed_TF)

        ## Plan TF learning schedule ##
        self._TF_schedule = [list(range(len(self._TF_sets))) for _ in range(self._N_seqTF)]
        for n_tf in range(self._N_seqTF) :
            TF.random.shuffle(self._TF_schedule[n_tf])

        self._current_TF_idx=0 #random.randint
        self._start_prob = 1/self._TF_set_size


        self._params = nn.ParameterDict({
            "prob": nn.Parameter(torch.tensor(self._start_prob).expand(self._nb_tf)), #Proba independantes
            "mag" : nn.Parameter(torch.tensor(float(TF.PARAMETER_MAX)) if self._shared_mag
                            else torch.tensor(float(TF.PARAMETER_MAX)).expand(self._nb_tf)), #[0, PARAMETER_MAX]
        })

        #for t in TF.TF_no_mag: self._params['mag'][self._TF.index(t)].data-=self._params['mag'][self._TF.index(t)].data #Mag inutile pour les TF ignore_mag

        #Distribution
        self._fixed_prob=fixed_prob
        self._samples = []
        self._mix_dist = False
        if mix_dist != 0.0:
            self._mix_dist = True
            self._mix_factor = max(min(mix_dist, 0.999), 0.0)

        #Mag regularisation
        if not self._fixed_mag:
            if  self._shared_mag :
                self._reg_tgt = torch.tensor(TF.PARAMETER_MAX, dtype=torch.float) #Encourage amplitude max
            else:
                self._reg_mask=[self._TF.index(t) for t in self._TF if t not in TF.TF_ignore_mag]
                self._reg_tgt=torch.full(size=(len(self._reg_mask),), fill_value=TF.PARAMETER_MAX) #Encourage amplitude max

    def forward(self, x):
        self._samples = []
        if self._data_augmentation:# and TF.random.random() < 0.5:
            device = x.device
            batch_size, h, w = x.shape[0], x.shape[2], x.shape[3]

            x = copy.deepcopy(x) #Evite de modifier les echantillons par reference (Problematique pour des utilisations paralleles)
            
            for n_tf in range(self._N_seqTF):

                tf_set = self._TF_sets[self._TF_schedule[n_tf][self._current_TF_idx]].to(device)
                #print(n_tf, tf_set)
                ## Echantillonage ##
                uniforme_dist = torch.ones(1,len(tf_set),device=device).softmax(dim=1)

                if not self._mix_dist:
                    self._distrib = uniforme_dist        
                else:
                    prob = self._params["prob"].detach() if self._fixed_prob else self._params["prob"]
                    curr_prob = torch.index_select(prob, 0, tf_set)
                    curr_prob = curr_prob /sum(curr_prob) #Contrainte sum(p)=1
                    self._distrib = (self._mix_factor*curr_prob+(1-self._mix_factor)*uniforme_dist)#.softmax(dim=1) #Mix distrib reel / uniforme avec mix_factor

                cat_distrib= Categorical(probs=torch.ones((batch_size, len(tf_set)), device=device)*self._distrib)
                sample = cat_distrib.sample()
                self._samples.append(sample)

                ## Transformations ##
                x = self.apply_TF(x, sample)
        return x

    def apply_TF(self, x, sampled_TF):
        device = x.device
        batch_size, channels, h, w = x.shape
        smps_x=[]
        
        for sel_idx, tf_idx in enumerate(self._TF_sets[self._current_TF_idx]):
            mask = sampled_TF==sel_idx #Create selection mask
            smp_x = x[mask] #torch.masked_select() ? (NEcessite d'expand le mask au meme dim)

            if smp_x.shape[0]!=0: #if there's data to TF
                magnitude=self._params["mag"] if self._shared_mag else self._params["mag"][tf_idx]
                if self._fixed_mag: magnitude=magnitude.detach() #Fmodel tente systematiquement de tracker les gradient de tout les param

                tf=self._TF[tf_idx]
                #print(magnitude)

                #In place
                #x[mask]=self._TF_dict[tf](x=smp_x, mag=magnitude)

                #Out of place
                smp_x = self._TF_dict[tf](x=smp_x, mag=magnitude)
                idx= mask.nonzero()
                idx= idx.expand(-1,channels).unsqueeze(dim=2).expand(-1,channels, h).unsqueeze(dim=3).expand(-1,channels, h, w) #Il y a forcement plus simple ...
                x=x.scatter(dim=0, index=idx, src=smp_x)
                                
        return x

    def adjust_param(self, soft=False): #Detach from gradient ?
        if not self._fixed_prob:
            if soft :
                self._params['prob'].data=F.softmax(self._params['prob'].data, dim=0) #Trop 'soft', bloque en dist uniforme si lr trop faible
            else:
                self._params['prob'].data = F.relu(self._params['prob'].data)
                #self._params['prob'].data = self._params['prob'].clamp(min=0.0,max=1.0)
                #self._params['prob'].data = self._params['prob']/sum(self._params['prob']) #Contrainte sum(p)=1

                self._params['prob'].data[0]=self._start_prob #Fixe p identite

        if not self._fixed_mag:
            #self._params['mag'].data = self._params['mag'].data.clamp(min=0.0,max=TF.PARAMETER_MAX) #Bloque une fois au extreme
            self._params['mag'].data = F.relu(self._params['mag'].data) - F.relu(self._params['mag'].data - TF.PARAMETER_MAX)

    def loss_weight(self): #A verifier
        if len(self._samples)==0 : return 1 #Pas d'echantillon = pas de ponderation

        prob = self._params["prob"].detach() if self._fixed_prob else self._params["prob"]

        #Plusieurs TF sequentielles (Attention ne prend pas en compte ordre !)
        w_loss = torch.zeros((self._samples[0].shape[0],self._TF_set_size), device=self._samples[0].device)
        for n_tf in range(self._N_seqTF):
            tmp_w = torch.zeros(w_loss.size(),device=w_loss.device)
            tmp_w.scatter_(dim=1, index=self._samples[n_tf].view(-1,1), value=1/self._N_seqTF)

            tf_set = self._TF_sets[self._TF_schedule[n_tf][self._current_TF_idx]].to(prob.device)
            curr_prob = torch.index_select(prob, 0, tf_set)
            curr_prob = curr_prob /sum(curr_prob) #Contrainte sum(p)=1

            #ATTENTION DISTRIB DIFFERENTE AVEC MIX
            assert not self._mix_dist
            w_loss += tmp_w * curr_prob /self._distrib #Ponderation par les proba (divisee par la distrib pour pas diminuer la loss)

        w_loss = torch.sum(w_loss,dim=1)
        return w_loss

    def reg_loss(self, reg_factor=0.005):
        if self._fixed_mag: # or self._fixed_prob: #Pas de regularisation si trop peu de DOF
            return torch.tensor(0)
        else:
            #return reg_factor * F.l1_loss(self._params['mag'][self._reg_mask], target=self._reg_tgt, reduction='mean') 
            params = self._params['mag'] if self._params['mag'].shape==torch.Size([]) else self._params['mag'][self._reg_mask]
            return reg_factor * F.mse_loss(params, target=self._reg_tgt.to(params.device), reduction='mean')

    def next_TF_set(self, idx=None):
        if idx:
            self._current_TF_idx=idx
        else:
            self._current_TF_idx+=1

        if self._current_TF_idx>=len(self._TF_schedule[0]): 
            self._current_TF_idx=0
            for n_tf in range(self._N_seqTF) :
                TF.random.shuffle(self._TF_schedule[n_tf])
            #print('-- New schedule --')

    def train(self, mode=None):
        if mode is None :
            mode=self._data_augmentation
        self.augment(mode=mode) #Inutile si mode=None
        super(Data_augV6, self).train(mode)

    def eval(self):
        self.train(mode=False)

    def augment(self, mode=True):
        self._data_augmentation=mode

    def __getitem__(self, key):
        return self._params[key]

    def __str__(self):
        dist_param=''
        if self._fixed_prob: dist_param+='Fx'
        mag_param='Mag'
        if self._fixed_mag: mag_param+= 'Fx'
        if self._shared_mag: mag_param+= 'Sh'
        if not self._mix_dist:
            return "Data_augV6(Uniform%s-%dTF(%d)x%d-%s)" % (dist_param, self._nb_tf, self._TF_set_size, self._N_seqTF, mag_param)
        else:
            return "Data_augV6(Mix%.1f%s-%dTF(%d)x%d-%s)" % (self._mix_factor,dist_param, self._nb_tf, self._TF_set_size, self._N_seqTF, mag_param)

class RandAugUDA(nn.Module): #RandAugment from UDA (for DA during training)
    def __init__(self, TF_dict=TF.TF_dict, N_TF=1, mag=TF.PARAMETER_MAX):
        super(RandAugUDA, self).__init__()

        self._data_augmentation = True

        self._TF_dict = TF_dict
        self._TF= list(self._TF_dict.keys())
        self._nb_tf= len(self._TF)
        self._N_seqTF = N_TF

        self.mag=nn.Parameter(torch.tensor(float(mag)))
        self._params = nn.ParameterDict({
            "prob": nn.Parameter(torch.tensor(0.5).unsqueeze(dim=0)),
            "mag" : nn.Parameter(torch.tensor(float(TF.PARAMETER_MAX))),
        })
        self._shared_mag = True
        self._fixed_mag = True

        self._op_list =[]
        for tf in self._TF:
            for mag in range(1, int(self._params['mag']*10), 1):
                self._op_list+=[(tf, self._params['prob'].item(), mag/10)]
        self._nb_op = len(self._op_list)

    def forward(self, x):
        if self._data_augmentation:# and TF.random.random() < 0.5:
            device = x.device
            batch_size, h, w = x.shape[0], x.shape[2], x.shape[3]

            x = copy.deepcopy(x) #Evite de modifier les echantillons par reference (Problematique pour des utilisations paralleles)
            
            for _ in range(self._N_seqTF):
                ## Echantillonage ## == sampled_ops = np.random.choice(transforms, N)
                uniforme_dist = torch.ones(1, self._nb_op, device=device).softmax(dim=1)
                cat_distrib= Categorical(probs=torch.ones((batch_size, self._nb_op), device=device)*uniforme_dist)
                sample = cat_distrib.sample()

                ## Transformations ##
                x = self.apply_TF(x, sample)
        return x

    def apply_TF(self, x, sampled_TF):
        smps_x=[]
        
        for op_idx in range(self._nb_op):
            mask = sampled_TF==op_idx #Create selection mask
            smp_x = x[mask] #torch.masked_select() ? (Necessite d'expand le mask au meme dim)

            if smp_x.shape[0]!=0: #if there's data to TF
                if TF.random.random() < self._op_list[op_idx][1]:
                    magnitude=self._op_list[op_idx][2]
                    tf=self._op_list[op_idx][0]

                    #In place
                    x[mask]=self._TF_dict[tf](x=smp_x, mag=torch.tensor(magnitude, device=x.device))
   
        return x

    def adjust_param(self, soft=False):
        pass #Pas de parametre a opti

    def loss_weight(self):
        return 1 #Pas d'echantillon = pas de ponderation

    def reg_loss(self, reg_factor=0.005):
        return torch.tensor(0) #Pas de regularisation
    
    def train(self, mode=None):
        if mode is None :
            mode=self._data_augmentation
        self.augment(mode=mode) #Inutile si mode=None
        super(RandAugUDA, self).train(mode)

    def eval(self):
        self.train(mode=False)

    def augment(self, mode=True):
        self._data_augmentation=mode

    def __getitem__(self, key):
        return self._params[key]

    def __str__(self):
        return "RandAugUDA(%dTFx%d-Mag%d)" % (self._nb_tf, self._N_seqTF, self.mag)

'''
import higher
class Augmented_model2(nn.Module):
    def __init__(self, data_augmenter, model):
        super(Augmented_model2, self).__init__()

        self._mods = nn.ModuleDict({
            'data_aug': data_augmenter,
            'model': model,
            'fmodel': None
            })

        self.augment(mode=True)

    def initialize(self):
        self._mods['model'].initialize()

    def forward(self, x):
        if self._mods['fmodel']:
            return self._mods['fmodel'](self._mods['data_aug'](x))
        else:
            return self._mods['model'](self._mods['data_aug'](x))

    def functional(self, opt, track_higher_grads=True):
        self._mods['fmodel'] = higher.patch.monkeypatch(self._mods['model'], device=None, copy_initial_weights=True)

        return higher.optim.get_diff_optim(opt, 
            self._mods['model'].parameters(),
            fmodel=self._mods['fmodel'],
            track_higher_grads=track_higher_grads)

    def detach_(self):
        tmp = self._mods['fmodel'].fast_params
        self._mods['fmodel']._fast_params=[]
        self._mods['fmodel'].update_params(tmp)
        for p in self._mods['fmodel'].fast_params:
            p.detach_().requires_grad_()
    
    def augment(self, mode=True):
        self._data_augmentation=mode
        self._mods['data_aug'].augment(mode)

    def train(self, mode=None):
        if mode is None :
            mode=self._data_augmentation
        self._mods['data_aug'].augment(mode)
        super(Augmented_model2, self).train(mode)
        return self

    def eval(self):
        return self.train(mode=False)
        #super(Augmented_model, self).eval()

    def items(self):
        """Return an iterable of the ModuleDict key/value pairs.
        """
        return self._mods.items()

    def update(self, modules):
        self._mods.update(modules)

    def is_augmenting(self):
        return self._data_augmentation

    def TF_names(self):
        try:
            return self._mods['data_aug']._TF
        except:
            return None

    def __getitem__(self, key):
        return self._mods[key]

    def __str__(self):
        return "Aug_mod("+str(self._mods['data_aug'])+"-"+str(self._mods['model'])+")"
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
            ## Geometric TF ##
            'Rotate',
            'TranslateX',
            'TranslateY',
            'ShearX',
            'ShearY',

            'Cutout',

            ## Color TF ##
            'Contrast',
            'Color',
            'Brightness',
            'Sharpness',
            #'Posterize',
            #'Solarize',

            'Invert',
            'AutoContrast',
            'Equalize',
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
            #if idx==10000:break

            for _ in range(aug_copy):
                chosen_policy = policies[np.random.choice(len(policies))]
                aug_image = augmentation_transforms.apply_policy(chosen_policy, image, use_mean_std=False) #Cast en float image
                #aug_image = augmentation_transforms.cutout_numpy(aug_image)

                self.unsup_data+=[(aug_image*255.).astype(self.sup_data.dtype)]#Cast float image to uint8
                self.unsup_targets+=[self.sup_targets[idx]]

        #self.unsup_data=(np.array(self.unsup_data)*255.).astype(self.sup_data.dtype) #Cast float image to uint8
        self.unsup_data=np.array(self.unsup_data)
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
        return "CIFAR10(Sup:{}-Unsup:{}-{}TF)".format(self.dataset_info['sup'], self.dataset_info['unsup'], len(self._TF))
