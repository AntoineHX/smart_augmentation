#from hyperopt import *
from hyperopt_v2 import *

import torchvision.transforms.functional as TF
import torchvision.transforms as T

#from scipy import ndimage
import kornia

import random


class LeNet_v3(nn.Module):
    def __init__(self, num_inp, num_out):
        super(LeNet_v3, self).__init__()
        self.params = nn.ParameterDict({
            'w1': nn.Parameter(torch.zeros(20, num_inp, 5, 5)),
            'b1': nn.Parameter(torch.zeros(20)),
            'w2': nn.Parameter(torch.zeros(50, 20, 5, 5)),
            'b2': nn.Parameter(torch.zeros(50)),
            'w3': nn.Parameter(torch.zeros(500,4*4*50)),
            'b3': nn.Parameter(torch.zeros(500)),
            'w4': nn.Parameter(torch.zeros(10, 500)),
            'b4': nn.Parameter(torch.zeros(10))
        })


    def initialize(self):
        nn.init.kaiming_uniform_(self.params["w1"], a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.params["w2"], a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.params["w3"], a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.params["w4"], a=math.sqrt(5))

    def forward(self, x):
        #print("Start Shape ", x.shape)
        out = F.relu(F.conv2d(input=x, weight=self.params["w1"], bias=self.params["b1"]))
        #print("Shape ", out.shape)
        out = F.max_pool2d(out, 2)
        #print("Shape ", out.shape)
        out = F.relu(F.conv2d(input=out, weight=self.params["w2"], bias=self.params["b2"]))
        #print("Shape ", out.shape)
        out = F.max_pool2d(out, 2)
        #print("Shape ", out.shape)
        out = out.view(out.size(0), -1)
        #print("Shape ", out.shape)
        out = F.relu(F.linear(out, self.params["w3"], self.params["b3"]))
        #print("Shape ", out.shape)
        out = F.linear(out, self.params["w4"], self.params["b4"])
        #print("Shape ", out.shape)
        return F.log_softmax(out, dim=1)


    def print_grad_fn(self):
        for n, p in self.params.items():
            print(n, p.grad_fn)

    def __str__(self):
        return "mnist_CNN_augmented / "

class Data_aug(nn.Module):
    def __init__(self):
        super(Data_aug, self).__init__()
        self.data_augmentation = True
        self.params = nn.ParameterDict({
            "prob": nn.Parameter(torch.tensor(0.5)),
            "mag": nn.Parameter(torch.tensor(180.0))
        })

        #self.params["mag"].register_hook(print)

    def forward(self, x):

        if self.data_augmentation and self.training and random.random() < self.params["prob"]:
            #print('Aug')
            batch_size = x.shape[0]
            # create transformation (rotation)
            alpha = self.params["mag"] # in degrees
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
        self.params['prob']=torch.tensor(0.0, device=self.device)
        nn.Module.eval(self)

    def data_augmentation(self, mode=True):
        self.data_augmentation=mode

    def print_grad_fn(self):
        for n, p in self.params.items():
            print(n, p.grad_fn)

    def __str__(self):
        return "Data_Augmenter / "

class Augmented_model(nn.Module):
    def __init__(self, model, data_augmenter):
        #self.model = model
        #self.data_aug = data_augmenter
        super(Augmented_model, self).__init__()#nn.Module.__init__(self)
        #super().__init__()
        self.mods = nn.ModuleDict({
            'data_aug': data_augmenter,
            'model': model
            })
        #for name, param in self.mods.named_parameters():
        #    print(name, type(param.data), param.size())

        #params = self.mods.named_parameters() #self.parameters()
        #parameters = [param for param in self.model.parameters()] + [param for param in self.data_aug.parameters()] 
        #Optimizable.__init__(self, params, optimizer)

    def initialize(self):
        self.mods['model'].initialize()

    def forward(self, x):
        return self.mods['model'](self.mods['data_aug'](x))

    #def adjust(self):
    #    self.optimizer.adjust(self) #Parametres des dict

    def data_augmentation(self, mode=True):
        self.mods['data_aug'].data_augmentation=mode

    def begin(self):
        for param in self.parameters():
            param.requires_grad_()  # keep gradient information…
            param.retain_grad()  # even if not a leaf…

    def print_grad_fn(self):
        for n, m in self.mods.items():
            m.print_grad_fn()

    def __str__(self):
        return str(self.mods['data_aug'])+ str(self.mods['model'])# + str(self.optimizer)