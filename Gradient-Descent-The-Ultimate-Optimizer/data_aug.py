from hyperopt import *
#from hyperopt_v2 import *

import torchvision.transforms.functional as TF
import torchvision.transforms as T

#from scipy import ndimage
import kornia

import random


class MNIST_FullyConnected_Augmented(Optimizable):
    """
    A fully-connected NN for the MNIST task. This is Optimizable but not itself
    an optimizer.
    """

    def __init__(self, num_inp, num_hid, num_out, optimizer, device = torch.device('cuda')):
        self.device = device
        #print(self.device)
        parameters = {
            "w1": torch.zeros(num_inp, num_hid, device=self.device).t(),
            "b1": torch.zeros(num_hid, device=self.device).t(),
            "w2": torch.zeros(num_hid, num_out, device=self.device).t(),
            "b2": torch.zeros(num_out, device=self.device).t(),

            #Data augmentation
            "prob": torch.tensor(0.5, device=self.device),
            "mag": torch.tensor(180.0, device=self.device),
        }
        super().__init__(parameters, optimizer)

    def initialize(self):
        nn.init.kaiming_uniform_(self.parameters["w1"], a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.parameters["w2"], a=math.sqrt(5))
        self.optimizer.initialize()
        #print(self.device)

    def forward(self, x):
        """Compute a prediction."""
        #print("Prob:",self.parameters["prob"].item())
        if random.random() < self.parameters["prob"]:
            #angle = 45
            #x = TF.rotate(x, angle)
            #print(self.device)
            #x = F.linear(x, torch.ones(28*28, 28*28, device=self.device).t()*self.parameters["mag"], bias=None)
            x = x + self.parameters["mag"]

        x = F.linear(x, self.parameters["w1"], self.parameters["b1"])
        x = torch.tanh(x)
        x = F.linear(x, self.parameters["w2"], self.parameters["b2"])
        x = torch.tanh(x)
        x = F.log_softmax(x, dim=1)
        return x

    def adjust(self):
        self.optimizer.adjust(self.parameters)

    def __str__(self):
        return "mnist_FC_augmented / " + str(self.optimizer)

class LeNet(Optimizable, nn.Module):
    def __init__(self, num_inp, num_out, optimizer, device = torch.device('cuda')):
        nn.Module.__init__(self)
        self.device = device
        parameters = {
            "w1": torch.zeros(20, num_inp, 5, 5, device=self.device),
            "b1": torch.zeros(20, device=self.device),
            "w2": torch.zeros(50, 20, 5, 5, device=self.device),
            "b2": torch.zeros(50, device=self.device),
            "w3": torch.zeros(500,4*4*50, device=self.device),
            "b3": torch.zeros(500, device=self.device),
            "w4": torch.zeros(10, 500, device=self.device),
            "b4": torch.zeros(10, device=self.device),

            #Data augmentation
            "prob": torch.tensor(1.0, device=self.device),
            "mag": torch.tensor(180.0, device=self.device),
        }
        super().__init__(parameters, optimizer)

    def initialize(self):
        nn.init.kaiming_uniform_(self.parameters["w1"], a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.parameters["w2"], a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.parameters["w3"], a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.parameters["w4"], a=math.sqrt(5))
        self.optimizer.initialize()

    def forward(self, x):

        if random.random() < self.parameters["prob"]:
            
            batch_size = x.shape[0]
            # create transformation (rotation)
            alpha = self.parameters["mag"] # in degrees
            angle = torch.ones(batch_size, device=self.device) * alpha

            # define the rotation center
            center = torch.ones(batch_size, 2, device=self.device)
            center[..., 0] = x.shape[3] / 2  # x
            center[..., 1] = x.shape[2] / 2  # y

            #print(x.shape, center)
            # define the scale factor
            scale = torch.ones(batch_size, device=self.device)

            # compute the transformation matrix
            M = kornia.get_rotation_matrix2d(center, angle, scale)

            # apply the transformation to original image
            x = kornia.warp_affine(x, M, dsize=(x.shape[2], x.shape[3])) #dsize=(h, w)

        #print("Start Shape ", x.shape)
        out = F.relu(F.conv2d(input=x, weight=self.parameters["w1"], bias=self.parameters["b1"]))
        #print("Shape ", out.shape)
        out = F.max_pool2d(out, 2)
        #print("Shape ", out.shape)
        out = F.relu(F.conv2d(input=out, weight=self.parameters["w2"], bias=self.parameters["b2"]))
        #print("Shape ", out.shape)
        out = F.max_pool2d(out, 2)
        #print("Shape ", out.shape)
        out = out.view(out.size(0), -1)
        #print("Shape ", out.shape)
        out = F.relu(F.linear(out, self.parameters["w3"], self.parameters["b3"]))
        #print("Shape ", out.shape)
        out = F.linear(out, self.parameters["w4"], self.parameters["b4"])
        #print("Shape ", out.shape)
        return F.log_softmax(out, dim=1)

    def adjust(self):
        self.optimizer.adjust(self.parameters)

    def __str__(self):
        return "mnist_CNN_augmented / " + str(self.optimizer)

class LeNet_v2(Optimizable, nn.Module):
    def __init__(self, num_inp, num_out, optimizer, device = torch.device('cuda')):
        
        nn.Module.__init__(self)
        self.device = device
        self.conv1 = nn.Conv2d(num_inp, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        #self.fc1 = nn.Linear(4*4*50, 500)
        self.fc1 = nn.Linear(1250, 500)
        self.fc2 = nn.Linear(500, 10)

        #print(self.conv1.weight)
        parameters = {
            "w1": self.conv1.weight,
            "b1": self.conv1.bias,
            "w2": self.conv2.weight,
            "b2": self.conv2.bias,
            "w3": self.fc1.weight,
            "b3": self.fc1.bias,
            "w4": self.fc2.weight,
            "b4": self.fc2.bias,

            #Data augmentation
            "prob": torch.tensor(0.5, device=self.device),
            "mag": torch.tensor(1.0, device=self.device),
        }
        Optimizable.__init__(self, parameters, optimizer)

    '''
    def forward(self, x): #Sature la memoire ???
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        #x = x.view(-1, 4*4*50)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    '''
    def forward(self, x):

        if random.random() < self.parameters["prob"].item():
            #print(self.parameters["prob"])
            #x = [T.ToTensor()(
            #        TF.affine(img=T.ToPILImage()(im), angle=self.parameters["mag"], translate=(0,0), scale=1, shear=0, resample=0, fillcolor=None))
            #    for im in torch.unbind(x,dim=0)]
            #x = torch.stack(x,dim=0)

            #x = [ndimage.rotate(im, self.parameters["mag"], reshape=False)
            #    for im in torch.unbind(x,dim=0)]
            #x = torch.stack(x,dim=0)

            #x = [im + self.parameters["mag"]
            #    for im in torch.unbind(x,dim=0)]
            #x = torch.stack(x,dim=0)
            
            batch_size = x.shape[0]
            # create transformation (rotation)
            alpha = self.parameters["mag"] * 180 # in degrees
            angle = torch.ones(batch_size, device=self.device) * alpha

            # define the rotation center
            center = torch.ones(batch_size, 2, device=self.device)
            center[..., 0] = x.shape[3] / 2  # x
            center[..., 1] = x.shape[2] / 2  # y

            #print(x.shape, center)
            # define the scale factor
            scale = torch.ones(batch_size, device=self.device)

            # compute the transformation matrix
            M = kornia.get_rotation_matrix2d(center, angle, scale)

            # apply the transformation to original image
            x = kornia.warp_affine(x, M, dsize=(x.shape[2], x.shape[3])) #dsize=(h, w)

        #print("Start Shape ", x.shape)
        out = F.relu(F.conv2d(input=x, weight=self.parameters["w1"], bias=self.parameters["b1"]))
        #print("Shape ", out.shape)
        out = F.max_pool2d(out, 2)
        #print("Shape ", out.shape)
        out = F.relu(F.conv2d(input=out, weight=self.parameters["w2"], bias=self.parameters["b2"]))
        #print("Shape ", out.shape)
        out = F.max_pool2d(out, 2)
        #print("Shape ", out.shape)
        out = out.view(out.size(0), -1)
        #print("Shape ", out.shape)
        out = F.relu(F.linear(out, self.parameters["w3"], self.parameters["b3"]))
        #print("Shape ", out.shape)
        out = F.linear(out, self.parameters["w4"], self.parameters["b4"])
        #print("Shape ", out.shape)
        return F.log_softmax(out, dim=1)
    
    def initialize(self):
        self.optimizer.initialize()

    def adjust(self):
        self.optimizer.adjust(self.parameters)

    def adjust_val(self):
        self.optimizer.adjust_val(self.parameters)

    def eval(self):
        self.parameters['prob']=torch.tensor(0.0, device=self.device)

    def __str__(self):
        return "mnist_CNN_augmented / " + str(self.optimizer)