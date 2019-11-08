import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, num_inp, num_out):
        super(LeNet, self).__init__()
        self._params = nn.ParameterDict({
            'w1': nn.Parameter(torch.zeros(20, num_inp, 5, 5)),
            'b1': nn.Parameter(torch.zeros(20)),
            'w2': nn.Parameter(torch.zeros(50, 20, 5, 5)),
            'b2': nn.Parameter(torch.zeros(50)),
            #'w3': nn.Parameter(torch.zeros(500,4*4*50)), #num_imp=1
            'w3': nn.Parameter(torch.zeros(500,5*5*50)), #num_imp=3
            'b3': nn.Parameter(torch.zeros(500)),
            'w4': nn.Parameter(torch.zeros(num_out, 500)),
            'b4': nn.Parameter(torch.zeros(num_out))
        })
        self.initialize()


    def initialize(self):
        nn.init.kaiming_uniform_(self._params["w1"], a=math.sqrt(5))
        nn.init.kaiming_uniform_(self._params["w2"], a=math.sqrt(5))
        nn.init.kaiming_uniform_(self._params["w3"], a=math.sqrt(5))
        nn.init.kaiming_uniform_(self._params["w4"], a=math.sqrt(5))

    def forward(self, x):
        #print("Start Shape ", x.shape)
        out = F.relu(F.conv2d(input=x, weight=self._params["w1"], bias=self._params["b1"]))
        #print("Shape ", out.shape)
        out = F.max_pool2d(out, 2)
        #print("Shape ", out.shape)
        out = F.relu(F.conv2d(input=out, weight=self._params["w2"], bias=self._params["b2"]))
        #print("Shape ", out.shape)
        out = F.max_pool2d(out, 2)
        #print("Shape ", out.shape)
        out = out.view(out.size(0), -1)
        #print("Shape ", out.shape)
        out = F.relu(F.linear(out, self._params["w3"], self._params["b3"]))
        #print("Shape ", out.shape)
        out = F.linear(out, self._params["w4"], self._params["b4"])
        #print("Shape ", out.shape)
        return F.log_softmax(out, dim=1)

    def __getitem__(self, key):
        return self._params[key]

    def __str__(self):
        return "LeNet"