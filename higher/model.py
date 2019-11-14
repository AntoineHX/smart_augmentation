import math
import torch
import torch.nn as nn
import torch.nn.functional as F

## Basic CNN ##
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

## Wide ResNet ##
#https://github.com/xternalz/WideResNet-pytorch/blob/master/wideresnet.py
#https://github.com/arcelien/pba/blob/master/pba/wrn.py
#https://github.com/szagoruyko/wide-residual-networks/blob/master/pytorch/resnet.py

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    #def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
    def __init__(self, num_classes, wrn_size, depth=28, dropRate=0.0):
        super(WideResNet, self).__init__()

        self.kernel_size = wrn_size
        self.depth=depth
        filter_size = 3
        nChannels = [min(self.kernel_size, 16), self.kernel_size, self.kernel_size * 2, self.kernel_size * 4]
        strides = [1, 2, 2]  # stride for each resblock

        #nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(filter_size, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, strides[0], dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, strides[1], dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, strides[2], dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

    def architecture(self):
        return super(WideResNet, self).__str__()

    def __str__(self):
        return "WideResNet(s{}-d{})".format(self.kernel_size, self.depth)
