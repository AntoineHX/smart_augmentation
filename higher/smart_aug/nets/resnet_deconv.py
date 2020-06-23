'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

https://github.com/yechengxi/deconvolution
'''


import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.nn.modules import conv
from torch.nn.modules.utils import _pair

from functools import partial

__all__ = ['ResNet18_DC', 'ResNet34_DC', 'ResNet50_DC', 'ResNet101_DC', 'ResNet152_DC', 'WRN_DC26_10']

### Deconvolution ###

#iteratively solve for inverse sqrt of a matrix
def isqrt_newton_schulz_autograd(A, numIters):
    dim = A.shape[0]
    normA=A.norm()
    Y = A.div(normA)
    I = torch.eye(dim,dtype=A.dtype,device=A.device)
    Z = torch.eye(dim,dtype=A.dtype,device=A.device)

    for i in range(numIters):
        T = 0.5*(3.0*I - Z@Y)
        Y = Y@T
        Z = T@Z
    #A_sqrt = Y*torch.sqrt(normA)
    A_isqrt = Z / torch.sqrt(normA)
    return A_isqrt

def isqrt_newton_schulz_autograd_batch(A, numIters):
    batchSize,dim,_ = A.shape
    normA=A.view(batchSize, -1).norm(2, 1).view(batchSize, 1, 1)
    Y = A.div(normA)
    I = torch.eye(dim,dtype=A.dtype,device=A.device).unsqueeze(0).expand_as(A)
    Z = torch.eye(dim,dtype=A.dtype,device=A.device).unsqueeze(0).expand_as(A)

    for i in range(numIters):
        T = 0.5*(3.0*I - Z.bmm(Y))
        Y = Y.bmm(T)
        Z = T.bmm(Z)
    #A_sqrt = Y*torch.sqrt(normA)
    A_isqrt = Z / torch.sqrt(normA)

    return A_isqrt



#deconvolve channels
class ChannelDeconv(nn.Module):
    def __init__(self,  block, eps=1e-2,n_iter=5,momentum=0.1,sampling_stride=3):
        super(ChannelDeconv, self).__init__()

        self.eps = eps
        self.n_iter=n_iter
        self.momentum=momentum
        self.block = block

        self.register_buffer('running_mean1', torch.zeros(block, 1))
        #self.register_buffer('running_cov', torch.eye(block))
        self.register_buffer('running_deconv', torch.eye(block))
        self.register_buffer('running_mean2', torch.zeros(1, 1))
        self.register_buffer('running_var', torch.ones(1, 1))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.sampling_stride=sampling_stride
    def forward(self, x):
        x_shape = x.shape
        if len(x.shape)==2:
            x=x.view(x.shape[0],x.shape[1],1,1)
        if len(x.shape)==3:
            print('Error! Unsupprted tensor shape.')

        N, C, H, W = x.size()
        B = self.block

        #take the first c channels out for deconv
        c=int(C/B)*B
        if c==0:
            print('Error! block should be set smaller.')

        #step 1. remove mean
        if c!=C:
            x1=x[:,:c].permute(1,0,2,3).contiguous().view(B,-1)
        else:
            x1=x.permute(1,0,2,3).contiguous().view(B,-1)

        if self.sampling_stride > 1 and H >= self.sampling_stride and W >= self.sampling_stride:
            x1_s = x1[:,::self.sampling_stride**2]
        else:
            x1_s=x1

        mean1 = x1_s.mean(-1, keepdim=True)

        if self.num_batches_tracked==0:
            self.running_mean1.copy_(mean1.detach())
        if self.training:
            self.running_mean1.mul_(1-self.momentum)
            self.running_mean1.add_(mean1.detach()*self.momentum)
        else:
            mean1 = self.running_mean1

        x1=x1-mean1

        #step 2. calculate deconv@x1 = cov^(-0.5)@x1
        if self.training:
            cov = x1_s @ x1_s.t() / x1_s.shape[1] + self.eps * torch.eye(B, dtype=x.dtype, device=x.device)
            deconv = isqrt_newton_schulz_autograd(cov, self.n_iter)

        if self.num_batches_tracked==0:
            #self.running_cov.copy_(cov.detach())
            self.running_deconv.copy_(deconv.detach())

        if self.training:
            #self.running_cov.mul_(1-self.momentum)
            #self.running_cov.add_(cov.detach()*self.momentum)
            self.running_deconv.mul_(1 - self.momentum)
            self.running_deconv.add_(deconv.detach() * self.momentum)
        else:
            # cov = self.running_cov
            deconv = self.running_deconv

        x1 =deconv@x1

        #reshape to N,c,J,W
        x1 = x1.view(c, N, H, W).contiguous().permute(1,0,2,3)

        # normalize the remaining channels
        if c!=C:
            x_tmp=x[:, c:].view(N,-1)
            if self.sampling_stride > 1 and H>=self.sampling_stride and W>=self.sampling_stride:
                x_s = x_tmp[:, ::self.sampling_stride ** 2]
            else:
                x_s = x_tmp

            mean2=x_s.mean()
            var=x_s.var()

            if self.num_batches_tracked == 0:
                self.running_mean2.copy_(mean2.detach())
                self.running_var.copy_(var.detach())

            if self.training:
                self.running_mean2.mul_(1 - self.momentum)
                self.running_mean2.add_(mean2.detach() * self.momentum)
                self.running_var.mul_(1 - self.momentum)
                self.running_var.add_(var.detach() * self.momentum)
            else:
                mean2 = self.running_mean2
                var = self.running_var

            x_tmp = (x[:, c:] - mean2) / (var + self.eps).sqrt()
            x1 = torch.cat([x1, x_tmp], dim=1)


        if self.training:
            self.num_batches_tracked.add_(1)

        if len(x_shape)==2:
            x1=x1.view(x_shape)
        return x1

#An alternative implementation
class Delinear(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True, eps=1e-5, n_iter=5, momentum=0.1, block=512):
        super(Delinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()



        if block > in_features:
            block = in_features
        else:
            if in_features%block!=0:
                block=math.gcd(block,in_features)
                print('block size set to:', block)
        self.block = block
        self.momentum = momentum
        self.n_iter = n_iter
        self.eps = eps
        self.register_buffer('running_mean', torch.zeros(self.block))
        self.register_buffer('running_deconv', torch.eye(self.block))


    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):

        if self.training:

            # 1. reshape
            X=input.view(-1, self.block)

            # 2. subtract mean
            X_mean = X.mean(0)
            X = X - X_mean.unsqueeze(0)
            self.running_mean.mul_(1 - self.momentum)
            self.running_mean.add_(X_mean.detach() * self.momentum)

            # 3. calculate COV, COV^(-0.5), then deconv
            # Cov = X.t() @ X / X.shape[0] + self.eps * torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
            Id = torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
            Cov = torch.addmm(self.eps, Id, 1. / X.shape[0], X.t(), X)
            deconv = isqrt_newton_schulz_autograd(Cov, self.n_iter)
            # track stats for evaluation
            self.running_deconv.mul_(1 - self.momentum)
            self.running_deconv.add_(deconv.detach() * self.momentum)

        else:
            X_mean = self.running_mean
            deconv = self.running_deconv

        w = self.weight.view(-1, self.block) @ deconv
        b = self.bias
        if self.bias is not None:
            b = b - (w @ (X_mean.unsqueeze(1))).view(self.weight.shape[0], -1).sum(1)
        w = w.view(self.weight.shape)
        return F.linear(input, w, b)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )



class FastDeconv(conv._ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,groups=1,bias=True, eps=1e-5, n_iter=5, momentum=0.1, block=64, sampling_stride=3,freeze=False,freeze_iter=100):
        self.momentum = momentum
        self.n_iter = n_iter
        self.eps = eps
        self.counter=0
        self.track_running_stats=True
        super(FastDeconv, self).__init__(
            in_channels, out_channels,  _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation),
            False, _pair(0), groups, bias, padding_mode='zeros')

        if block > in_channels:
            block = in_channels
        else:
            if in_channels%block!=0:
                block=math.gcd(block,in_channels)

        if groups>1:
            #grouped conv
            block=in_channels//groups

        self.block=block

        self.num_features = kernel_size**2 *block
        if groups==1:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_deconv', torch.eye(self.num_features))
        else:
            self.register_buffer('running_mean', torch.zeros(kernel_size ** 2 * in_channels))
            self.register_buffer('running_deconv', torch.eye(self.num_features).repeat(in_channels // block, 1, 1))

        self.sampling_stride=sampling_stride*stride
        self.counter=0
        self.freeze_iter=freeze_iter
        self.freeze=freeze

    def forward(self, x):
        N, C, H, W = x.shape
        B = self.block
        frozen=self.freeze and (self.counter>self.freeze_iter)
        if self.training and self.track_running_stats:
            self.counter+=1
            self.counter %= (self.freeze_iter * 10)

        if self.training and (not frozen):

            # 1. im2col: N x cols x pixels -> N*pixles x cols
            if self.kernel_size[0]>1:
                X = torch.nn.functional.unfold(x, self.kernel_size,self.dilation,self.padding,self.sampling_stride).transpose(1, 2).contiguous()
            else:
                #channel wise
                X = x.permute(0, 2, 3, 1).contiguous().view(-1, C)[::self.sampling_stride**2,:]

            if self.groups==1:
                # (C//B*N*pixels,k*k*B)
                X = X.view(-1, self.num_features, C // B).transpose(1, 2).contiguous().view(-1, self.num_features)
            else:
                X=X.view(-1,X.shape[-1])

            # 2. subtract mean
            X_mean = X.mean(0)
            X = X - X_mean.unsqueeze(0)

            # 3. calculate COV, COV^(-0.5), then deconv
            if self.groups==1:
                #Cov = X.t() @ X / X.shape[0] + self.eps * torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
                Id=torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
                Cov = torch.addmm(self.eps, Id, 1. / X.shape[0], X.t(), X)
                deconv = isqrt_newton_schulz_autograd(Cov, self.n_iter)
            else:
                X = X.view(-1, self.groups, self.num_features).transpose(0, 1)
                Id = torch.eye(self.num_features, dtype=X.dtype, device=X.device).expand(self.groups, self.num_features, self.num_features)
                Cov = torch.baddbmm(self.eps, Id, 1. / X.shape[1], X.transpose(1, 2), X)

                deconv = isqrt_newton_schulz_autograd_batch(Cov, self.n_iter)

            if self.track_running_stats:
                self.running_mean.mul_(1 - self.momentum)
                self.running_mean.add_(X_mean.detach() * self.momentum)
                # track stats for evaluation
                self.running_deconv.mul_(1 - self.momentum)
                self.running_deconv.add_(deconv.detach() * self.momentum)

        else:
            X_mean = self.running_mean
            deconv = self.running_deconv

        #4. X * deconv * conv = X * (deconv * conv)
        if self.groups==1:
            w = self.weight.view(-1, self.num_features, C // B).transpose(1, 2).contiguous().view(-1,self.num_features) @ deconv
            b = self.bias - (w @ (X_mean.unsqueeze(1))).view(self.weight.shape[0], -1).sum(1)
            w = w.view(-1, C // B, self.num_features).transpose(1, 2).contiguous()
        else:
            w = self.weight.view(C//B, -1,self.num_features)@deconv
            b = self.bias - (w @ (X_mean.view( -1,self.num_features,1))).view(self.bias.shape)

        w = w.view(self.weight.shape)
        x= F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups)

        return x

### ResNet

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, deconv=None):
        super(BasicBlock, self).__init__()
        if deconv:
            self.conv1 = deconv(in_planes, planes, kernel_size=3, stride=stride, padding=1)
            self.conv2 = deconv(planes, planes, kernel_size=3, stride=1, padding=1)
            self.deconv = True
        else:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.deconv = False


        self.shortcut = nn.Sequential()

        if not deconv:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
            #self.bn1 = nn.GroupNorm(planes//16,planes)
            #self.bn2 = nn.GroupNorm(planes//16,planes)

            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                    #nn.GroupNorm(self.expansion * planes//16,self.expansion * planes)
                )
        else:
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    deconv(in_planes, self.expansion*planes, kernel_size=1, stride=stride)
                )

    def forward(self, x):

        if self.deconv:
            out = F.relu(self.conv1(x))
            out = self.conv2(out)
            out += self.shortcut(x)
            out = F.relu(out)
            return out

        else: #self.batch_norm:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, deconv=None):
        super(Bottleneck, self).__init__()


        if deconv:
            self.deconv = True
            self.conv1 = deconv(in_planes, planes, kernel_size=1)
            self.conv2 = deconv(planes, planes, kernel_size=3, stride=stride, padding=1)
            self.conv3 = deconv(planes, self.expansion*planes, kernel_size=1)

        else:
            self.deconv = False

            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.bn3 = nn.BatchNorm2d(self.expansion * planes)
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()

        if not deconv:
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )
        else:
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    deconv(in_planes, self.expansion * planes, kernel_size=1, stride=stride)
                )

    def forward(self, x):

        """
        No batch normalization for deconv.
        """
        if self.deconv:
            out = F.relu((self.conv1(x)))
            out = F.relu((self.conv2(out)))
            out = self.conv3(out)
            out += self.shortcut(x)
            out = F.relu(out)
            return out
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, deconv=None,channel_deconv=None):
        super(ResNet, self).__init__()
        self.in_planes = 64

        if deconv:
            self.deconv = True
            self.conv1 = deconv(3, 64, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)


        if not deconv:
            self.bn1 = nn.BatchNorm2d(64)

        #this line is really recent, take extreme care if the result is not good.
        if channel_deconv:
            self.deconv1=channel_deconv()

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, deconv=deconv)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, deconv=deconv)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, deconv=deconv)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, deconv=deconv)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, deconv):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, deconv))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if hasattr(self,'bn1'):
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if hasattr(self, 'deconv1'):
            out = self.deconv1(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def_deconv = partial(FastDeconv,bias=True, eps=1e-5, n_iter=5,block=64,sampling_stride=3)
#channel_deconv=partial(ChannelDeconv, block=512,eps=1e-5, n_iter=5,sampling_stride=3) #Pas forcément conseillé

def ResNet18_DC(num_classes,deconv=def_deconv,channel_deconv=None):
    return ResNet(BasicBlock, [2,2,2,2],num_classes=num_classes, deconv=deconv,channel_deconv=channel_deconv)

def ResNet34_DC(num_classes,deconv=def_deconv,channel_deconv=None):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes, deconv=deconv,channel_deconv=channel_deconv)

def ResNet50_DC(num_classes,deconv=def_deconv,channel_deconv=None):
    return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes, deconv=deconv,channel_deconv=channel_deconv)

def ResNet101_DC(num_classes,deconv=def_deconv,channel_deconv=None):
    return ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes, deconv=deconv,channel_deconv=channel_deconv)

def ResNet152_DC(num_classes,deconv=def_deconv,channel_deconv=None):
    return ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes, deconv=deconv,channel_deconv=channel_deconv)

import math
class Wide_ResNet_Cifar_DC(nn.Module):

    def __init__(self, block, layers, wfactor, num_classes=10, deconv=None, channel_deconv=None):
        super(Wide_ResNet_Cifar_DC, self).__init__()
        self.depth=layers[0]*6+2
        self.widen_factor=wfactor

        self.inplanes = 16
        self.conv1 = deconv(3, 16, kernel_size=3, stride=1, padding=1)
        if channel_deconv:
            self.deconv1=channel_deconv()
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16*wfactor, layers[0], stride=1, deconv=deconv)
        self.layer2 = self._make_layer(block, 32*wfactor, layers[1], stride=2, deconv=deconv)
        self.layer3 = self._make_layer(block, 64*wfactor, layers[2], stride=2, deconv=deconv)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64*block.expansion*wfactor, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride, deconv):
        # downsample = None
        # if stride != 1 or self.inplanes != planes * block.expansion:
        #     downsample = nn.Sequential(
        #         nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(planes * block.expansion)
        #     )

        # layers = []
        # layers.append(block(self.inplanes, planes, stride, downsample))
        # self.inplanes = planes * block.expansion
        # for _ in range(1, blocks):
        #     layers.append(block(self.inplanes, planes))

        # return nn.Sequential(*layers)
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, stride, deconv))
            self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if hasattr(self, 'deconv1'):
            out = self.deconv1(out)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def __str__(self):
        """ Get name of model

        """
        return "Wide_ResNet_cifar_DC%d_%d"%(self.depth,self.widen_factor)

def WRN_DC26_10(depth=26, width=10, deconv=def_deconv, channel_deconv=None, **kwargs):
    assert (depth - 2) % 6 == 0
    n = int((depth - 2) / 6)
    return Wide_ResNet_Cifar_DC(BasicBlock, [n, n, n], width, deconv=deconv,channel_deconv=channel_deconv, **kwargs)

def test():
    net = ResNet18_DC()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()