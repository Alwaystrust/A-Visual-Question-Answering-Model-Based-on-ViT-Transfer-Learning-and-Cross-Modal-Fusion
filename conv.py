import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np



def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return ModuleParallel(nn.Conv1d(in_planes, out_planes, kernel_size=3,stride=stride, padding=1, bias=bias))

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return ModuleParallel(nn.Conv1d(in_planes, out_planes, kernel_size=1,stride=stride, padding=0, bias=bias))


class Exchange(nn.Module):
    def __init__(self):
        super(Exchange, self).__init__()

    def forward(self, x, bn, bn_threshold):
        bn1, bn2 = bn[0].weight.abs(), bn[1].weight.abs()
        x1, x2 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
        x1[:, bn1 >= bn_threshold] = x[0][:, bn1 >= bn_threshold]
        x1[:, bn1 < bn_threshold] = x[1][:, bn1 < bn_threshold]
        x2[:, bn2 >= bn_threshold] = x[1][:, bn2 >= bn_threshold]
        x2[:, bn2 < bn_threshold] = x[0][:, bn2 < bn_threshold]
        return [x1, x2]


class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


class BatchNorm1dParallel(nn.Module):
    def __init__(self, num_features, num_parallel):
        super(BatchNorm1dParallel, self).__init__()
        for i in range(num_parallel):
            setattr(self, 'bn_' + str(i), nn.BatchNorm1d(num_features))

    def forward(self, x_parallel):
        return [getattr(self, 'bn_' + str(i))(x) for i, x in enumerate(x_parallel)]


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, num_parallel, bn_threshold, stride=1, downsample = None):
        super(Bottleneck, self).__init__()
        
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = BatchNorm1dParallel(planes, num_parallel)
        
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = BatchNorm1dParallel(planes, num_parallel)
        
        self.conv3 = conv1x1(planes, planes * 4)
        self.bn3 = BatchNorm1dParallel(planes * 4, num_parallel)
        
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        
        self.num_parallel = num_parallel
        self.downsample = downsample
        self.stride = stride

        self.exchange = Exchange()
        self.bn_threshold = bn_threshold
        
        self.bn2_list = []
        for module in self.bn2.modules():
            if isinstance(module, nn.BatchNorm1d):
                self.bn2_list.append(module)

    def forward(self, x):
        residual = x
        out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        #这里体现了这个输入的x是[data,data]，所以这就是为什么需要parallel来包裹所有的函数，因为需要分别两边一起计算
        if len(x) > 1:
            out = self.exchange(out, self.bn2_list, self.bn_threshold)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        #这两步是残差连接，downsample的意思就是说如果前面的卷积bn卷积bn有维度缩减的话，那么这一部分肯定要对前面的网络做一个下采样，从而能够完成拼接。
        #我们假设无 先简化
        if self.downsample is not None:
            residual = self.downsample(x)

        out = [out[l] + residual[l] for l in range(self.num_parallel)]
        out = self.relu(out)

        return out