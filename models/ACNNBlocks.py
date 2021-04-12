import torch
import torch.nn.functional as F
import torch.nn as nn
import time
import math

BN = False

class ChannelFreqGate(nn.Module):
    def __init__(self, freq_len, kernel, init_weight=True, batch_norm = BN):
        super(ChannelFreqGate, self).__init__()
        self.batch_norm = batch_norm

        if self.batch_norm:
            self.conv1 = nn.Conv1d(freq_len, freq_len, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm1d(freq_len)
        else:
            self.conv1 = nn.Conv1d(freq_len, freq_len, 3, padding=1, bias=True)

        self.conv2 = nn.Conv1d(freq_len, kernel, 3, padding=1, bias=True)

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = torch.mean(x, 3).transpose(1, 2)

        if self.batch_norm:
            x = F.relu(self.bn1(self.conv1(x)))
        else:
            x = F.relu(self.conv1(x))

        x = self.conv2(x).transpose(1, 2).unsqueeze(1).unsqueeze(-1)
        return x


class ChannelTimeGate(nn.Module):
    def __init__(self, time_len, kernel, init_weight=True, batch_norm = BN):
        super(ChannelTimeGate, self).__init__()
        self.batch_norm = batch_norm

        if self.batch_norm:
            self.conv1 = nn.Conv1d(time_len, time_len, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm1d(time_len)
        else :
            self.conv1 = nn.Conv1d(time_len, time_len, 3, padding=1, bias=True)

        self.conv2 = nn.Conv1d(time_len, kernel, 3, padding=1, bias=True)

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = torch.mean(x, 2).transpose(1, 2)
        if self.batch_norm:
            x = F.relu(self.bn1(self.conv1(x)))
        else: 
            x = F.relu(self.conv1(x))

        x = self.conv2(x).transpose(1, 2).unsqueeze(1).unsqueeze(-2)

        return x


class ACNN2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride=1, padding=0, dilation=1, bias=False, init_weight=True, freq_len=0, time_len=0):
        super(ACNN2d, self).__init__()
        assert len(kernel) == 2
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.freq_len = freq_len
        self.time_len = time_len


        self.channelfreq_att = ChannelFreqGate(freq_len, kernel[0])
        self.channeltime_att = ChannelTimeGate(time_len, kernel[1])

        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel[0], kernel[1]), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channel), requires_grad=True)
        else:
            self.bias = None

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        batch = x.shape[0]
        weight_f = self.channelfreq_att(x).expand(-1, self.out_channel, -1, -1, self.kernel[1])
        weight_t = self.channeltime_att(x).expand(-1, self.out_channel, -1, self.kernel[0], -1)
        weight = torch.sigmoid(weight_f + weight_t)

        weight = self.weight.unsqueeze(0).expand_as(weight)*weight
        weight = weight.view([-1, weight.shape[-3], weight.shape[-2], weight.shape[-1]])

        if self.bias is not None:
            local_bias = self.bias.expand(weight.shape[0])
        else:
            local_bias = self.bias

        x = F.conv2d(x.view([1, -1, x.shape[-2], x.shape[-1]]), weight=weight, bias=local_bias, stride=self.stride, padding=self.padding, groups=batch)

        x = x.view([batch, self.out_channel, x.shape[-2], x.shape[-1]])

        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8, flen = 0, tlen = 0):
        super(BasicBlock, self).__init__()
        self.conv1 = ACNN2d(in_channel=inplanes, out_channel=planes, kernel=(3, 3), stride=stride, padding=1, bias=False, freq_len = flen, time_len = tlen)
        self.bn1 = nn.BatchNorm2d(planes)
        if stride != 1:
            self.conv2 = ACNN2d(in_channel=planes, out_channel=planes, kernel=(3, 3), stride=1, padding=1, bias=False, freq_len=math.ceil(flen/stride[0]), time_len=math.ceil(tlen/stride[1]))
        else : 
            self.conv2 = ACNN2d(in_channel=planes, out_channel=planes, kernel=(3, 3), stride=1, padding=1, bias=False, freq_len = flen, time_len = tlen)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out