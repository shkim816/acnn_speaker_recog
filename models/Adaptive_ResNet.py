import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from .ACNNBlocks import ACNN2d, BasicBlock
import time
import math
from .aamsoftmax import LossFunction

class ResNet(nn.Module):
    def __init__(self, block, layers, num_filters, nOut, classlen, window, stride, mode, encoder_type, n_mels = 257):
        super(ResNet, self).__init__()

        self.stride = stride
        self.window = window[0]
        self.encoder_type = encoder_type
        
        self.inplanes   = num_filters[0]
        self.n_mels     = n_mels


        self.instancenorm   = nn.InstanceNorm1d(n_mels)

        self.conv1 = ACNN2d(in_channel=1, out_channel=num_filters[0], kernel=(7, 7), stride=2, padding=0, bias=False, freq_len=self.n_mels, time_len=window[0])
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.pool1 = nn.MaxPool2d(3, stride=(2, 2))

        self.layer1 = self._make_layer(block, num_filters[0], layers[0], flen = 62, tlen = window[1])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=(2, 2), flen = 62, tlen = window[2])
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=(2, 2), flen = 31, tlen = window[3])
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=(2, 1), flen = 16, tlen = window[4])

        self.conv5 = nn.Conv2d(num_filters[3], num_filters[4] , kernel_size=(8,1), stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(num_filters[4])


        if self.encoder_type == "SAP":
            self.sap_linear = nn.Linear(num_filters[4] * block.expansion, num_filters[4] * block.expansion)
            self.attention = self.new_parameter(num_filters[4] * block.expansion, 1)
            out_dim = num_filters[4] * block.expansion
        elif self.encoder_type == "TAP":
            self.apool = nn.AdaptiveAvgPool2d((1, 1))
            out_dim = num_filters[4] * block.expansion
        else:
            raise ValueError('Undefined encoder')

        self.fc1 = nn.Linear(out_dim, nOut)
        
        if mode == 'iden':
            self.fc2 = nn.Linear(nOut,classlen)
            self.use_label = False
        elif mode == 'veri':
            self.fc2 = LossFunction(nOut = nOut, nClasses = classlen, margin = 0.2, scale = 30)
            self.use_label = True

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, flen = 0, tlen = 0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ACNN2d(in_channel=self.inplanes, out_channel=planes * block.expansion, kernel=(1, 1), stride=stride, padding=0, bias=False, freq_len=flen, time_len=tlen),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, flen = flen, tlen = tlen))
        self.inplanes = planes * block.expansion
        if downsample is not None:
            flen = math.ceil(flen/stride[0])
            tlen = math.ceil(tlen/stride[1])
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, flen = flen, tlen = tlen))

        return nn.Sequential(*layers)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, x, label = None):
        batsize = x.shape[0]
        with torch.no_grad():
            x = self.instancenorm(x).detach()
            x = x.unfold(2, self.window, self.stride).transpose(1, 2).reshape([-1, x.shape[-2], self.window]).unsqueeze(1).contiguous().detach()
        #print(x.shape)

        #time.sleep(10000)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        #print(x.shape)
        x = self.layer1(x)
        #print(x.shape)
        x = self.layer2(x)
        #print(x.shape)
        x = self.layer3(x)
        #print(x.shape)
        x = self.layer4(x)
        #print(x.shape)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        #print(x.shape)
        x = x.view(batsize, -1, x.shape[-3], x.shape[-1]).transpose(1, 2)
        
        #time.sleep(10000)
        if self.encoder_type == "SAP":
            x = x.reshape(batsize, x.shape[1], -1)
            x = x.permute(0,2,1)
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            x = torch.sum(x * w, dim=1)
        elif self.encoder_type == "TAP":
            x = self.apool(x)
        #print(x.shape)
        x = x.view(x.size()[0], -1)
        #print(x.shape)
        x = self.fc1(x)

        if label is None:
            #print(x.shape)
            return x

        if self.use_label:
            x = self.fc2(x, label)
        else:
            x = self.fc2(x)

        return x


def MainModel(nOut, classlen, mode, nc_divide, encoder_type):
    # Number of filters

    utt_time = 295

    if nc_divide == 1:
        num_window = [23, 4, 4, 2, 1]
        stride = 16
    elif nc_divide == 2:
        num_window = [39, 8, 8, 4, 2]
        stride = 32
    elif nc_divide == 3:
        num_window = [55, 12, 12, 6, 3]
        stride = 48
    elif nc_divide == 6:
        num_window = [103, 24, 24, 12, 6]
        stride = 96
    elif nc_divide == 9:
        num_window = [151, 36, 36, 18, 9]
        stride = 144
    else:
        raise ValueError('Undefined nc_divide %d'%(nc_divide))

    print('Adaptive ResNet %d(%dx%d) with embedding size is %d, encoder %s.'%(num_window[0], int(18/nc_divide), nc_divide, nOut, encoder_type))

    num_filters = [32, 64, 128, 256, 512]
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_filters, nOut, classlen = classlen, window = num_window, stride = stride, mode = mode, encoder_type = encoder_type)
    return model, utt_time