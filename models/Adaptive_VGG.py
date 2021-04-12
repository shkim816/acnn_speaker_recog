import torch
import torch.nn.functional as F
import torch.nn as nn
import time
from .ACNNBlocks import ACNN2d
from .aamsoftmax import LossFunction

class VGG(nn.Module):

    def __init__(self, nOut, classlen, stride, window, mode, encoder_type):
        super(VGG, self).__init__()

        self.stride = stride
        self.window = window[0]
        self.encoder_type = encoder_type

        self.instancenorm = nn.InstanceNorm1d(257)
        # self.conv1 = nn.Conv2d(1, 96, (7, 7), stride = 2, padding = 1, bias = False)
        self.conv1 = ACNN2d(in_channel=1, out_channel=96, kernel=(7, 7), stride=2, padding=1, bias=False, freq_len=257, time_len=window[0])
        self.bn1 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d(3, stride=(1, 2))

        # self.conv2 = nn.Conv2d(96, 256, 5, stride = 2, padding = 1, bias = False)
        self.conv2 = ACNN2d(in_channel=96, out_channel=256, kernel=(5, 5), stride=2, padding=1, bias=False, freq_len=125, time_len=window[1])
        self.bn2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(3, stride=2)

        # self.conv3 = nn.Conv2d(256, 384, (3, 3), padding = 1, bias = False)
        self.conv3 = ACNN2d(in_channel=256, out_channel=384, kernel=(3, 3), stride=1, padding=1, bias=False, freq_len=30, time_len=window[2])
        # self.conv3 = feednet(256, 384, (3, 3), padding = 1)
        self.bn3 = nn.BatchNorm2d(384)

        # self.conv4 = nn.Conv2d(384, 256, (3, 3), padding = 1, bias = False)
        self.conv4 = ACNN2d(in_channel=384, out_channel=256, kernel=(3, 3), stride=1, padding=1, bias=False, freq_len=30, time_len=window[3])
        self.bn4 = nn.BatchNorm2d(256)

        # self.conv5 = nn.Conv2d(256, 256, (3, 3), padding = 1, bias = False)
        self.conv5 = ACNN2d(in_channel=256, out_channel=256, kernel=(3, 3), stride=1, padding=1, bias=False, freq_len=30, time_len=window[4])
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d((5, 1), stride=(3, 1))

        # ACNN2d(in_channel=256, out_channel=512, kernel=(9, 1), stride=1, padding=1, bias=False, freq_len=9, time_len=1)
        self.conv6 = nn.Conv2d(256, 512, (9, 1), bias=False)
        # nn.Conv2d(256, 512, (9, 1), bias=False)
        self.bn6 = nn.BatchNorm2d(512)

        if self.encoder_type == "SAP":
            self.sap_linear = nn.Linear(512, 512)
            self.attention = self.new_parameter(512, 1)
        elif self.encoder_type == "TAP":
            self.apool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            raise ValueError('Undefined encoder')

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(512, nOut)

        if mode == 'iden':
            self.fc2 = nn.Linear(nOut,classlen)
            self.use_label = False
        elif mode == 'veri':
            self.fc2 = LossFunction(nOut = nOut, nClasses = classlen, margin = 0.2, scale = 30)
            self.use_label = True

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, x, label = None):

        batsize = x.shape[0]
        x = self.instancenorm(x).detach()  # batch x 257 x 300
        # x = x.unsqueeze(1).detach()
        x = x.unfold(2, self.window, self.stride).transpose(1, 2).reshape(-1, x.shape[-2], self.window).unsqueeze(1).contiguous().detach()
        # batch x 257 x 8 x 90 -> batch x 8 x 257 x 90 -> (8 x batch) x 1 x 257 x 90 : 0~8 / 8~16 / 16~24 ...

        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        x = F.relu(self.bn3(self.conv3(x)))

        x = F.relu(self.bn4(self.conv4(x)))

        x = self.pool5(F.relu(self.bn5(self.conv5(x))))

        x = F.relu(self.bn6(self.conv6(x)))  # 800 x 4096 x 1 x N

        x = x.view(batsize, -1, x.shape[-3], x.shape[-1]).transpose(1, 2)
        

        if self.encoder_type == "SAP":
            x = x.reshape(batsize, x.shape[1], -1)
            x = x.permute(0,2,1)
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            x = torch.sum(x * w, dim=1)
        elif self.encoder_type == "TAP":
            x = self.apool(x)

        x = x.view(x.size()[0], -1)
        x = self.fc1(x)

        if label is None:
            return x

        if self.use_label:
            x = self.fc2(x, label)
        else:
            x = self.fc2(x)

        return x


def MainModel(nOut, classlen, mode, nc_divide, encoder_type):
    # Number of filters

    utt_time = 305

    if nc_divide == 1:
        num_window = [33, 7, 1, 1, 1]
        stride = 16
    elif nc_divide == 2:
        num_window = [49, 11, 2, 2, 2]
        stride = 32
    elif nc_divide == 3:
        num_window = [65, 15, 3, 3, 3]
        stride = 48
    elif nc_divide == 6:
        num_window = [113, 27, 6, 6, 6]
        stride = 96
    elif nc_divide == 9:
        num_window = [161, 39, 9, 9, 9]
        stride = 144
    else:
        raise ValueError('Undefined nc_divide %d'%(nc_divide))

    print('Adaptive VGG-M %d(%dx%d) with embedding size is %d, encoder %s.'%(num_window[0], int(18/nc_divide), nc_divide, nOut, encoder_type))

    model = VGG(nOut, classlen = classlen, window = num_window, stride = stride, mode = mode, encoder_type = encoder_type)
    return model, utt_time
