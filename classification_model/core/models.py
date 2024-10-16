import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
from torch import nn
import torch.nn.functional as F

DROPOUT = 0.1

# https://arxiv.org/abs/2103.13620
class SubSpectralNorm(nn.Module):
    def __init__(self, channels, sub_bands, eps=1e-5):
        super().__init__()
        self.sub_bands = sub_bands
        self.bn = nn.BatchNorm2d(channels*sub_bands, eps=eps)

    def forward(self, x):
        N, C, F, T = x.size()
        x = x.view(N, C * self.sub_bands, F // self.sub_bands, T)
        
        x = self.bn(x)
        return x.view(N, C, F, T)


class NormalBlock(nn.Module):
    def __init__(self, n_chan: int, *, dilation: int = 1, dropout: float = DROPOUT, use_subspectral: bool = True):
        super().__init__()
        norm_layer = SubSpectralNorm(n_chan, 5) if use_subspectral else nn.BatchNorm2d(n_chan)
        self.f2 = nn.Sequential(
            nn.Conv2d(n_chan, n_chan, kernel_size=(3, 1), padding="same", groups=n_chan),
            norm_layer,
        )
        self.f1 = nn.Sequential(
            nn.Conv2d(n_chan, n_chan, kernel_size=(1, 3), padding="same", groups=n_chan, dilation=(1, dilation)),
            nn.BatchNorm2d(n_chan),
            nn.SiLU(),
            nn.Conv2d(n_chan, n_chan, kernel_size=1),
            nn.Dropout2d(dropout)
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        n_freq = x.shape[2]
        x1 = self.f2(x)

        x2 = torch.mean(x1, dim=2, keepdim=True)
        x2 = self.f1(x2)
        x2 = x2.repeat(1, 1, n_freq, 1)

        return self.activation(x + x1 + x2)


class TransitionBlock(nn.Module):
    def __init__(self, in_chan: int, out_chan: int, *, dilation: int = 1, stride: int = 1, dropout: float = DROPOUT, use_subspectral: bool = True):
        super().__init__()

        if stride == 1:
            conv = nn.Conv2d(out_chan, out_chan, kernel_size=(3, 1), groups=out_chan, padding="same")
        else:
            conv = nn.Conv2d(out_chan, out_chan, kernel_size=(3, 1), stride=(stride, 1), groups=out_chan, padding=(1, 0))

        norm_layer = SubSpectralNorm(out_chan, 5) if use_subspectral else nn.BatchNorm2d(out_chan)
        self.f2 = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(),
            conv,
            norm_layer,
        )

        self.f1 = nn.Sequential(
            nn.Conv2d(out_chan, out_chan, kernel_size=(1, 3), padding="same", groups=out_chan, dilation=(1, dilation)),
            nn.BatchNorm2d(out_chan),
            nn.SiLU(),
            nn.Conv2d(out_chan, out_chan, kernel_size=1),
            nn.Dropout2d(dropout)
        )

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.f2(x)
        n_freq = x.shape[2]
        x1 = torch.mean(x, dim=2, keepdim=True)
        x1 = self.f1(x1)
        x1 = x1.repeat(1, 1, n_freq, 1)

        return self.activation(x + x1)


class BcResNetModel(nn.Module):
    def __init__(self, n_class: int = 35, *, scale: int = 1, dropout: float = DROPOUT, use_subspectral: bool = False):
        super().__init__()

        self.input_conv = nn.Conv2d(1, 16*scale, kernel_size=(5, 5), stride=(2, 1), padding=2)

        self.t1 = TransitionBlock(16*scale, 8*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n11 = NormalBlock(8*scale, dropout=dropout, use_subspectral=use_subspectral)

        self.t2 = TransitionBlock(8*scale, 12*scale, dilation=2, stride=2, dropout=dropout, use_subspectral=use_subspectral)
        self.n21 = NormalBlock(12*scale, dilation=2, dropout=dropout, use_subspectral=use_subspectral) 

        self.t3 = TransitionBlock(12*scale, 16*scale, dilation=4, stride=2, dropout=dropout, use_subspectral=use_subspectral)
        self.n31 = NormalBlock(16*scale, dilation=4, dropout=dropout, use_subspectral=use_subspectral)
        self.n32 = NormalBlock(16*scale, dilation=4, dropout=dropout, use_subspectral=use_subspectral)
        self.n33 = NormalBlock(16*scale, dilation=4, dropout=dropout, use_subspectral=use_subspectral)

        self.t4 = TransitionBlock(16*scale, 20*scale, dilation=8, dropout=dropout, use_subspectral=use_subspectral)
        self.n41 = NormalBlock(20*scale, dilation=8, dropout=dropout, use_subspectral=use_subspectral)
        self.n42 = NormalBlock(20*scale, dilation=8, dropout=dropout, use_subspectral=use_subspectral)
        self.n43 = NormalBlock(20*scale, dilation=8, dropout=dropout, use_subspectral=use_subspectral)

        self.dw_conv = nn.Conv2d(20*scale, 20*scale, kernel_size=(5, 5), groups=20)
        self.onexone_conv = nn.Conv2d(20*scale, 32*scale, kernel_size=1)

        self.head_conv = nn.Conv2d(32*scale, n_class, kernel_size=1)
    
    def forward(self, x: torch.Tensor):
        #import pdb; pdb.set_trace()
        x = self.input_conv(x)
        x = self.t1(x)
        x = self.n11(x)

        x = self.t2(x)
        x = self.n21(x)

        x = self.t3(x)
        x = self.n31(x)
        x = self.n32(x)
        x = self.n33(x)

        x = self.t4(x)
        x = self.n41(x)
        x = self.n42(x)
        x = self.n43(x)

        x = self.dw_conv(x)
        x = self.onexone_conv(x)

        x = torch.mean(x, dim=3, keepdim=True)
        x = self.head_conv(x)
        x = x.squeeze()

        try: 
            x = torch.mean(x, dim=2, keepdim=True)
            x = x.squeeze()
        except:
            #import pdb; pdb.set_trace()
            pass
        return F.log_softmax(x, dim=-1)

        
class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        #x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        #x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        #x = x.squeeze()
        return F.log_softmax(x, dim=2)

class M11(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=4, n_channel=64):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.conv3 = nn.Conv1d(n_channel, n_channel, kernel_size=3,padding=1)
        self.bn3 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)

        self.conv4 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3,padding=1)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.conv5 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3,padding=1)
        self.bn5 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
 
        self.conv6 = nn.Conv1d(2 * n_channel, 4 * n_channel, kernel_size=3,padding=1)
        self.bn6 = nn.BatchNorm1d(4 * n_channel)
        self.conv7 = nn.Conv1d(4 * n_channel, 4 * n_channel, kernel_size=3,padding=1)
        self.bn7 = nn.BatchNorm1d(4 * n_channel)
        self.conv8 = nn.Conv1d(4 * n_channel, 4 * n_channel, kernel_size=3,padding=1)
        self.bn8 = nn.BatchNorm1d(4 * n_channel)
        self.pool4 = nn.MaxPool1d(4)

        self.conv9 = nn.Conv1d(4 * n_channel, 8 * n_channel, kernel_size=3,padding=1)
        self.bn9 = nn.BatchNorm1d(8 * n_channel)
        self.conv10 = nn.Conv1d(8 * n_channel, 8 * n_channel, kernel_size=3,padding=1)
        self.bn10 = nn.BatchNorm1d(8 * n_channel)

        self.fc1 = nn.Linear(8 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool2(x)

        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.conv5(x)
        x = F.relu(self.bn5(x))
        x = self.pool3(x)

        x = self.conv6(x)
        x = F.relu(self.bn6(x))
        x = self.conv7(x)
        x = F.relu(self.bn7(x))
        x = self.conv8(x)
        x = F.relu(self.bn8(x))
        x = self.pool4(x)

        x = self.conv9(x)
        x = F.relu(self.bn9(x))
        x = self.conv10(x)
        x = F.relu(self.bn10(x))

        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)