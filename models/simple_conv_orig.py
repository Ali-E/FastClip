import torch
import torch.nn as nn
import torch.nn.functional as F

from models.spectral_normalization_deflate_complex_both_bn import SpectralNorm



class CNNBN(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, dilation=1, padding_mode='zeros', bias=False, device='cpu', clip_flag=True, clip=1., clip_steps=50, clip_opt_iter=1, init_delay=0, summary=False, identifier=0, writer=None, bn=True, save_info=False):
        super(CNNBN, self).__init__()
        self.input_size = [28]

        self.sub_conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, padding_mode=padding_mode, bias=bias)

        self.kernels = [(self.sub_conv1.weight, self.input_size)]

        self.bn1 = nn.BatchNorm2d(out_planes, momentum=0.1, track_running_stats=True)
        self.bn_flag = bn

    def get_all_kernels(self):
        return self.kernels

    def forward(self, x):
        self.input_size[0] = x.shape[-1]
        x = self.sub_conv1(x)
        if self.bn_flag:
            x = self.bn1(x)
        return x


class SimpleConv_orig(nn.Module):
    def __init__(self, in_chan=3, k=1, num_classes=10, leaky_relu=False, device='cpu', clip_flag=True, clip=1., clip_steps=50, clip_opt_iter=5, init_delay=0, summary=False, identifier=0, writer=None, bias=True, bn=True, kernel_size=3, stride=1, padding=1, dilation=1, padding_mode='zeros', lin_layer=None):
        super(SimpleConv_orig, self).__init__()
        self.kernels = []
        self.conv1 = CNNBN(in_chan, 64, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False, padding_mode=padding_mode, device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+1, bn=bn)
        self.kernels.extend(self.conv1.get_all_kernels())

        W = 32
        if in_chan == 1:
            W = 28

        outdim = (W - kernel_size+2*padding)//stride + 1
        size = outdim*outdim*64

        if lin_layer is not None:
            size = lin_layer

        self.fc1_ = nn.Linear(size, 10)  

        self.fc1 = self.fc1_


    def get_all_kernels(self):
        return self.kernels


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        return x
    


def simpleConv_orig(**kwargs):
    return SimpleConv_orig(**kwargs)



