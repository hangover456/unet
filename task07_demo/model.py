"""PyTorch 2D U-Net adaptation: 4 down/up levels, 3 output labels, base width 32.

This does not load the legacy Keras weights. Nearest-upsample + 3x3 convolution
is used for upsampling. Width is reduced from the legacy 64 to fit a 4 GB GPU.
"""
import torch
from torch import nn
class DoubleConv(nn.Sequential):
    def __init__(self,a,b):super().__init__(nn.Conv2d(a,b,3,padding=1),nn.ReLU(inplace=True),nn.Conv2d(b,b,3,padding=1),nn.ReLU(inplace=True))
class UNet2D(nn.Module):
    def __init__(self,base=32):
        super().__init__();c=[base*2**i for i in range(5)]
        self.enc=nn.ModuleList([DoubleConv(1,c[0])]+[DoubleConv(c[i-1],c[i]) for i in range(1,5)])
        self.up=nn.ModuleList([nn.Conv2d(c[i+1],c[i],3,padding=1) for i in range(3,-1,-1)])
        self.dec=nn.ModuleList([DoubleConv(c[i]*2,c[i]) for i in range(3,-1,-1)]);self.head=nn.Conv2d(base,3,1)
    def forward(self,x):
        skips=[]
        for i,layer in enumerate(self.enc):
            if i:x=torch.nn.functional.max_pool2d(x,2)
            x=layer(x);skips.append(x)
        for i,(up,dec) in enumerate(zip(self.up,self.dec)):
            s=skips[3-i];x=up(torch.nn.functional.interpolate(x,size=s.shape[-2:],mode='nearest'));x=dec(torch.cat([s,x],1))
        return self.head(x)
