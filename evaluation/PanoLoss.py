from .utils_torch import SphericalHarmonics

import torch
import torch.nn as nn
import numpy as np
from IPython import embed

def pano(a,B,device='cuda'):
            kernel=lambda x:SphericalHarmonics.from_tensor(x,device).reconstruct_to_canvas(device).data
            return torch.stack([kernel(a[i])  for i in range(B)])

'''
    PanoLoss compute the loss from sphere 
    harmonics in panorama
    src, tar: [B, 27]
    ret: [B]
'''
class PanoLoss(nn.Module):
    def __init__(self):
        super(PanoLoss,self).__init__()
        self.loss=nn.MSELoss()
        return 
    def forward(self,src,tar):
        B, _ = src.shape
        ta = pano(src,B)
        tb = pano(tar,B)
        ret = self.loss(ta,tb)
        return ret   
        

