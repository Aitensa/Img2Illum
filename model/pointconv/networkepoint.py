"""PointAR model architecture definition
"""

from re import I
import torch 
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from etc.pointconv_util import PointConvDensitySetAbstraction
from etc.epconv_v1_util import EquSO3ConvModel 
from etc.epconv_v2_util import EquFeatExtraction
from etc.epconv_v3_util import EquFeatModule
class EquPointConvModelV1(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.save_hyperparameters()
        self.opt = self.hparams['opt']
        n_points = int(self.hparams['n_points'])
        self.ep = EquSO3ConvModel(self.opt)
        self.sa1 = PointConvDensitySetAbstraction(
            npoint=n_points, nsample=32, in_channel=3+3,
            mlp=[64, 128], bandwidth=0.1, group_all=False
        )

        self.sa2 = PointConvDensitySetAbstraction(
            npoint=1, nsample=0, in_channel=128 + 3,
            mlp=[128, 256], bandwidth=0.2, group_all=True
        )
        
        self.fc3 = nn.Linear(256+64, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.drop4 = nn.Dropout(0.2)

        self.fc5 = nn.Linear(64, self.hparams['n_shc'])

    def forward(self, xyz, rgb):
        B, nc, _ = xyz.shape
        #print(xyz.shape)
        #yw_xyz = self.ep(xyz.permute(0,2,1))
        yw_rgb = self.ep(rgb.permute(0,2,1))
        l1_xyz, l1_points = self.sa1(xyz, rgb)
        _, l2_points = self.sa2(l1_xyz, l1_points)
        #print(yw_xyz.shape) 
        x = torch.cat((l2_points.view(B, 256),yw_rgb.view(B,-1)),1)
        x = self.drop3(F.relu(self.bn3(self.fc3(x))))
        x = self.drop4(F.relu(self.bn4(self.fc4(x))))
        x = self.fc5(x)
        x = F.relu(x)

        return x





class EquPointConvModelV2(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.save_hyperparameters()
        self.opt = self.hparams['opt']
        n_points = int(self.hparams['n_points'])
        self.ep = EquFeatExtraction(self.opt)
        self.sa1 = PointConvDensitySetAbstraction(
            npoint=160, nsample=16, in_channel=128+3,
            mlp=[64, 128], bandwidth=0.2, group_all=False
        )

        self.sa2 = PointConvDensitySetAbstraction(
            npoint=1, nsample=0, in_channel=128 + 3,
            mlp=[128, 256], bandwidth=0.2, group_all=True
        )
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.drop4 = nn.Dropout(0.2)

        self.fc5 = nn.Linear(64, self.hparams['n_shc'])

    def forward(self, xyz, rgb):
        B, nc, _ = xyz.shape
        #print(xyz.shape)
        #yw_xyz = self.ep(xyz.permute(0,2,1))
        # yw_rgb = self.ep(rgb.permute(0,2,1))
        eq_feat = self.ep(xyz.permute(0,2,1),rgb.permute(0,2,1))
        l1_xyz, l1_points = self.sa1(eq_feat.xyz, eq_feat.feats.max(-1)[0])
        _, l2_points = self.sa2(l1_xyz, l1_points)
        #print(yw_xyz.shape) 
        # x = torch.cat((l2_points.view(B, 256),yw_rgb.view(B,-1)),1)
        x = l2_points.view(B,256)
        x = self.drop3(F.relu(self.bn3(self.fc3(x))))
        x = self.drop4(F.relu(self.bn4(self.fc4(x))))
        x = self.fc5(x)
        x = F.relu(x)

        return x

class EquPointConvModelV3(pl.LightningModule):
    
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.save_hyperparameters()
        self.opt = self.hparams['opt']
        n_points = int(self.hparams['n_points'])
        self.ep = EquFeatModule(self.opt)
        self.sa1 = PointConvDensitySetAbstraction(
            npoint=n_points, nsample=32, in_channel=3+3,
            mlp=[64, 128], bandwidth=0.1, group_all=False
        )

        self.sa2 = PointConvDensitySetAbstraction(
            npoint=1, nsample=0, in_channel=128 + 3,
            mlp=[128, 256], bandwidth=0.2, group_all=True
        )
        
        self.fc3 = nn.Linear(256 + 64, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.drop4 = nn.Dropout(0.2)

        self.fc5 = nn.Linear(64, self.hparams['n_shc'])

    def forward(self, xyz, rgb):
        B, nc, _ = xyz.shape
        #print(xyz.shape)
        #yw_xyz = self.ep(xyz.permute(0,2,1))
        # yw_rgb = self.ep(rgb.permute(0,2,1))
        eq_feat = self.ep(xyz.permute(0,2,1),rgb.permute(0,2,1))
        l1_xyz, l1_points = self.sa1(xyz, rgb)
        _, l2_points = self.sa2(l1_xyz, l1_points)
        #print(yw_xyz.shape) 
        # x = torch.cat((l2_points.view(B, 256),yw_rgb.view(B,-1)),1)
        
        x = torch.cat((l2_points.view(B, 256),eq_feat.view(B,-1)),1)
        x = self.drop3(F.relu(self.bn3(self.fc3(x))))
        x = self.drop4(F.relu(self.bn4(self.fc4(x))))
        x = self.fc5(x)
        x = F.relu(x)

        return x






