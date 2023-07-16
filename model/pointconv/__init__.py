"""Training control and evaluation
"""

import torch
import torch.nn.functional as F
from evaluation.utils import SphericalHarmonics
from evaluation.PanoLoss import PanoLoss
from trainer.base import BaseTrainer
from model.pointconv.networkep import EPPointConvModel
from model.pointconv.network import PointConvModel
from model.pointconv.networkepoint import EquPointConvModelV1, EquPointConvModelV2, EquPointConvModelV3
class EPPointAR(EPPointConvModel, BaseTrainer):
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def calculate_train_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale
        
        shc_mse_raw = F.mse_loss(source, target)
        shc_mse = F.mse_loss(source_norm, target_norm)
        
        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }

    def calculate_valid_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale

        shc_mse_raw = F.mse_loss(source, target)
        shc_mse = F.mse_loss(source_norm, target_norm)

        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }


class PanoEPPointAR(EPPointConvModel, BaseTrainer):
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def calculate_train_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale
        
        loss=PanoLoss()
        
        shc_mse_raw = loss(source,target)
        shc_mse = loss(source_norm,target_norm)

        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }

    def calculate_valid_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale

        loss=PanoLoss()
        
        shc_mse_raw = loss(source,target)
        shc_mse = loss(source_norm,target_norm)
        # shc_mse_raw = F.mse_loss(source, target)
        # shc_mse = F.mse_loss(source_norm, target_norm)

        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }

class SHCPointAR(PointConvModel, BaseTrainer):
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def calculate_train_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale

        shc_mse_raw = F.mse_loss(source, target)
        shc_mse = F.mse_loss(source_norm, target_norm)

        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }

    def calculate_valid_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale

        shc_mse_raw = F.mse_loss(source, target)
        shc_mse = F.mse_loss(source_norm, target_norm)

        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }


class PanoPointAR(PointConvModel, BaseTrainer):
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def calculate_train_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale

        # shc_mse_raw = F.mse_loss(source, target)
        # shc_mse = F.mse_loss(source_norm, target_norm)
        loss=PanoLoss()
        
        shc_mse_raw = loss(source,target) 
        shc_mse = loss(source_norm,target_norm) 
        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }

    def calculate_valid_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale

        loss=PanoLoss()
        
        shc_mse_raw = loss(source,target) 
        shc_mse = loss(source_norm,target_norm) 
        # shc_mse_raw = F.mse_loss(source, target)
        # shc_mse = F.mse_loss(source_norm, target_norm)

        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }

class UnitePointAR(PointConvModel, BaseTrainer):
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def calculate_train_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale

        # shc_mse_raw = F.mse_loss(source, target)
        # shc_mse = F.mse_loss(source_norm, target_norm)
        loss=PanoLoss()
        
        shc_mse_raw = loss(source,target) + F.mse_loss(source,target)
        shc_mse = loss(source_norm,target_norm) + F.mse_loss(source_norm,target_norm)
        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }

    def calculate_valid_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale

        loss=PanoLoss()
        
        shc_mse_raw = loss(source,target) + F.mse_loss(source,target)
        shc_mse = loss(source_norm,target_norm) + F.mse_loss(source_norm,target_norm)
        # shc_mse_raw = F.mse_loss(source, target)
        # shc_mse = F.mse_loss(source_norm, target_norm)

        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }


class UniteEquPointAR_V1(EquPointConvModelV1, BaseTrainer):
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def calculate_train_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale
        
        loss=PanoLoss()
        
        shc_mse_raw = loss(source,target) + F.mse_loss(source,target)
        shc_mse = loss(source_norm,target_norm) + F.mse_loss(source_norm,target_norm)

        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }

    def calculate_valid_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale

        loss=PanoLoss()
        
        shc_mse_raw = loss(source,target) + F.mse_loss(source,target)
        shc_mse = loss(source_norm,target_norm) + F.mse_loss(source_norm,target_norm)
        # shc_mse_raw = F.mse_loss(source, target)
        # shc_mse = F.mse_loss(source_norm, target_norm)

        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }


class UniteEquPointAR_V2(EquPointConvModelV2, BaseTrainer):
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def calculate_train_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale

        loss=PanoLoss()
        
        shc_mse_raw = loss(source,target) + F.mse_loss(source,target)
        shc_mse = loss(source_norm,target_norm) + F.mse_loss(source_norm,target_norm)

        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }

    def calculate_valid_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale

        loss=PanoLoss()
        
        shc_mse_raw = loss(source,target) + F.mse_loss(source,target)
        shc_mse = loss(source_norm,target_norm) + F.mse_loss(source_norm,target_norm)
        # shc_mse_raw = F.mse_loss(source, target)
        # shc_mse = F.mse_loss(source_norm, target_norm)

        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }




class UniteEquPointAR_V3(EquPointConvModelV3, BaseTrainer):
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def calculate_train_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale
        
        loss=PanoLoss()
        
        shc_mse_raw = loss(source,target) + F.mse_loss(source,target)
        shc_mse = loss(source_norm,target_norm) + F.mse_loss(source_norm,target_norm)

        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }

    def calculate_valid_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale

        loss=PanoLoss()
        
        shc_mse_raw = loss(source,target) + F.mse_loss(source,target)
        shc_mse = loss(source_norm,target_norm) + F.mse_loss(source_norm,target_norm)
        # shc_mse_raw = F.mse_loss(source, target)
        # shc_mse = F.mse_loss(source_norm, target_norm)

        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }



class PanoEquPointAR_V1(EquPointConvModelV1, BaseTrainer):
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def calculate_train_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale
        
        loss=PanoLoss()
        
        shc_mse_raw = loss(source,target) 
        shc_mse = loss(source_norm,target_norm) 

        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }

    def calculate_valid_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale

        loss=PanoLoss()
        
        shc_mse_raw = loss(source,target) 
        shc_mse = loss(source_norm,target_norm) 
        # shc_mse_raw = F.mse_loss(source, target)
        # shc_mse = F.mse_loss(source_norm, target_norm)

        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }


class PanoEquPointAR_V2(EquPointConvModelV2, BaseTrainer):
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def calculate_train_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale

        loss=PanoLoss()
        
        shc_mse_raw = loss(source,target) 
        shc_mse = loss(source_norm,target_norm) 

        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }

    def calculate_valid_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale

        loss=PanoLoss()
        
        shc_mse_raw = loss(source,target) 
        shc_mse = loss(source_norm,target_norm) 
        # shc_mse_raw = F.mse_loss(source, target)
        # shc_mse = F.mse_loss(source_norm, target_norm)

        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }




class PanoEquPointAR_V3(EquPointConvModelV3, BaseTrainer):
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def calculate_train_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale
        
        loss=PanoLoss()
        
        shc_mse_raw = loss(source,target) 
        shc_mse = loss(source_norm,target_norm) 

        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }

    def calculate_valid_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale

        loss=PanoLoss()
        
        shc_mse_raw = loss(source,target) 
        shc_mse = loss(source_norm,target_norm) 
        # shc_mse_raw = F.mse_loss(source, target)
        # shc_mse = F.mse_loss(source_norm, target_norm)

        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }






class SHCEquPointAR_V1(EquPointConvModelV1, BaseTrainer):
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def calculate_train_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale
        
        loss=PanoLoss()
        
        shc_mse_raw =  F.mse_loss(source,target)
        shc_mse =  F.mse_loss(source_norm,target_norm)

        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }

    def calculate_valid_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale

        loss=PanoLoss()
        
        shc_mse_raw =  F.mse_loss(source,target)
        shc_mse =  F.mse_loss(source_norm,target_norm)
        # shc_mse_raw = F.mse_loss(source, target)
        # shc_mse = F.mse_loss(source_norm, target_norm)

        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }


class SHCEquPointAR_V2(EquPointConvModelV2, BaseTrainer):
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def calculate_train_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale

        loss=PanoLoss()
        
        shc_mse_raw =  F.mse_loss(source,target)
        shc_mse =  F.mse_loss(source_norm,target_norm)

        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }

    def calculate_valid_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale

        loss=PanoLoss()
        
        shc_mse_raw =  F.mse_loss(source,target)
        shc_mse =  F.mse_loss(source_norm,target_norm)
        # shc_mse_raw = F.mse_loss(source, target)
        # shc_mse = F.mse_loss(source_norm, target_norm)

        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }




class SHCEquPointAR_V3(EquPointConvModelV3, BaseTrainer):
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def calculate_train_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale
        
        loss=PanoLoss()
        
        shc_mse_raw =  F.mse_loss(source,target)
        shc_mse =  F.mse_loss(source_norm,target_norm)

        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }

    def calculate_valid_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale

        loss=PanoLoss()
        
        shc_mse_raw = loss(source,target) + F.mse_loss(source,target)
        shc_mse = loss(source_norm,target_norm) + F.mse_loss(source_norm,target_norm)
        # shc_mse_raw = F.mse_loss(source, target)
        # shc_mse = F.mse_loss(source_norm, target_norm)

        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }
class UniteWeightEquPointAR_V1(EquPointConvModelV1, BaseTrainer):
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def calculate_train_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale
        
        loss=PanoLoss()
        
        shc_mse_raw = self.hparams['alpha'] * loss(source,target) + self.hparams['beta'] * F.mse_loss(source,target)
        shc_mse = self.hparams['alpha'] * loss(source_norm,target_norm) + self.hparams['beta'] *F.mse_loss(source_norm,target_norm)

        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }

    def calculate_valid_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale

        loss=PanoLoss()
        
        # shc_mse_raw = loss(source,target) + F.mse_loss(source,target)
        # shc_mse = loss(source_norm,target_norm) + F.mse_loss(source_norm,target_norm)
        shc_mse_raw = self.hparams['alpha'] * loss(source,target) + self.hparams['beta'] * F.mse_loss(source,target)
        shc_mse = self.hparams['alpha'] * loss(source_norm,target_norm) + self.hparams['beta'] *F.mse_loss(source_norm,target_norm)

        # shc_mse_raw = F.mse_loss(source, target)
        # shc_mse = F.mse_loss(source_norm, target_norm)

        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }


class UniteWeightEquPointAR_V2(EquPointConvModelV2, BaseTrainer):
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def calculate_train_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale

        loss=PanoLoss()
        
        shc_mse_raw = self.hparams['alpha'] * loss(source,target) + self.hparams['beta'] * F.mse_loss(source,target)
        shc_mse = self.hparams['alpha'] * loss(source_norm,target_norm) + self.hparams['beta'] *F.mse_loss(source_norm,target_norm)

        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }

    def calculate_valid_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale

        loss=PanoLoss()
        
        shc_mse_raw = self.hparams['alpha'] * loss(source,target) + self.hparams['beta'] * F.mse_loss(source,target)
        shc_mse = self.hparams['alpha'] * loss(source_norm,target_norm) + self.hparams['beta'] *F.mse_loss(source_norm,target_norm)
         # shc_mse_raw = F.mse_loss(source, target)
        # shc_mse = F.mse_loss(source_norm, target_norm)

        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }




class UniteWeightEquPointAR_V3(EquPointConvModelV3, BaseTrainer):
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def calculate_train_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale
        
        loss=PanoLoss()
        
        shc_mse_raw = self.hparams['alpha'] * loss(source,target) + self.hparams['beta'] * F.mse_loss(source,target)
        shc_mse = self.hparams['alpha'] * loss(source_norm,target_norm) + self.hparams['beta'] *F.mse_loss(source_norm,target_norm)

        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }

    def calculate_valid_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale

        loss=PanoLoss()
        
        shc_mse_raw = self.hparams['alpha'] * loss(source,target) + self.hparams['beta'] * F.mse_loss(source,target)
        shc_mse = self.hparams['alpha'] * loss(source_norm,target_norm) + self.hparams['beta'] *F.mse_loss(source_norm,target_norm)
        # shc_mse_raw = F.mse_loss(source, target)
        # shc_mse = F.mse_loss(source_norm, target_norm)

        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }



