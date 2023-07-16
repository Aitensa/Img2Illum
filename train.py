#!/usr/bin/env python
""" PointAR training script
"""

import os
import fire
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchsummary import summary
from model.pointconv import PointAR
from trainer.utils import train_valid_test_split

from datasets.pointar.loader import PointARTestDataset
from datasets.pointar.loader import PointARTrainDataset
# from datasets.pointar.loader import PointARTrainD10Dataset
import argparse 



parser = argparse.ArgumentParser(description='visualization tool')
parser.add_argument('--epn',default='u_v3',type=str,help='concrete model')
args = parser.parse_args()

if args.epn=='u_v3':
    from model.pointconv import UniteEquPointAR_V3 as PointAR 
elif args.epn=='u_v2':
    from model.pointconv import UniteEquPointAR_V2 as PointAR 
elif args.epn =='u_p':
    from model.pointconv import UnitePointAR as PointAR
elif args.epn =='p_p':
    from model.pointconv import PanoPointAR as PointAR
elif args.epn == 'p_v2':
    from model.pointconv import PanoEquPointAR_V2 as PointAR 
elif args.epn == 'p_v3':
    from model.pointconv import PanoEquPointAR_V3 as PointAR 
elif args.epn == 's_v2':
    from model.pointconv import SHCEquPointAR_V2 as PointAR 
elif args.epn == 's_v3':
    from model.pointconv import SHCEquPointAR_V3 as PointAR
elif args.epn == 's_p':
    from model.pointconv import SHCPointAR as PointAR

class ModelSavingCallback(pl.Callback):
    def __init__(self, sample_input):
        self.sample_input = sample_input

    def on_epoch_end(self, trainer, pl_module):
        dump_path = f'./dist/model_dumps'
        os.system(f'mkdir -p {dump_path}')

        trainer.save_checkpoint(f'{dump_path}/{trainer.current_epoch}.ckpt')


def train(debug=False,
          use_hdr=False,
          normalize=True,
          n_points=1280,
          num_workers=16,
          batch_size=32):
    """Train PointAR model

    Parameters
    ----------
    debug : bool
        Set debugging flag
    use_hdr : bool
        Use HDR SH coefficients data for training
    normalize : bool
        Normalize SH coefficients
    n_points : int
        Number of model input points, default 1280
    num_workers : int
        Number of workers for loading data, default 16
    batch_size : int
        Training batch size
    """

    # Specify dataset
    TestDataset = PointARTestDataset
    TrainDataset = TestDataset if debug else PointARTrainDataset

    # Get loaders ready
    loader_param = {'use_hdr': use_hdr}
    loaders, scaler = train_valid_test_split(
        TrainDataset, loader_param,
        TestDataset, loader_param,
        normalize=normalize, num_workers=num_workers, batch_size=batch_size)

    train_loader, valid_loader, test_loader = loaders

    # Get model ready
    ''' 
    model = PointAR(hparams={
        'n_shc': 27,
        'n_points': n_points,
        'min': torch.from_numpy(scaler.min_) if normalize else torch.zeros((27)),
        'scale': torch.from_numpy(scaler.scale_) if normalize else torch.ones((27))
    })
    '''
    model = PointAR.load_from_checkpoint(f'./dist/model_dumps_ldr/3.ckpt')

    # Train
    sample_input = (
        torch.zeros((1, 3, n_points)).float().cuda(),
        torch.zeros((1, 3, n_points)).float().cuda())
    print(model)
   
    trainer = pl.Trainer(
        gpus=3,
        check_val_every_n_epoch=1,
        callbacks=[
            ModelSavingCallback(
                sample_input=sample_input
            ),
            EarlyStopping(monitor='valid_shc_mse')
        ])

    # Start training
    trainer.fit(
        model,
        train_dataloader=train_loader,
        val_dataloaders=[valid_loader, test_loader])


if __name__ == '__main__':
    fire.Fire(train)
