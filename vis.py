import os
import numpy as np
import math

import torch
import argparse
import matplotlib.pyplot as plt
from model.pointconv import EPPointAR  as PointAR
from evaluation.utils import SphericalHarmonics
from datasets.pointar.loader import PointARTestDataset
from IPython import embed


parser = argparse.ArgumentParser(description='visualization tool')
parser.add_argument('--use_hdr',default=False,action='store_true')
parser.add_argument('--epoch',default=0,type=int)
parser.add_argument('--model_type',default='model_dumps',type=str)
parser.add_argument('--model',default=None,type=str)
parser.add_argument('--maxn',default=30,type=int,help='max iteration is 743')
parser.add_argument('--augment',default=False,action='store_true')
parser.add_argument('--epn',default='v1',type=str,help='concrete model')
parser.add_argument('--dataset',default='test',type=str,help='the dataset type')
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

    
    
    
use_hdr = args.use_hdr
dataset = PointARTestDataset(dataset=args.dataset,use_hdr=use_hdr,augment=args.augment)
epoch = args.epoch
if args.model is not None:
    model = PointAR.load_from_checkpoint(args.model)
else:
    model = PointAR.load_from_checkpoint(f'./dist/{args.model_type}/{epoch}.ckpt')
print(args)
print(model)
model.eval()
model.cuda()
n_scale = torch.Tensor(model.hparams['scale'])
n_min = torch.Tensor(model.hparams['min'])
hdir='./viz/{}'.format(args.model_type)
if not os.path.exists(hdir):
    os.makedirs(hdir)
hhdir = os.path.join(hdir+'/{}_{}'.format(('use_hdr' if use_hdr else 'use_ldr'),('rotate' if args.augment else 'non_rotate')))
if not os.path.exists(hhdir):
    os.makedirs(hhdir)
gt =[]
pred =[]
import torch.nn.functional as F
mse = []
file = os.path.join(hhdir,'sh.npz')
mse_file = os.path.join(hhdir,'mse.npz')
if not os.path.exists(file):
    os.system(r'touch {}'.format(file))
for idx in range(args.maxn):
    from IPython import embed 
    embed()
    x, y = dataset[idx]
    xyz, rgb = x
    xyz,rgb = xyz.cuda(),rgb.cuda() 
    shc_gt = y
    gt.append(shc_gt.detach().cpu().numpy().reshape(3,-1).T.flatten())
    shc_pd = model.forward(torch.unsqueeze(xyz, dim=0), torch.unsqueeze(rgb, dim=0)).cpu()
    shc_pd = (shc_pd - n_min) / n_scale
    pred.append(shc_pd.detach().cpu().numpy().reshape(3,-1).T.flatten())
    mse.append(F.mse_loss(shc_gt.detach(),shc_pd.detach()))
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].imshow(SphericalHarmonics.from_array(shc_gt.numpy()).reconstruct_to_canvas().data)
    ax[1].imshow(SphericalHarmonics.from_array(shc_pd.detach().numpy()[0]).reconstruct_to_canvas().data)
    path = os.path.join(hhdir+'/{}_{}.png'.format(idx,epoch))
    if not os.path.exists(path):
        os.system(r'touch {}'.format(path))
    plt.savefig(path)
    plt.close()
np.savez(file,gt=np.array(gt),pred=np.array(pred))
with open(os.path.join(hhdir, 'mse.txt'), 'w') as f:
    for idx, m in enumerate(mse):
        f.write('idx: {}, mse: {}\n'.format(idx, m.item()))





