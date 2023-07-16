import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='MSE Calculator')
parser.add_argument('--path', default="", type=str)
args = parser.parse_args()

def calculate_mse(npz_path):
    hhdir = os.path.dirname(npz_path)
    
    npz = np.load(npz_path,allow_pickle=True)
    gt = npz['gt']
    pred = npz['pred']
    mse = np.mean((gt - pred) ** 2, axis=1)
    
    with open(os.path.join(hhdir, 'mse.txt'), 'w') as f:
        for idx, m in enumerate(mse):
            f.write('idx: {}, mse: {}\n'.format(idx, m.item()))

def traverse_folders(path):
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            npz_path = os.path.join(root, dir, 'sh.npz')
            
            if os.path.isfile(npz_path):
                calculate_mse(npz_path)
            else:
                print("No 'sh.npz' file found in", os.path.join(root, dir))

traverse_folders(args.path)

