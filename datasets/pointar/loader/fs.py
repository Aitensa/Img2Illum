"""Data loader
"""

import h5py
import configs
import numpy as np
import math
from torch import from_numpy
from torch.utils.data import Dataset


def euler_rotation(vertices, angels=(0., 0., 0.)):
    # Euler XYZ rotation matrix

    rx, ry, rz = angels

    R_z = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ]).astype(np.float32)

    R_y = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ]).astype(np.float32)

    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ]).astype(np.float32)

    R = R_z @ R_y @ R_x
    vertices_rotated = np.dot(R, vertices)

    return vertices_rotated


class BaseDataset(Dataset):
    dataset_name: str
    arr_target: np.ndarray
    arr_indices: np.ndarray
    data_root=f'{configs.pointar_dataset_path}/package'
    def __init__(self, use_hdr=False, augment=False, dataset_path=None):
        super(BaseDataset, self).__init__()
        
        hdr_mark = 'hdr' if use_hdr else 'ldr'
        print ("hdr_mark: ", hdr_mark)
        self.augment =augment
        self.arr_target = np.moveaxis(self.arr_target[hdr_mark], 1, -1)
        #print ("self.arr_target: ", self.arr_target)
        self.arr_indices = np.arange(len(self.arr_target), dtype=np.int)
        #self.arr_indices = np.arange(10, dtype=np.int)
        #print("self.arr_indices: ", self.arr_indices)

    def __getitem__(self, idx):
        point_cloud = self.__load_source__(idx)
        target_shc = self.__load_target__(idx)

        point_cloud = np.moveaxis(point_cloud, 0, -1)

        xyz = point_cloud[:3, :]
        rgb = point_cloud[3:6, :]
        if self.augment and np.random.rand()<0.5:
            rx,ry,rz=math.radians(np.random.rand()*180),math.radians(np.random.rand()*180),math.radians(np.random.rand()*180)
            xyz=euler_rotation(xyz,(rx,ry,rz))
            rgb=euler_rotation(rgb,(rx,ry,rz))
            
        # Original SH coefficients data is channel last
        # change to channel first as PyTorch use it
        target_shc = target_shc.reshape((-1))
        target = from_numpy(target_shc)

        xyz = from_numpy(xyz)
        rgb = from_numpy(rgb)

        return (xyz, rgb), target

    def __len__(self):
        return len(self.arr_indices)

    def __load_source__(self, idx):
        pc = np.load(
            f'{configs.pointar_dataset_path}/' +
            f'{self.dataset_name}/{self.arr_indices[idx]}/' +
            'point_cloud.npz'
        )['point_cloud']

        idx = np.sort(np.random.choice(
            pc.shape[0], 1280, replace=False))
        #print("idx: ", idx)
        pc = pc[idx]

        return pc

    def __load_target__(self, idx):
        return self.arr_target[self.arr_indices[idx]]


class PointARTrainDataset(BaseDataset):
    def __init__(self, dataset='train',*args, **kwargs):
        self.arr_target = np.load(f'{self.data_root}/{dataset}-shc.npz') 
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset


class PointARTestDataset(BaseDataset):
    def __init__(self,dataset='test', *args, **kwargs):
        self.arr_target = np.load(f'{self.data_root}/{dataset}-shc.npz') 
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset 
        self.augment=False


