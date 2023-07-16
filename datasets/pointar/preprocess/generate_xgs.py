"""Data generation
PointAR dataset generation code with CUDA GPU acceleration
"""

#import pdb
import zipfile
import io
import os
import json
import math
import imageio
import importlib
import numpy as np
from functools import lru_cache

from tqdm import tqdm
from multiprocessing import Pool

import pycuda.driver as drv
from pycuda.compiler import SourceModule


from utils import srgb_to_linear
from utils import load_rgbd_as_tensor
from utils import get_batched_basis_at

from utils import cube_uv_to_xyz
from utils import euler_rotation_xyz
from utils import cartesian_to_spherical
from utils import get_euler_rotation_matrix


# Allow cuda execution
importlib.import_module('pycuda.autoinit')

DS_RATE = 4  # Set DS_RATE to downsample input images before other operations for faster execution
DEBUG = True  # Enabling DEBUG will save intermediate results


module = SourceModule(
    open('./cuda/preprocess.cu', 'r').read()
)

make_point_cloud = module.get_function("makePointCloud")
camera_adjustment = module.get_function("cameraAdjustment")
make_sh_coefficients = module.get_function("makeSHCoefficients")



@lru_cache(maxsize=3)
def get_cube_idx(width, height):
    # Make UV grid to generate a cubemap
    cubemap_resolution = 128
    sample_u = np.arange(cubemap_resolution) / (cubemap_resolution - 1)
    sample_v = np.arange(cubemap_resolution) / (cubemap_resolution - 1)
    sample_uv = np.stack(np.meshgrid(sample_u, sample_v),
                         axis=-1).astype(np.float32)
    sample_uv = sample_uv.reshape((-1, 2))

    cubemap_xyz = np.empty(
        (6, cubemap_resolution * cubemap_resolution, 3),
        dtype=np.float32)

    for i in range(6):
        cubemap_xyz[i] = cube_uv_to_xyz(i, sample_uv)

    cubemap_xyz_flt = cubemap_xyz.reshape((-1, 3))
    cubemap_sph = cartesian_to_spherical(cubemap_xyz_flt)

    # Rotate XYZ for correcting orientation
    cubemap_xyz_flt = euler_rotation_xyz(
        cubemap_xyz_flt,
        (math.radians(-90), math.radians(0), 0))
    cubemap_xyz_flt = euler_rotation_xyz(
        cubemap_xyz_flt,
        (math.radians(0), math.radians(90), 0))

    cubemap_sph_tmp = cubemap_sph[:, :2]  # select theta and phi
    cubemap_sph_tmp = cubemap_sph_tmp + np.array([0, math.pi])
    cubemap_sph_tmp = cubemap_sph_tmp / np.array([math.pi, math.pi * 2])
    cubemap_sph_tmp = cubemap_sph_tmp * np.array([height - 1, width - 1])
    cubemap_sph_tmp = cubemap_sph_tmp.astype(np.int)

    idx = np.arange(cubemap_sph_tmp.shape[0], dtype=np.int)
    cubemap_idx = cubemap_sph_tmp[:, 0] * width + cubemap_sph_tmp[:, 1]

    cubemap_weight = cubemap_xyz_flt.reshape((6, -1, 3))
    cubemap_tmp = np.sum(cubemap_xyz_flt * cubemap_xyz_flt,
                         axis=-1)[:, np.newaxis]
    cubemap_weight = 4 / (np.sqrt(cubemap_tmp) * cubemap_tmp)
    cubemap_weight = cubemap_weight.astype(np.float32)

    cubemap_norm = cubemap_xyz_flt / \
        np.linalg.norm(cubemap_xyz_flt, axis=-1)[:, np.newaxis]
    cubemap_basis = get_batched_basis_at(cubemap_norm)
    cubemap_basis = cubemap_basis.astype(np.float32)

    shc_norm = (4 * math.pi) / np.sum(cubemap_weight)
    shc_norm = shc_norm.astype(np.float64)

    return cubemap_xyz_flt, idx, cubemap_idx, cubemap_weight, cubemap_basis, shc_norm


def process_output_gpu():
    # Fetch LDR Image data
    #f_zip = zipfile.ZipFile(f'/data1/guoshu/illummaps/illummaps_2t7WUuJeko7.zip')
    #ill_map_ldr = imageio.imread(io.BytesIO(f_zip.read(f'illummaps/floor/0_i.png'))) / 255

    ill_map_ldr = imageio.imread(f'0_i.png') / 255

    # Make cubemap canvas for LDR and HDR
    cubemap_xyz_flt, idx, cubemap_idx, cubemap_weight, cubemap_basis, shc_norm = get_cube_idx(
        ill_map_ldr.shape[1], ill_map_ldr.shape[0])
    cubemap_len = cubemap_xyz_flt.shape[0]

    ill_map_ldr_2d = ill_map_ldr.reshape((-1, 3))

    cubemap_color_ldr = np.empty((cubemap_len, 3), dtype=np.float32)

    cubemap_color_ldr[idx, :] = ill_map_ldr_2d[cubemap_idx, :]

    # LDR Image need to convert to linear color space
    srgb_to_linear(cubemap_color_ldr)

    #pdb.set_trace()
    # Calculate the SH coefficients
    cubemap_clr_ldr = cubemap_color_ldr * cubemap_weight

    cubemap_clr_ldr = cubemap_clr_ldr.astype(np.float32)

    len_pixels = cubemap_len // 6
    shc_ldr = np.zeros((9, 3), dtype=np.float64)

    #imageio.imwrite('cubemap.png', cubemap_color_ldr * 255)

    make_sh_coefficients(
        drv.InOut(shc_ldr),
        drv.In(cubemap_basis),
        drv.In(cubemap_clr_ldr),
        grid=(6, (len_pixels + 1024 - 1) // 1024, 1),
        block=(1, 1024, 1))

    # normalize
    #shc_ldr = (shc_ldr * shc_norm).reshape(-1).astype(np.float32)
    shc_ldr = (shc_ldr * shc_norm).astype(np.float32)

    print(shc_ldr)

process_output_gpu()
