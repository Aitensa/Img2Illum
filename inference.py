import cv2
import imageio
import math
import importlib
import numpy as np
import matplotlib.pyplot as plt
# import pycuda.driver as drv
# import pycuda.autoinit
# from pycuda.compiler import SourceModule
from DepthEst import DepthEst 
from evaluation.utils import SphericalHarmonics
import torch
import os 
import time
import argparse
# import ipdb

DS_RATE = 4  # Set DS_RATE to downsample input images before other operations for faster execution
# drv.init()
# pycuda_ctx = drv.Device(0).retain_primary_context()


# Allow cuda execution
# ipdb.set_trace()
# importlib.import_module('pycuda.autoinit')
# module = SourceModule(open('./datasets/pointar/preprocess/cuda/preprocess.cu', 'r').read())

# make_point_cloud = module.get_function("makePointCloud")
# camera_adjustment = module.get_function("cameraAdjustment")
parser = argparse.ArgumentParser(description='inference tool')
parser.add_argument('--epoch',default=0,type=int)
parser.add_argument('--model_type',default='model_dumps',type=str)
parser.add_argument('--epn',default='v1',type=str,help='concrete model')
parser.add_argument('--rgb',default=None,type=str,help='the path of rgb ')
parser.add_argument('--refine',default=None,type=str,help='whether to use refine depth groud truth')
parser.add_argument('--model',default=None,type=str)
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

  
# epoch = args.epoch
# DepthPred = DepthEst(depth=None)
if args.model is not None:
    model = PointAR.load_from_checkpoint(args.model)
else:
    model = PointAR.load_from_checkpoint(f'./dist/{args.model_type}/{epoch}.ckpt')

# # from IPython import embed
# # embed()
print(model)


# from IPython import embed
# # embed()
n_scale = torch.Tensor(model.hparams['scale'])
n_min = torch.Tensor(model.hparams['min'])


def pycuda_dummy_measure_time():
    pycuda_ctx.push()
    event_start = drv.Event().record()
    pycuda_ctx.pop()

    time.sleep(2)

    pycuda_ctx.push()
    event_stop = drv.Event().record().synchronize()
    print(event_stop.time_since(event_start))
    pycuda_ctx.pop()


def get_euler_rotation_matrix(angels=(0, 0, 0)):
    rx, ry, rz = angels

    R_z = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])

    R_y = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])

    R = R_z @ R_y @ R_x
    R = R.astype(np.float32)

    return R

def ds_rgbd(rgb, refine=None):
    # #illummaps-testlist.txt 0
    # color_image = f'/data1/guoshu/Matterport/v1/scans/2t7WUuJeko7/undistorted_color_images/11f452db8a6b42f88c78be639670b423_i2_4.jpg'
    # depth_image = f'/data1/guoshu/Matterport/v1/scans/2t7WUuJeko7/undistorted_depth_images/11f452db8a6b42f88c78be639670b423_d2_4.png'

    # # According to Matterport document, we need to flip up down the image
    # org_color_img = cv2.imread(color_image)
    # org_depth_img = cv2.imread(depth_image)
    if refine!=None:
        refine = cv2.imread(refine)
    org_depth_img = DepthPred.est_single_image(rgb,refine)
    org_color_img = imageio.imread(rgb)

    img_color = np.flipud(org_color_img) / 255
    img_depth = np.flipud(org_depth_img) 

    img_color = img_color[::DS_RATE, ::DS_RATE, :]
    img_depth = img_depth[::DS_RATE, ::DS_RATE]

    img_color = img_color.astype(np.float32)
    img_depth = img_depth.astype(np.float32)

    return img_color, img_depth


def generate_pc(img_color, img_depth):
    # ./datasets/matterport3d/configs/2t7WUuJeko7.json
    #"11f452db8a6b42f88c78be639670b423_d2_4.png": {"intrinsics": {"fx": 1073.03, "fy": 1073.05, "cx": 644.239, "cy": 515.663}, "camera_to_world_matrix"
    torch.cuda.init()  # any torch cuda initialization before pycuda calls, torch.randn(10).cuda() works too

    pycuda_dummy_measure_time()  # measures time of a 2-second sleep using pycuda Events

    fx = int(1073.03 // DS_RATE)
    fy = int(1073.05 // DS_RATE)
    cx = int(644.239 // DS_RATE)
    cy = int(515.663 // DS_RATE)

    mat_ctw = np.array([0.138209, 0.657619, -0.740564, -0.00874733, \
            -0.990398, 0.093893, -0.101458, 0.0453511, \
            0.00281372, 0.747477, 0.664281, 1.68927, \
            0.0, 0.0, 0.0, 1.0], \
            dtype=np.float32)
    mat_ctw = mat_ctw.reshape((4, 4))
    mat_rotation = get_euler_rotation_matrix((math.radians(-90), 0, 0))

    width = img_color.shape[1]
    height = img_color.shape[0]

    intrinsics = np.array(
        [fx, fy, cx, cy, width, height],
        dtype=np.float32)

    # Generate Point Cloud
    xyz_camera_space = np.empty_like(img_color, dtype=np.float32)
    a= torch.cuda.ByteTensor(32,32,1)
    make_point_cloud(
        drv.Out(xyz_camera_space),
        drv.In(img_depth),
        drv.In(intrinsics),
        grid=(
            (width + 32 - 1) // 32,
            (height + 32 - 1) // 32,
            1),
        block=(32, 32, 1)
    )

    xyz_camera_space = np.array(xyz_camera_space, dtype=np.float32)
    xyz_world_space = xyz_camera_space.reshape((-1, 3))

    camera_adjustment(
        drv.InOut(xyz_world_space),
        drv.In(mat_ctw),
        drv.In(mat_rotation),
        drv.In(intrinsics),
        grid=(
            (width + 32 - 1) // 32,
            (height + 32 - 1) // 32,
            1),
        block=(32, 32, 1)
    )

    # illummaps-testlist.txt
    ix = int(float(1012.93) // DS_RATE)
    iy = int(float(656.568) // DS_RATE)

    # Prepare to save results
    #ray_dirs = xyz_camera_space.reshape((-1, 3))
    #ray_dirs_norm = np.linalg.norm(ray_dirs, axis=-1)
    #ray_dirs_norm[ray_dirs_norm == 0] = 1
    #ray_dirs = ray_dirs / ray_dirs_norm[:, np.newaxis]

    idx = iy * width + ix
    xyz_shifted = xyz_world_space - xyz_world_space[idx]

    img_color = img_color.reshape((-1, 3))

    #pc = np.concatenate((xyz_shifted, img_color, ray_dirs), axis=-1)
    #pc = pc.reshape((-1, 9)).astype(np.float32)
    pc = np.concatenate((xyz_shifted, img_color), axis=-1)
    pc = pc.reshape((-1, 6)).astype(np.float32)
    pc = pc[:, :6]

    # This point cloud has ray direction
    # np.savez_compressed(
    #     f'./inference/point_cloud',
    #     point_cloud=pc
    # )

    idx = np.sort(np.random.choice(pc.shape[0], 1280, replace=False))
    
    pc = pc[idx]
    np.savez_compressed(
        f'./inference/point_cloud',
        point_cloud=pc
    )
    from IPython import embed
    embed()
    # np.savetxt('./point_cloud_1280.txt', pc)
    return pc


def load_pc():
    a = np.load(f'./inference/point_cloud.npz')
    # b = np.load(f'./inference/b/point_cloud.npz')
    # np.savetxt('./inference/point_cloud_a.txt', a['point_cloud'])
    # np.savetxt('./inference/point_cloud_b.txt', b['point_cloud'])
    return a['point_cloud']
    

def predict(pc):
    model.eval()
    model.cuda()
    xyz = torch.unsqueeze(torch.tensor(pc[:, :3].T),dim=0)#(1, 3, 1280)
    rgb = torch.unsqueeze(torch.tensor(pc[:, 3:6].T),dim=0)
    xyz = (xyz - xyz.min())/400
    xyz = xyz.to(model.device)
    rgb = rgb.to(model.device)

    # Inference
    t1 = time.time()
    # from IPython import embed
    # embed()
    p = model.forward(xyz, rgb).detach().cpu()
    t2 = time.time()
    #torch.onnx.export(model, (xyz,rgb), 'model.onnx')
    # p = (p - n_min) / n_scale
    p = p.numpy()
    p1 = p.reshape(3,9).T
    coefficients = p1.reshape((-1))
    
    plt.imshow(SphericalHarmonics.from_array(coefficients).reconstruct_to_canvas().data)
    path = os.path.join('./inference/{}.png'.format(args.rgb.split('/')[-1].split('.')[0]))
    if not os.path.exists(path):
        os.system(r'touch {}'.format(path))
    plt.savefig(path)
    plt.close()
    print("shc_pd: ", coefficients)
    print("time: ", t2-t1)

    # l2 loss
    # loss_ldr =np.sum(np.power((coefficients - ldr_sh_coefficients), 2))
    # print("loss_ldr: ", loss_ldr)



def main():
    # img_color, img_depth = ds_rgbd(args.rgb,args.refine)
    # pc = generate_pc(img_color, img_depth)
    print("here")
    pc = load_pc()
    predict(pc)


if __name__ == '__main__':
    main()







