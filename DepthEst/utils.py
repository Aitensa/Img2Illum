import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model, datasets

from logging import root
import torch 
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.multiprocessing as mp 

from DepthEst.lib.test_utils import refine_focal, refine_shift
from DepthEst.lib.multi_depth_model_woauxi import RelDepthModel
from DepthEst.lib.net_tools import load_ckpt
from DepthEst.lib.spvcnn_classsification import SPVCNN_CLASSIFICATION
from DepthEst.lib.test_utils import reconstruct_depth, recover_metric_depth
import os
import importlib 
from collections import OrderedDict

class RefineScale():
    def __init__(self, depth=None):
        super(RefineScale, self).__init__()
        self.param_a = 3000
        self.param_b = 0 
        self.depth=None
    def fit(self,pred_depth, gt_depth=None):
        
        depth1 = pred_depth.squeeze()
        depth2 = gt_depth.squeeze() if gt_depth!=None else None
        if type(depth2).__module__!=np.__name__:
            if type(self.depth).__module__!=np.__name__:
                return pred_depth/pred_depth.max() * self.param_a + self.param_b
            else:
                depth2 = self.depth
        # print("predicted depth.shape: ", depth1.shape)
        # print("raw sensor depth.shape: ", depth2.shape)
        ### sample 16 points uniformly.
        sample_w = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19]) * depth1.shape[1]//20
        sample_h = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19]) * depth1.shape[0]//20
        X = np.array([])
        y = np.array([])
        mask = (depth1 > 1e-8) & (depth2 > 1e-8)
        X = depth1[mask]
        y = depth2[mask]
        # for h in depth1.shape:
        #     for w in sample_w:
        #         if (depth2[h, w] > 1e-6 and depth1[h, w]> 1e-6):
        #             X = np.append(X, depth1[h, w])
        #             y = np.append(y, depth2[h, w])
        # X = np.reshape(X, (-1, 1))
        ### Use Ransac to get the linear mapping.
        # # Fit line using all data
        # lr = linear_model.LinearRegression()
        # lr.fit(X, y)
        
        try:
            # Robustly fit linear model with RANSAC algorithm
            ransac = linear_model.RANSACRegressor(max_trials=1000)
            ransac.fit(X, y)
            # inlier_mask = ransac.inlier_mask_
            # outlier_mask = np.logical_not(inlier_mask)


            # # Predict data of estimated models
            # line_X = np.arange(X.min(), X.max())[:, np.newaxis]
            # # line_y = lr.predict(line_X)
            # line_y_ransac = ransac.predict(line_X)

            # Compare estimated coefficients
            # print("Estimated coefficients (linear regression, RANSAC):")
            # print(lr.coef_, ransac.estimator_.coef_)
            # print("lr: ", lr.coef_, lr.intercept_)
            # print("ransac: ", ransac.estimator_.coef_,  ransac.estimator_.intercept_)
            ###Evaluation
            disp = np.empty_like(depth1,dtype=np.float32)
            disp = depth1 * ransac.estimator_.coef_ + ransac.estimator_.intercept_
            if self.param_a==3000 and self.param_b==0:
                return disp 
            Loss_new = abs((disp-depth2)).sum()
            disp2 = depth1 * self.param_a + self.param_b 
            Loss_raw = abs((disp2- depth2)).sum()
            if Loss_new < Loss_raw:
                self.param_a = ransac.estimatior._coef 
                self.param_b = ransac.estimator.intercept_
                return disp
            else:
                
                return disp2
        except ValueError as e:
            # print(e.message)
            return depth1 * self.param_a + self.param_b



class DepthEst():
    def __init__(self,**args):
        super(DepthEst,self).__init__()
        self.refine = RefineScale(args['depth'])
        self.shift_model = SPVCNN_CLASSIFICATION(input_channel=3,
                                        num_classes=1,
                                        cr=1.0,
                                        pres=0.01,
                                        vres=0.01
                                        ).eval()
        self.focal_model = SPVCNN_CLASSIFICATION(input_channel=5,
                                        num_classes=1,
                                        cr=1.0,
                                        pres=0.01,
                                        vres=0.01
                                        ).eval()
        self.depth_model = RelDepthModel(backbone='resnext101').eval()
        self.load_ckpt('./res101.pth')
        self.to_cuda()
    def scale_torch(self, img):
        """
        Scale the image and output it in torch.tensor.
        :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
        :param scale: the scale factor. float
        :return: img. [C, H, W]
        """
        if len(img.shape) == 2:
            img = img[np.newaxis, :, :]
        if img.shape[2] == 3:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225) )])
            img = transform(img)
        else:
            img = img.astype(np.float32)
            img = torch.from_numpy(img)
        return img
    
    def reconstruct3D_from_depth(self, rgb, pred_depth):
        cam_u0 = rgb.shape[1] / 2.0
        cam_v0 = rgb.shape[0] / 2.0
        pred_depth_norm = pred_depth - pred_depth.min() + 0.5

        dmax = np.percentile(pred_depth_norm, 98)
        pred_depth_norm = pred_depth_norm / dmax

        # proposed focal length, FOV is 60', Note that 60~80' are acceptable.
        proposed_scaled_focal = (rgb.shape[0] // 2 / np.tan((60/2.0)*np.pi/180))

        # recover focal
        focal_scale_1 = refine_focal(pred_depth_norm, proposed_scaled_focal, self.focal_model, u0=cam_u0, v0=cam_v0)
        predicted_focal_1 = proposed_scaled_focal / focal_scale_1.item()

        # recover shift
        shift_1 = refine_shift(pred_depth_norm, self.shift_model, self.predicted_focal_1, cam_u0, cam_v0)
        shift_1 = shift_1 if shift_1.item() < 0.6 else torch.tensor([0.6])
        depth_scale_1 = pred_depth_norm - shift_1.item()

        # recover focal
        focal_scale_2 = refine_focal(depth_scale_1, predicted_focal_1, self.focal_model, u0=cam_u0, v0=cam_v0)
        predicted_focal_2 = predicted_focal_1 / focal_scale_2.item()

        return shift_1, predicted_focal_2, depth_scale_1
    
    def est_single_image(self, rgb, base=None):
        if os.path.isfile(rgb):
            rgb = cv2.imread(rgb)
        rgb_c = rgb[:,:,::-1].copy()
        A_resize = cv2.resize(rgb_c,(448,448))
        rgb_half = cv2.resize(rgb,(rgb.shape[1]//2,rgb.shape[0]//2),interpolation=cv2.INTER_LINEAR)
        img_torch = self.scale_torch(A_resize)[None, :, :, :]
        pred_depth = self.depth_model.inference(img_torch).cpu().numpy().squeeze()
        pred_depth_ori = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))
        
        # # recover focal length, shift, and scale-invariant depth
        # shift, focal_length, depth_scaleinv = reconstruct3D_from_depth(rgb, pred_depth_ori)
        # disp = 1 / depth_scaleinv
        # disp = self.refine.fit(disp,base)
        # pred_depth_ori = self.refine.fit(pred_depth_ori)
        

        # if GT depth is available, uncomment the following part to recover the metric depth
        # if type(base).__module__ == np.__name__:
            
        #     pred_depth_ori = recover_metric_depth(pred_depth_ori/pred_depth_ori.max() * 60000, base[:,:,0].copy())
        # else:
        #     pred_depth_ori = (pred_depth_ori/pred_depth_ori.max() * 60000).astype(np.uint16)
        pred_depth_ori = self.refine.fit(pred_depth_ori, base[:,:,0] if base != None else None).astype(np.uint16)

        return pred_depth_ori

    
    def end(self):
        self.to_cpu()
    def load_ckpt(self, load_ckpt):
        """
        Load checkpoint.
        """
        if os.path.isfile(load_ckpt):
            print("loading checkpoint %s" % load_ckpt)
            checkpoint = torch.load(load_ckpt)
            if self.shift_model is not None:
                self.shift_model.load_state_dict(self.strip_prefix_if_present(checkpoint['shift_model'], 'module.'),
                                        strict=True)
            if self.focal_model is not None:
                self.focal_model.load_state_dict(self.strip_prefix_if_present(checkpoint['focal_model'], 'module.'),
                                        strict=True)
            self.depth_model.load_state_dict(self.strip_prefix_if_present(checkpoint['depth_model'], "module."),
                                        strict=True)
            del checkpoint
            torch.cuda.empty_cache()
    def strip_prefix_if_present(self, state_dict, prefix):
        keys = sorted(state_dict.keys())
        if not all(key.startswith(prefix) for key in keys):
            return state_dict
        stripped_state_dict = OrderedDict()
        for key, value in state_dict.items():
            stripped_state_dict[key.replace(prefix, "")] = value
        return stripped_state_dict
    def to_cuda(self):
        self.shift_model.cuda()
        self.focal_model.cuda()
        self.depth_model.cuda()
    def to_cpu(self):
        self.shift_model.cpu()
        self.focal_model.cpu()
        self.depth_model.cpu()
        torch.cuda.empty_cache()
    def refine_depth(self, pred, gt):
        if type(pred).__module__ == torch.__name__:
            pred = pred.cpu().numpy()
        if type(gt).__module__ == torch.__name__:
            gt = gt.cpu().numpy()
        gt = gt.squeeze()
        pred = pred.squeeze()
        mask = (gt > 1e-8) & (pred > 1e-8)

        gt_mask = gt[mask]
        pred_mask = pred[mask]
        coef = (pred_mask/gt_mask).mean()
        # a, b = np.polyfit(pred_mask, gt_mask, deg=1)
        pred_metric = pred * coef 
        return pred_metric










