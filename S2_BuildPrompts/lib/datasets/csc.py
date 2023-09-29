import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.utils.data as data
from ..utils.transforms import fliplr_joints, crop, generate_target, transform_pixel_csc


class CSCINF(data.Dataset):
    def __init__(self, cfg, is_train=False, transform=None):
        self.data_root = cfg.DATASET.ROOT            #数据集根目录
        assert os.path.exists(self.data_root), "data root ({}) is invalid!".format(self.data_root)
        
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # 获取图像文件名表
        self.image_files = sorted(list(filter(lambda x: x.endswith('.jpg') or x.endswith('.png'), 
                                  os.listdir(self.data_root))))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        image_path = os.path.join(self.data_root, img_file)
        img_pil = Image.open(image_path).convert('RGB')
        img_pil = img_pil.resize((self.input_size[0], self.input_size[1]))
        img = np.array(img_pil, dtype=np.float32)
        img = (img/255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        meta = {'index': str(idx), 'name': str(img_file)}
        return img, meta


class CSC(data.Dataset):

    def __init__(self, cfg, is_train=True, transform=None):
        # specify annotation file for dataset
        if is_train:
            self.csv_file = cfg.DATASET.TRAINSET
        else:
            self.csv_file = cfg.DATASET.TESTSET

        self.is_train = is_train                     #是否训练模式
        self.transform = transform                   #图片预处理
        self.data_root = cfg.DATASET.ROOT            #数据集根目录
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.sigma = cfg.MODEL.SIGMA
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rot_factor = cfg.DATASET.ROT_FACTOR
        self.label_type = cfg.MODEL.TARGET_TYPE
        self.flip = cfg.DATASET.FLIP

        # load annotations
        self.landmarks_frame = pd.read_csv(self.csv_file)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):

        # 从csv中读取信息

        image_file = self.landmarks_frame.iloc[idx, 0]
        image_file_new = '/'.join(image_file.split('\\'))
        image_path = os.path.join(self.data_root, image_file_new)
                                                             
        box_x1, box_y1, box_x2, box_y2 = self.landmarks_frame.iloc[idx, 1:5]
        pts_h, pts_w = self.landmarks_frame.iloc[idx, 5:7]
        box_size = (abs(box_x2-box_x1), abs(box_y2-box_y1))
        
        pts = self.landmarks_frame.iloc[idx, 7:].values
        pts = pts.astype('float').reshape(-1,2)

        nparts = pts.shape[0]
        
        img_pil = Image.open(image_path).convert('RGB')
        img_pil = img_pil.resize((self.input_size[0], self.input_size[1]))
        img = np.array(img_pil, dtype=np.float32)
        img = (img/255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])

        # TODO : 训练时翻转、旋转等

        # 准备高斯热力图
        target = np.zeros((nparts, self.output_size[0], self.output_size[1]))
        tpts = pts.copy()
        for i in range(nparts):
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = transform_pixel_csc(tpts[i, 0:2]+1, (pts_h,pts_w), 
                                                   self.output_size)
                target[i] = generate_target(target[i], tpts[i]-1,self.sigma,
                                            label_type=self.label_type)

        target = torch.Tensor(target)
        tpts = torch.Tensor(tpts)

        meta = {'index': idx, 'pts': torch.Tensor(pts), 'tpts': tpts, 'box_size': box_size}

        return img, target, meta


if __name__ == '__main__':
    pass
