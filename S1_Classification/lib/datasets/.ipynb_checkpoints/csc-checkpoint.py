from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.utils.data as data


class CSC(data.Dataset):
    def __init__(self, cfg, transforms):
        self.data_root = cfg.DATA_DIR            #数据目录
        assert os.path.exists(self.data_root), "data root ({}) is invalid!".format(self.data_root)
        self.transforms = transforms

        self.image_files = sorted(list(filter(lambda x: x.endswith('.jpg') or x.endswith('.png'), 
                                  os.listdir(self.data_root))))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        img_file = self.image_files[index]
        img_path = os.path.join(self.data_root, img_file)
        img_pil = Image.open(img_path).convert('RGB')
        if self.transforms:
            img = self.transforms(img_pil)
        return img, img_file


if __name__ == '__main__':
    pass
