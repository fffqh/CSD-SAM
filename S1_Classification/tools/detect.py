from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import csv
import shutil
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import _init_paths
import models
from config import config
from config import update_config
from core.function import detect
from utils.modelsummary import get_model_summary
from utils.utils import create_logger
from datasets.csc import CSC

def parse_args():
    parser = argparse.ArgumentParser(description='Detect KeyFrame network')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        required=True,
                        default='')
    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        required=True,
                        default='')
    args = parser.parse_args()
    update_config(config, args)
    return args

def main():
    args = parse_args()
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'detect')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))
    
    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.'+config.MODEL.NAME+'.get_cls_net')(
        config)

    dump_input = torch.rand(
        (1, 3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0])
    )
    logger.info(get_model_summary(model, dump_input))

    if config.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
    else:
        model_state_file = os.path.join(final_output_dir,
                                        'final_state.pth.tar')
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    gpus = list(config.GPUS)
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    detect_transforms = transforms.Compose([
            transforms.Resize(config.MODEL.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    
    detect_dataset = CSC(config, detect_transforms)
    detect_loader = torch.utils.data.DataLoader(
        detect_dataset,
        batch_size = config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    detect(detect_loader, model, final_output_dir)
    # 复制所有预测为True的flw_img
    output_csv_path = os.path.join(final_output_dir, 'output.csv')
    assert os.path.exists(output_csv_path), "output.csv do not exist!"
    true_dir = os.path.join(config.DATA_DIR, './true')
    if os.path.exists(true_dir):
        shutil.rmtree(path=true_dir)
    os.mkdir(true_dir)
    print("Create start:{}.".format(config.DATA_DIR+'./true'))
    with open(output_csv_path,'r', encoding='utf8', newline='') as output_csv_f:
        csv_headers = ['file_name','pred']
        output_csv_reader = csv.DictReader(output_csv_f, csv_headers)
        next(output_csv_reader) #跳过表头行
        for output in output_csv_reader:
            file_name = output['file_name']
            pred = output['pred']
            if pred=='True':
                src = os.path.join(config.DATA_DIR, file_name)
                des = os.path.join(true_dir, file_name)
                shutil.copyfile(src, des)
    print("Create done.")
    
    
if __name__ == '__main__':
    main()
