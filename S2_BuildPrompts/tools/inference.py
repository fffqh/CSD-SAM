import os
import sys
import pprint
import shutil
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
from lib.utils import utils
from lib.core import function
from lib.config import config, update_config
from lib.datasets import get_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='CSC inference')
    parser.add_argument('--cfg', help='inference configuration filename',
                        required=True, type=str)
    parser.add_argument('--model-file', help='model parameters', 
                        required=True, type=str)
    parser.add_argument('--input-dir', help='flow image to input',
                        required=True, type=str)
    parser.add_argument('--output-dir', help='predictions to output',
                    required=True, type=str)
    parser.add_argument('--save-image', help='save image with points',
                        action='store_true')
    args = parser.parse_args()
    assert os.path.exists(args.input_dir), "input dir is invalid : {}".format(args.input_dir)
    update_config(config, args)
    return args


def main():
    args = parse_args()

    # 设置log
    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'inference')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # 设置cuda
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # 保证不对模型进行初始化后，建立模型对象
    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.DATASET.ROOT = args.input_dir #更新数据路径
    config.freeze()
    model = models.get_face_alignment_net(config)

    # 模型数据并行化
    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # 将参数文件导入模型
    state_dict = torch.load(args.model_file)
    print("[debug] state_dict type:", type(state_dict))
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    else:
        model.module.load_state_dict(state_dict)

    
    # 建立数据读取对象
    dataset_type = get_dataset(config)
    inf_loader = DataLoader(
        dataset=dataset_type(config,
                             is_train=False),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    # 进行推理
    function.inference(config, inf_loader, model, final_output_dir, save_image=args.save_image)

    # 将结果复制到主目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    src_path = os.path.join(final_output_dir, 'predictions.csv')
    des_path = os.path.join(args.output_dir, 'prompts.csv')
    shutil.copyfile(src_path, des_path)
    print("output : {}".format(des_path))

    
if __name__ == '__main__':
    main()

