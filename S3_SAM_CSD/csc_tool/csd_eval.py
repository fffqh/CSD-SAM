import os
import cv2
import csv
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from lof import outliers

_NO_LOF = False
_NO_LA = False
_NO_PA = False

def parse_args():
    parser = argparse.ArgumentParser(description='CSD Evaluation.')
    parser.add_argument('--csv-path',
                        help='path of csv file',
                        required=True,
                        type=str)
    parser.add_argument('--ori-root',
                        help='path of original images',
                        required=True,
                        type=str)
    parser.add_argument('--flw-root',
                        help='path of flow images',
                        required=True,
                        type=str)
    parser.add_argument('--mgt-root',
                        help='path of mask gound turth',
                        required=True,
                        type=str)
    parser.add_argument('--ann-root',
                        help='path of box gound turth',
                        required=True,
                        type=str)
    parser.add_argument('--kpnum',
                        help='number of keypoints',
                        default=10,
                        type=int)
    parser.add_argument('--h',
                       help='if use sam_h',
                       action='store_true')
    parser.add_argument('--noLOF',
                       help='disable LOF',
                       action='store_true')
    parser.add_argument('--noLA',
                       help='disable LA',
                       action='store_true')
    parser.add_argument('--noPA',
                       help='disable PA',
                       action='store_true')
    parser.add_argument('--pV',
                       help='super param V',
                       default=10,
                       type=int)
    parser.add_argument('--plofk',
                       help='super param lof k',
                       default=5,
                       type=int)
    parser.add_argument('--plofv',
                       help='super param lof v',
                       default=2.5,
                       type=float)
    parser.add_argument('--data-name',
                       help='Data Name',
                       default="video",
                       type=str)
    
    
    args = parser.parse_args()
    return args

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# 加入离群点检测和负样本点
# 加入负样本点调整的正样本点
def get_input_points(flw_img, flw_hw, ori_hw, csv_msg, 
                     k_num=6, pV=20, plofk=5, plofv=2.5):
    global _NO_LOF
    global _NO_LA
    global _NO_PA
    pt_h, pt_w = float(csv_msg['imgh']), float(csv_msg['imgw'])
    im_h, im_w = ori_hw
    fw_h, fw_w = flw_hw
    if _NO_PA:
        pV=0
    input_points = []
    input_labels = []
    for k in range(k_num):
        x = csv_msg['xp'+str(k+1)]
        y = csv_msg['yp'+str(k+1)]

        input_x = float(x)/pt_w*im_w
        input_y = float(y)/pt_h*im_h
        input_points.append([input_x, input_y])
        
        ## 光流图空白点设置为负样本点
        flow_x = float(x)/pt_w*fw_w
        flow_y = float(y)/pt_h*fw_h
        flw_hsv = cv2.cvtColor(flw_img, cv2.COLOR_RGB2HSV) #HSV颜色空间
        
        
        if flw_hsv[int(flow_y)][int(flow_x)][1] <= 3: # 饱和度低的点
            if _NO_LA:
                input_labels.append(1)
            else:
                print('LA')
                input_labels.append(0)
            
            # 对该点进行误差调整
            new_flag, new_s = 0, 0
            new_input_x, new_input_y = input_x, input_y
            for i in range(-pV,pV+1):
                for j in range(-pV,pV+1):
                    flw_s = flw_hsv[int(flow_y+i)][int(flow_x+j)][1]
                    if flw_s <= 5:
                        continue
                    if flw_s > new_s:
                        new_s = flw_s
                        new_input_x = float(int(flow_x + j))/fw_w*im_w
                        new_input_y = float(int(flow_y + i))/fw_h*im_h 
                        new_flag = 1
            if new_flag:
                print('PA')
                input_points.append([new_input_x, new_input_y])
                input_labels.append(1)
        else:
            input_labels.append(1)
    ## 离群点剔除 (LOF)
    if not _NO_LOF:
        print('LOF')
        points_lof = outliers(plofk, plofv, input_points)
        for lof in points_lof:
            input_labels[lof['index']] = 0

    return np.array(input_points), np.array(input_labels)

def get_input_box(ori_hw, input_points, input_labels):
    im_h, im_w = ori_hw
    box_x1, box_y1 = im_w-1, im_h-1
    box_x2, box_y2 = 0, 0
    N,_ = input_points.shape
    for i in range(N):
        x, y = input_points[i][0], input_points[i][1]
        l = input_labels[i]
        if l == 1:
            box_x1 = x if x < box_x1 else box_x1
            box_y1 = y if y < box_y1 else box_y1
            box_x2 = x if x > box_x2 else box_x2
            box_y2 = y if y > box_y2 else box_y2
    input_box = np.array([box_x1, box_y1, box_x2, box_y2])
    return input_box

def get_mask(image, input_points, input_labels, input_box, sam_bh=0):
    import sys
    sys.path.append("..")
    from segment_anything import sam_model_registry, SamPredictor

    sam_checkpoint = ["checkpoints/sam_vit_b_01ec64.pth", "checkpoints/sam_vit_h_4b8939.pth"]
    model_type = ["vit_b", "vit_h"]
    sam = sam_model_registry[model_type[sam_bh]](checkpoint=sam_checkpoint[sam_bh])
    sam.to(device=0)

    predictor = SamPredictor(sam)
    predictor.set_image(image)
    
    mask_output, _, lw_output = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        box=input_box,
        multimask_output=False,
    )

    mask_output, _, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        mask_input=lw_output,
        box=input_box,
        multimask_output=False,
    )
    return mask_output

def calc_iou(mask1, mask2):
    assert mask1.shape == mask2.shape, "mask shape do not match!"
    #print(mask1.max(), mask2.max())
    i_mask = mask1*mask2 
    u_mask = np.array((mask1, mask2)).max(axis=0)
    #print(i_mask.max(), u_mask.max())
    iou = np.sum(i_mask) / np.sum(u_mask)
    io1 = np.sum(i_mask) / np.sum(mask1)
    io2 = np.sum(i_mask) / np.sum(mask2)
    return iou, io1, io2

def calc_box_iou(boxes_gt, box_input):
    #print(boxes_gt)
    iou = []
    ix1, iy1, ix2, iy2 = \
        box_input[0],box_input[1],\
        box_input[2],box_input[3]
    
    for box_gt in boxes_gt:
        x1,y1=min(box_gt[0][0],box_gt[1][0]),min(box_gt[0][1],box_gt[1][1])
        x2,y2=max(box_gt[0][0],box_gt[1][0]),max(box_gt[0][1],box_gt[1][1])
        
        h = max(0, min(x2, ix2) - max(x1, ix1))
        w = max(0, min(y2, iy2) - max(y1, iy1))
        inter = h*w
        area_box1 = (ix2-ix1)*(iy2-iy1)
        area_box2 = (x2-x1)*(y2-y1)
        union = area_box1 + area_box2 - inter
        iou.append(inter / union)
    return iou

def main():
    args = parse_args()
    csv_path = args.csv_path
    ori_root = args.ori_root
    flw_root = args.flw_root
    mgt_root = args.mgt_root
    kpnum = args.kpnum

    ########---------
    global _NO_LOF
    _NO_LOF = bool(args.noLOF)
    global _NO_LA
    _NO_LA = bool(args.noLA)
    global _NO_PA
    _NO_PA = bool(args.noPA)
    print('noLOF:{} noLA:{} noPA:{}'.format(_NO_LOF, _NO_LA, _NO_PA))
    ########---------
    avg_iou = AverageMeter()
    avg_io1 = AverageMeter()
    avg_io2 = AverageMeter()
    avg_io12 = AverageMeter()
    avg_box_iou = AverageMeter()
    
    ########---------
    csv_f = open(csv_path, 'r', encoding='utf8', newline='')
    csv_headers = ['idx','image_path','imgh','imgw']
    for i in range(kpnum):
        csv_headers.append('xp'+str(i+1))
        csv_headers.append('yp'+str(i+1))
    csv_reader = csv.DictReader(csv_f, csv_headers)
    next(csv_reader) #跳过表头行
    #########---------

    #########---------
    out_name = args.data_name + '_pV' + str(args.pV) + '_plofk' +str(args.plofk) + '_plofv' +str(args.plofv)
    if _NO_LOF:
        out_name += "_noLOF"
    if _NO_LA:
        out_name += "_noLA"
    if _NO_PA:
        out_name += "_noPA"
    csv_out_path = out_name + '.csv'
    csv_out_f = open(csv_out_path, 'w', encoding='utf8', newline='')
    csv_headers = ['idx', 'iou', 'io1', 'io2', 'io12', 'box_iou']
    csv_writer = csv.DictWriter(csv_out_f, csv_headers)
    csv_writer.writeheader()
    
    #########---------

    for csv_msg in csv_reader:
        img_name, _ = os.path.splitext(os.path.basename(csv_msg['image_path']))
        img_path = os.path.join(ori_root, img_name + '.jpg')
        flw_path = os.path.join(flw_root, img_name + '.png')        
        mgt_path = os.path.join(mgt_root, img_name + '.png')
        if not os.path.exists(mgt_path):
            print("[{}] 缺少标注".format(img_name))
            continue

        # 读取原图
        ori_img = cv2.imread(img_path)
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        ori_hw = ori_img.shape[0], ori_img.shape[1]
        # 读取光流图
        flw_img = cv2.imread(flw_path)
        flw_img = cv2.cvtColor(flw_img, cv2.COLOR_BGR2RGB)
        flw_hw = flw_img.shape[0], flw_img.shape[1]
        # 读取ground turth
        mgt_img = cv2.imread(mgt_path)
        mgt_img = cv2.cvtColor(mgt_img, cv2.COLOR_BGR2GRAY)
        ret, mgt_mask = cv2.threshold(mgt_img,127,1,cv2.THRESH_BINARY)
        mgt_hw = mgt_img.shape[0], mgt_img.shape[1]
        # 读取 box
        boxes_gt = []
        ann_path = os.path.join(args.ann_root, img_name+'.json')
        with open(ann_path,'r',encoding='utf8') as fp:
            json_data = json.load(fp)
            shapes = json_data['shapes']
            for shape in shapes:
                if shape['shape_type']=='rectangle':
                    boxes_gt.append(shape['points'])

        # 推理
        input_points, input_labels = get_input_points(flw_img, flw_hw, ori_hw, csv_msg, k_num=kpnum, 
                                                      pV=args.pV, plofk=args.plofk, plofv=args.plofv)
        input_box = get_input_box(ori_hw, input_points, input_labels)
        output_mask = get_mask(ori_img, input_points, input_labels, input_box,
                               sam_bh=1 if args.h else 0)

        # 评估
        img_iou, io1, io2 = calc_iou(mgt_mask, output_mask[0].astype(np.uint))
        io12 = (io1+io2)/2
        avg_iou.update(img_iou)
        avg_io1.update(io1)
        avg_io2.update(io2)
        avg_io12.update(io12)
        
        csv_dict = {}
        csv_dict['idx'] = img_name
        csv_dict['iou'] = img_iou
        csv_dict['io1'] = io1
        csv_dict['io2'] = io2
        csv_dict['io12'] = io12

        box_iou_list = calc_box_iou(boxes_gt, input_box)
        box_iou = -1 if len(box_iou_list) == 0 else max(box_iou_list)
        csv_dict['box_iou'] = box_iou
        if box_iou >= 0:
            avg_box_iou.update(box_iou)
        
        csv_writer.writerow(csv_dict)
        print('[{}] iou:{:.4f} io1:{:.4f} io2:{:.4f} iou12:{:.4f} box_iou:{:.4f}'.format(
            img_name, img_iou, io1, io2, io12, box_iou))
        
    csv_out_f.close()
    print('average of iou : {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
            avg_iou.avg, avg_io1.avg, avg_io2.avg, avg_io12.avg, avg_box_iou.avg))
    
if __name__ == '__main__':
    main()
