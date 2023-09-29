import os
import cv2
import csv
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO SAM Evaluation.')
    parser.add_argument('--lab-root',
                        help='root of yolo detect label',
                        required=True,
                        type=str)
    parser.add_argument('--ori-root',
                        help='path of original images',
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
    parser.add_argument('--out-root',
                        help='path of out',
                        default='/root/autodl-tmp/vis_out_yolo',
                        type=str)
    parser.add_argument('--h',
                       help='if use sam_h',
                       action='store_true')
    parser.add_argument('--data-name',
                       help='Data Name',
                       default="video",
                       type=str)
    parser.add_argument('--viz',
                       help='if save viz',
                       action='store_true')
    
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
    
def get_mask(image, input_boxes, sam_bh=0):
    import sys
    sys.path.append("..")
    from segment_anything import sam_model_registry, SamPredictor

    sam_checkpoint = ["checkpoints/sam_vit_b_01ec64.pth", "checkpoints/sam_vit_h_4b8939.pth"]
    model_type = ["vit_b", "vit_h"]
    sam = sam_model_registry[model_type[sam_bh]](checkpoint=sam_checkpoint[sam_bh])
    sam.to(device=0)
   
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    
    box_N = input_boxes.shape[0]
    lw_output = None
    for i in range(box_N):
        input_box = input_boxes[i]
        mask_output, _, lw = predictor.predict(
            mask_input=None,
            box=input_box,
            multimask_output=False,
        )
        lw_output = lw

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


def calc_box_iou(boxes_gt, box_inputs):
    #print(boxes_gt)
    iou = []
    box_N = box_inputs.shape[0]
    for i in range(box_N):
        box_input = box_inputs[i]
        
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

def draw_box(boxes, img):
    box_N = boxes.shape[0]
    for i in range(box_N):
        box = boxes[i]
        x0, y0 = int(box[0]), int(box[1])
        x1, y1 = int(box[2]), int(box[3])
        box_color = (241,215,164)
        cv2.rectangle(img, (x0, y0), (x1, y1), color=box_color, thickness=2)
    return img

def draw_mask(mask, img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    mask_color = (0,165,255)
    mask_b = mask * mask_color[0]
    mask_g = mask * mask_color[1]
    mask_r = mask * mask_color[2]
    mask_rbg = np.concatenate((mask_b, mask_g, mask_r), axis=0).transpose(1,2,0)
    mask_img = 0.3*mask_rbg + img
    return mask_img
    


def main():
    args = parse_args()
    lab_root = args.lab_root
    ori_root = args.ori_root
    mgt_root = args.mgt_root
    ann_root = args.ann_root
    
    ########---------
    avg_iou = AverageMeter()
    avg_io1 = AverageMeter()
    avg_io2 = AverageMeter()
    avg_io12 = AverageMeter()
    avg_box_iou = AverageMeter()
    
    if args.viz and not os.path.exists(args.out_root):
        os.makedirs(args.out_root)
    
    #########---------
    out_name = args.data_name
    csv_out_path = out_name + '.csv'
    csv_out_f = open(csv_out_path, 'w', encoding='utf8', newline='')
    csv_headers = ['idx', 'yolo_box_iou', 'iou', 'io1', 'io2', 'io12']
    csv_writer = csv.DictWriter(csv_out_f, csv_headers)
    csv_writer.writeheader()

    #########---------
    file_names = list(filter(lambda x: x.endswith('.png'),os.listdir(mgt_root)))
    print(file_names)
    for file_name in file_names:
        image_name,_ = os.path.splitext(file_name)
        label_path = os.path.join(lab_root, image_name+'.txt')
        img_path = os.path.join(ori_root, image_name+'.jpg')
        mgt_path = os.path.join(mgt_root, image_name+'.png')
        ann_path = os.path.join(ann_root, image_name+'.json')
        
        csv_dict = {}
        csv_dict['idx'] = image_name
        
        # yolo未检测出csc框
        if not os.path.exists(label_path):
            csv_dict['yolo_box_iou'] = 0
            csv_dict['iou'] = 0
            csv_dict['io1'] = 0
            csv_dict['io2'] = 0
            csv_dict['io12'] = 0
            csv_writer.writerow(csv_dict)
            avg_iou.update(0)
            avg_io1.update(0)
            avg_io2.update(0)
            avg_io12.update(0)
            avg_box_iou.update(0)
            print('label not exist!')
            continue
        
        # 读取原图
        ori_img = cv2.imread(img_path)
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        ori_hw = ori_img.shape[0], ori_img.shape[1]
        # 读取 ground truth mask
        mgt_img = cv2.imread(mgt_path)
        mgt_img = cv2.cvtColor(mgt_img, cv2.COLOR_BGR2GRAY)
        ret, mgt_mask = cv2.threshold(mgt_img,127,1,cv2.THRESH_BINARY)
        mgt_hw = mgt_img.shape[0], mgt_img.shape[1]
        # 读取 ground truth box
        boxes_gt = []
        with open(ann_path,'r',encoding='utf8') as fp:
            json_data = json.load(fp)
            shapes = json_data['shapes']
            for shape in shapes:
                if shape['shape_type']=='rectangle':
                    boxes_gt.append(shape['points'])

        # 读取 yolo box
        box_list = []
        with open(label_path,'r',encoding='utf8') as yolo_fp:
            while True:
                line = yolo_fp.readline()
                if line == '':
                    break
                yolo_labels = line.strip().split()
                w = float(yolo_labels[3])*ori_hw[1]
                h = float(yolo_labels[4])*ori_hw[0]
                x1=float(yolo_labels[1])*ori_hw[1] - 0.5*w
                y1=float(yolo_labels[2])*ori_hw[0] - 0.5*h
                x2=x1+w #x1+w
                y2=y1+h #y1+h
                box_list.append([int(x1),int(y1),int(x2),int(y2)])
        input_box = np.array(box_list)
        
        # 推理
        output_mask = get_mask(ori_img, input_box, sam_bh=1 if args.h else 0)
        
        # 评估
        img_iou, io1, io2 = calc_iou(mgt_mask, output_mask[0].astype(np.uint))
        io12 = (io1+io2)/2
        avg_iou.update(img_iou)
        avg_io1.update(io1)
        avg_io2.update(io2)
        avg_io12.update(io12)
        
        csv_dict['iou'] = img_iou
        csv_dict['io1'] = io1
        csv_dict['io2'] = io2
        csv_dict['io12'] = io12        

        box_iou_list = calc_box_iou(boxes_gt, input_box)
        box_iou = -1 if len(box_iou_list) == 0 else max(box_iou_list)
        csv_dict['yolo_box_iou'] = box_iou
        if box_iou >= 0:
            avg_box_iou.update(box_iou)
        csv_writer.writerow(csv_dict)
        print('[{}] iou:{:.4f} io1:{:.4f} io2:{:.4f} iou12:{:.4f} box_iou:{:.4f}'.format(
            image_name, img_iou, io1, io2, io12, box_iou))
        
        # 保存 img+yolo_box
        #out_img = draw_box(input_box, ori_img)
        #cv2.imwrite(os.path.join(args.out_root,image_name+'_out.png'), out_img)
        
        # 保存 img+yolo_box+mask
        mask = output_mask.astype(np.uint8)
        out_img=draw_box(input_box, ori_img)
        out_img=draw_mask(mask, out_img)
        cv2.imwrite(os.path.join(args.out_root, image_name+"_final"+".jpg"), out_img)
        
    csv_out_f.close()
    print('average of iou : {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
            avg_iou.avg, avg_io1.avg, avg_io2.avg, avg_io12.avg, avg_box_iou.avg))
    
    
if __name__ == '__main__':
    main()