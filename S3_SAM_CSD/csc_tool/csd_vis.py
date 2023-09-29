import os
import cv2
import csv
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from lof import outliers
# csv_path = '../predictions_06_10.csv'
# ori_root = 'D:/data/vedio_seq_06'
# out_root = 'D:/data/inf_06_10p'
# kpnum = 10


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
    parser.add_argument('--out-root',
                        help='path of mask gound turth',
                        required=True,
                        type=str)
    parser.add_argument('--kpnum',
                        help='number of keypoints',
                        default=10,
                        type=int)
    parser.add_argument('--h',
                       help='if use sam_h',
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

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.3])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=30):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='o', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='o', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def draw_points(coords, labels, img):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    pos_num = pos_points.shape[0]
    neg_num = neg_points.shape[0]
    
    point_size = 5
    pos_point_color = (0, 255, 0)
    neg_point_color = (0, 0, 255) # BGR
    thickness = 4 # 可以为 0 、4、8
    for i in range(pos_num):
        cv2.circle(img, (int(pos_points[i][0]),int(pos_points[i][1])), point_size, pos_point_color, thickness)
    for i in range(neg_num):
        cv2.circle(img, (int(neg_points[i][0]),int(neg_points[i][1])), point_size, neg_point_color, thickness)
    return img

def draw_box(box, img):
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
    

def img_masked(mask, img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    mask_bgra = np.concatenate((mask, mask, mask, mask), axis=0).transpose(1,2,0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img = img * mask_bgra
#    h,w,_ = img.shape
#    for i in range(0,h): #访问所有行
#        for j in range(0,w): #访问所有列
#            if img[i,j,0] == 0 and img[i,j,1] == 0 and img[i,j,2] == 200:
#                img[i,j,3] = 0
    return img
    

# 加入离群点检测和负样本点
# 加入负样本点调整的正样本点
def get_input_points(flw_img, flw_hw, ori_hw, csv_msg, k_num=6, pV=20, plofk=5, plofv=2.5):
    pt_h, pt_w = float(csv_msg['imgh']), float(csv_msg['imgw'])
    im_h, im_w = ori_hw
    fw_h, fw_w = flw_hw

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
                input_points.append([new_input_x, new_input_y])
                input_labels.append(1)
        else:
            input_labels.append(1)
    ## 离群点剔除 (LOF)
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


def draw_contours(ori_img, mask):
    #cv_mask = cv2.fromarray(mask)
    cv_mask = (mask*255).transpose(1,2,0)
    contours, _ = cv2.findContours(cv_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ori_img, contours, -1, (0,0,255), 3)
    return ori_img

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


def main():
    ########--------
    args = parse_args()
    csv_path = args.csv_path
    ori_root = args.ori_root
    flw_root = args.flw_root
    out_root = args.out_root
    kpnum = args.kpnum
    if not os.path.exists(out_root):
        os.makedirs(out_root)    
    ########---------
    csv_f = open(csv_path, 'r', encoding='utf8', newline='')
    csv_headers = ['idx','image_path','imgh','imgw']
    for i in range(kpnum):
        csv_headers.append('xp'+str(i+1))
        csv_headers.append('yp'+str(i+1))
    csv_reader = csv.DictReader(csv_f, csv_headers)
    next(csv_reader) #跳过表头行
    #########---------
    for csv_msg in csv_reader:
        img_name, _ = os.path.splitext(os.path.basename(csv_msg['image_path']))
        img_path = os.path.join(ori_root, img_name + '.jpg')
        flw_path = os.path.join(flw_root, img_name + '.png')

        ori_img = cv2.imread(img_path)
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        ori_hw = ori_img.shape[0], ori_img.shape[1]

        flw_img = cv2.imread(flw_path)
        flw_img = cv2.cvtColor(flw_img, cv2.COLOR_BGR2RGB)
        flw_hw = flw_img.shape[0], flw_img.shape[1]

        input_points, input_labels = get_input_points(flw_img, flw_hw, ori_hw, csv_msg, k_num=kpnum)
        input_box = get_input_box(ori_hw, input_points, input_labels)
        output_mask = get_mask(ori_img, input_points, input_labels, input_box)
        
        # 保存单独的mask
        gray_mask = output_mask.transpose(1,2,0).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(out_root, img_name+"_mask" +".jpg"),gray_mask)
        out_img = cv2.cvtColor(gray_mask, cv2.COLOR_GRAY2BGR)
        # 保存mask+point+box
        out_img = draw_points(input_points, input_labels, out_img)
        out_img = draw_box(input_box, out_img)
        cv2.imwrite(os.path.join(out_root, img_name+"_out"+".jpg"), out_img)
        
        # 保存img+mask
        mask = output_mask.astype(np.uint8)
        img_mask=draw_mask(mask, ori_img)
        cv2.imwrite(os.path.join(out_root, img_name+"_final"+".jpg"), img_mask)
        
        # 保存mask(img)
        
        img_mkd = img_masked(mask, ori_img)
        cv2.imwrite(os.path.join(out_root, img_name+"_mkd"+".png"), img_mkd, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        
        print("[{}] Done.".format(img_name))
        # 绘制边框
        #ori_img = draw_contours(ori_img, output_mask.astype('uint8'))
        #plt.figure(figsize=(20,15))
        #plt.imshow(ori_img)
        #show_points(input_points, input_labels, plt.gca())
        #show_box(input_box, plt.gca())
        #show_mask(output_mask, plt.gca())
        #plt.axis('off')
        #plt.savefig(os.path.join(out_root, img_name+'_out.jpg'), dpi=200)
        #print('Save Result to : {} {}'.format(out_root, img_name))
        #plt.close()
        
    csv_f.close()
    print('Done.')
    #########

if __name__ == '__main__':
    main()
