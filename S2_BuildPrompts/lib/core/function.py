# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import os
import cv2
import torch
import numpy as np

from .evaluation import decode_preds, compute_nme, decode_preds_csc

logger = logging.getLogger(__name__)


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

        
        
def save_image_with_points(inp, preds, meta, config):
    batch_size,_,image_size,_ = inp.shape
    num_classes = config.MODEL.NUM_JOINTS
    
    for bi in range(batch_size):
        # tensor 转 cv2
        img_tensor = inp[bi].clone().detach()
        img_tensor = img_tensor.to(torch.device('cpu'))
        img_cv2 = img_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
        img_cv2 = cv2.cvtColor(img_cv2,cv2.COLOR_RGB2BGR)
        # 关键点绘制 preds.shape = 16, 6, 2
        for ki in range(num_classes):
            x = preds[bi][ki][0]
            y = preds[bi][ki][1]
            img_cv2 = cv2.circle(img_cv2, (int(x),int(y)),3,(0,0,255),-1)
        # 保存图片
        image_path = os.path.join(config.TEST.OUTPUT_DIR, "kp_{}.png".format(meta['index'][bi]))
        cv2.imwrite(image_path, img_cv2)
        print("保存图片至:", image_path)

def train(config, train_loader, model, critertion, optimizer,
          epoch, writer_dict):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    nme_count = 0
    nme_batch_sum = 0

    end = time.time()

    for i, (inp, target, meta) in enumerate(train_loader):
        # measure data time
        data_time.update(time.time()-end)

        # compute the output
        output = model(inp)
        target = target.cuda(non_blocking=True)

        loss = critertion(output, target)

        _,_,image_size,_ = inp.shape
        _,_,htmp_size,_  = output.shape
        # NME
        score_map = output.data.cpu()
        #preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])
        preds = decode_preds_csc(score_map, [image_size,image_size], [htmp_size, htmp_size])
        print("[debug] preds:", preds.shape)
        nme_batch = compute_nme(preds, meta)
        nme_batch_sum = nme_batch_sum + np.sum(nme_batch)
        nme_count = nme_count + preds.size(0)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inp.size(0))

        batch_time.update(time.time()-end)
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=inp.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()
    nme = nme_batch_sum / nme_count
    msg = 'Train Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f}'\
        .format(epoch, batch_time.avg, losses.avg, nme)
    logger.info(msg)


def validate(config, val_loader, model, criterion, epoch, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(val_loader.dataset), num_classes, 2))

    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()

    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(val_loader):
            data_time.update(time.time() - end)
            output = model(inp)
            target = target.cuda(non_blocking=True)
            _,_,image_size,_ = inp.shape
            _,_,htmp_size,_  = output.shape

            score_map = output.data.cpu()
            # loss
            loss = criterion(output, target)

            preds = decode_preds_csc(score_map, [image_size,image_size], [htmp_size, htmp_size])
            # NME
            nme_temp = compute_nme(preds, meta)
            # Failure Rate under different threshold
            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + preds.size(0)
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            losses.update(loss.item(), inp.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = 'Test Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(epoch, batch_time.avg, losses.avg, nme,
                                failure_008_rate, failure_010_rate)
    logger.info(msg)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', losses.avg, global_steps)
        writer.add_scalar('valid_nme', nme, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return nme, predictions

def test(config, data_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(data_loader.dataset), num_classes, 2))

    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()

    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(data_loader):
            data_time.update(time.time() - end)
            output = model(inp)
            score_map = output.data.cpu()

            batch_size,_,image_size,_ = inp.shape
            _,_,htmp_size,_  = output.shape

            preds = decode_preds_csc(score_map, [image_size, image_size], [htmp_size, htmp_size])

            # NME
            nme_temp = compute_nme(preds, meta)

            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + preds.size(0)
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # save_image
            for bi in range(batch_size):
                # tensor 转 cv2
                img_tensor = inp[bi].clone().detach()
                img_tensor = img_tensor.to(torch.device('cpu'))
                img_cv2 = img_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
                img_cv2 = cv2.cvtColor(img_cv2,cv2.COLOR_RGB2BGR)
                # 关键点绘制 preds.shape = 16, 6, 2
                for ki in range(num_classes):
                    x = preds[bi][ki][0]
                    y = preds[bi][ki][1]
                    img_cv2 = cv2.circle(img_cv2, (int(x),int(y)),3,(0,0,255),-1)
                # 保存图片
                image_path = os.path.join(config.TEST.OUTPUT_DIR, "kp_{}.png".format(meta['index'][bi]))
                cv2.imwrite(image_path, img_cv2)
                print("保存图片至:", image_path)

    nme = nme_batch_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = 'Test Results time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(batch_time.avg, losses.avg, nme,
                                failure_008_rate, failure_010_rate)
    logger.info(msg)

    return nme, predictions




def test_bak(config, data_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(data_loader.dataset), num_classes, 2))

    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()

    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(data_loader):
            data_time.update(time.time() - end)
            output = model(inp)
            score_map = output.data.cpu()
            #preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])
            
            batch_size,_,image_size,_ = inp.shape
            _,_,htmp_size,_  = output.shape
            
            preds = decode_preds_csc(score_map, [image_size, image_size], [htmp_size, htmp_size])

            # NME
            nme_temp = compute_nme(preds, meta)

            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + preds.size(0)
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = 'Test Results time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(batch_time.avg, losses.avg, nme,
                                failure_008_rate, failure_010_rate)
    logger.info(msg)

    return nme, predictions


def inference(config, data_loader, model, final_output_dir, save_image=False):
    num_classes = config.MODEL.NUM_JOINTS
    #predictions = torch.zeros((len(data_loader.dataset), num_classes, 2))
    
    # csv写入准备
    import csv
    csv_headers = ['idx', 'image_path', 'imgh', 'imgw']
    for i in range(num_classes):
        csv_headers.append('xp'+str(i+1))
        csv_headers.append('yp'+str(i+1))
    csv_path = os.path.join(final_output_dir, 'predictions.csv')
    csv_f = open(csv_path, 'w', encoding='utf8', newline='')
    csv_writer = csv.DictWriter(csv_f, csv_headers)
    csv_writer.writeheader() #写入文件头

    with torch.no_grad():      
        for inp, meta in data_loader:
            output = model(inp)
            score_map = output.data.cpu()
            img_sz, htm_sz = config.MODEL.IMAGE_SIZE, config.MODEL.HEATMAP_SIZE
            preds = decode_preds_csc(score_map, img_sz, htm_sz)

            # 构造写入序列化文件的矩阵
            # bn = score_map.size(0)
            # for n in range(bn):
            #     predictions[meta['index'][n], :, :] = preds[n, :, :]
            if save_image:
                if not os.path.exists(config.TEST.OUTPUT_DIR):
                    os.makedirs(config.TEST.OUTPUT_DIR)
                save_image_with_points(inp, preds, meta, config)
            # 构造写入csv文件的字典
            bn = score_map.size(0)
            for n in range(bn):
                csv_dict = {}
                csv_dict['idx'] = meta['index'][n]
                csv_dict['image_path'] = meta['name'][n]
                csv_dict['imgh'] = img_sz[0]
                csv_dict['imgw'] = img_sz[1]
                for i in range(num_classes):
                    x,y = preds[n][i][0],preds[n][i][1]
                    csv_dict['xp'+str(i+1)] = float(x)
                    csv_dict['yp'+str(i+1)] = float(y)
                csv_writer.writerow(csv_dict)
    csv_f.close()
    msg = 'Inference Results Save to : {}'.format(csv_path)
    logger.info(msg)


