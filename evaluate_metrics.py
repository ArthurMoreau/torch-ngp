import argparse
import numpy as np
import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
import os
import lpips
import torch
import csv

def computeMetrics(imgs_list, gt_list):
    metrics_table = {}
    ssim_list = []
    mse_list = []
    psnr_list = []
    lpips_list = []
    loss_fn_alex = lpips.LPIPS(net='alex')
    #loss_fn_alex.net = loss_fn_alex.net.float()

    for i in range(len(imgs_list)):
        max = (imgs_list[i]).max()
        min = (imgs_list[i]).min()
        # print(max)
        # print(min)
        ssim_ = ssim(imgs_list[i], gt_list[i], data_range=max - min, channel_axis=2)
        mse_ = mean_squared_error(imgs_list[i], gt_list[i])
        psnr_ = psnr(imgs_list[i], gt_list[i])

        ## image from -1 to 1
        lpips_ = loss_fn_alex(torch.Tensor(imgs_list[i]*2-1).permute(2, 0, 1).unsqueeze(0), torch.Tensor(gt_list[i]*2-1).permute(2, 0, 1).unsqueeze(0)).item()

        ssim_list.append(ssim_)
        mse_list.append(mse_)
        psnr_list.append(psnr_)
        lpips_list.append(lpips_)

    metrics_table['ssim'] = ssim_list
    metrics_table['mse'] = mse_list
    metrics_table['psnr'] = psnr_list
    metrics_table['lpips'] = lpips_list

    return metrics_table

def load_imgs(img_path, gt_path, dsize=(235,132)):
    imgs_list = []
    gt_list = []
    for filename in os.listdir(img_path):
        img = cv2.imread(os.path.join(img_path,filename))
        if img is not None:
            img = cv2.resize(img, dsize)
            img = img /255.0
            imgs_list.append(img)
    for filename in os.listdir(gt_path):
        gt = cv2.imread(os.path.join(gt_path,filename))
        if gt is not None:
            gt = cv2.resize(gt, dsize)
            gt = gt /255.0
            gt_list.append(gt)
    return imgs_list, gt_list

def createCSV(outputpath, metrcis_table):
    field_names = ['ssim', 'mse', 'psnr', 'lpips']
    
    with open(outputpath, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = field_names)
        writer.writeheader()
        writer.writerow(metrcis_table)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_pre', type=str, help="prediction images path")
    parser.add_argument('--path_gt', type=str, help="grounftruth images path")
    parser.add_argument('--output_path', type=str, help="output csv path")
    parser.add_argument('--W', type=int, default=235, help="loaded image size")
    parser.add_argument('--H', type=int, default=132, help="loaded image size")

    opt = parser.parse_args()
    print(opt)

    path_pre = opt.path_pre
    path_gt = opt.path_gt
    img_list, gt_list = load_imgs(path_pre, path_gt, dsize=(opt.W,opt.H))
    metrcis_table = computeMetrics(img_list, gt_list)
    createCSV(opt.output_path, metrcis_table)


