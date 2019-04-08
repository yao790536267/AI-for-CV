#!/usr/bin/env python
# coding: utf-8

#导入工具包
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
import os

#裁剪函数
def image_crop(img):
    height, width,channel = img.shape
    img_crop = img[int(random.uniform(0,0.1) * width):int(random.uniform(0.9,1.0) * width),
                   int(random.uniform(0,0.1) * height):int(random.uniform(0.9,1.0) * height)]
    return img_crop

#调色函数
def random_light_color(img,band = 50):
    B,G,R = cv2.split(img)
    #蓝色灰度矩阵整体偏移，＞255取255，<0就取0
    b_rand = random.randint(-band,band)
    if b_rand == 0:
        pass
    elif b_rand > 0:
        lim = 255 - b_rand
        B[B > lim] = 255
        B[B <= lim] = (b_rand + B[B <= lim]).astype(img.dtype)
    elif b_rand < 0:
        lim = 0 - b_rand
        B[B < lim] = 0
        B[B >= lim] = (b_rand + B[B >= lim]).astype(img.dtype)
    #绿色灰度矩阵整体偏移，＞255取255，<0就取0
    g_rand = random.randint(-band,band)
    if g_rand == 0:
        pass
    elif g_rand > 0:
        lim = 255 - g_rand
        G[G > lim] = 255
        G[G <= lim] = (g_rand + G[G <= lim]).astype(img.dtype)
    elif g_rand < 0:
        lim = 0 - g_rand
        G[G < lim] = 0
        G[G >= lim] = (g_rand + G[G >= lim]).astype(img.dtype)
    #红色灰度矩阵整体偏移，＞255取255，<0就取0
    r_rand = random.randint(-band,band)
    if r_rand == 0:
        pass
    elif r_rand > 0:
        lim = 255 - r_rand
        R[R > lim] = 255
        R[R <= lim] = (r_rand + R[R <= lim]).astype(img.dtype)
    elif r_rand < 0:
        lim = 0 - r_rand
        R[R < lim] = 0
        R[R >= lim] = (r_rand + R[R >= lim]).astype(img.dtype)
    img_merge = cv2.merge((B,G,R))
    return img_merge

#调饱和度函数
def saturation(img,band = 50):
    #将BGR转换到HLS空间
    img_hls = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    H,L,S = cv2.split(img_hls)
    #饱和度矩阵整体偏移，＞255取255，<0就取0
    s_rand = random.randint(-band,band)
    if s_rand == 0:
        pass
    elif s_rand > 0:
        lim = 255 - s_rand
        S[S > lim] = 255
        S[S <= lim] = (s_rand + S[S <= lim]).astype(img.dtype)
    elif s_rand < 0:
        lim = 0 - s_rand
        S[S < lim] = 0
        S[S >= lim] = (s_rand + S[S >= lim]).astype(img.dtype)
    img_hls = cv2.merge((H,L,S))
    img_bgr_sat = cv2.cvtColor(img_hls,cv2.COLOR_HLS2BGR)
    return img_bgr_sat

#旋转函数
def rotation(img):
    M = cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2),
                                random.randint(0,360),
                                random.uniform(0.9,1.0))
    img_rotate = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))
    return img_rotate

#投影函数
def random_warp(img,factor = 0.15):
    height,width,channel = img.shape
    #原图4个不共线点
    random_margin = int(min(factor * height,factor * width))
    x1 = random.randint(-random_margin,random_margin)
    y1 = random.randint(-random_margin,random_margin)
    x2 = random.randint(width - random_margin - 1,width - 1)
    y2 = random.randint(-random_margin,random_margin)
    x3 = random.randint(width - random_margin - 1,width - 1)
    y3 = random.randint(height - random_margin - 1,height - 1)
    x4 = random.randint(-random_margin,random_margin)
    y4 = random.randint(height - random_margin - 1,height - 1)
    #目标图4个不共线点
    dx1 = random.randint(-random_margin,random_margin)
    dy1 = random.randint(-random_margin,random_margin)
    dx2 = random.randint(width - random_margin - 1,width - 1)
    dy2 = random.randint(-random_margin,random_margin)
    dx3 = random.randint(width - random_margin - 1,width - 1)
    dy3 = random.randint(height - random_margin - 1,height - 1)
    dx4 = random.randint(-random_margin,random_margin)
    dy4 = random.randint(height - random_margin - 1,height - 1)
    
    pst1 = np.float32([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
    pst2 = np.float32([[dx1,dy1],[dx2,dy2],[dx3,dy3],[dx4,dy4]])
    M_warp = cv2.getPerspectiveTransform(pst1,pst2)
    img_warp = cv2.warpPerspective(img,M_warp,(width,height))
    return img_warp

#统一做所有处理函数
def process(img):
    img = image_crop(img)
    img = random_light_color(img)
    img = saturation(img)
    img = rotation(img)
    img = random_warp(img)
    return img

#导入数据、处理并输出 
def findjpg(jpglist = [],path=None):
    """Finding the *.jpg file in specify path"""
    filelist = os.listdir(path)
    for filename in filelist:
        if filename.endswith(".jpg"): #Specify to find the jpg file.
            jpglist.append(filename)
    return jpglist

jpglist = findjpg()

def generate(jpglist,num = 5):
    for jpgname in jpglist:
        jpgimg = cv2.imread(jpgname)
        for i in range(num):
            output = process(jpgimg)
            cv2.imwrite("Copy{}_of_{}".format(i,jpgname),output,[int(cv2.IMWRITE_JPEG_QUALITY), random.randint(85,100)])
generate(jpglist)




