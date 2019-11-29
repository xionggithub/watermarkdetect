#coding=utf-8
import cv2
import numpy as np
import time
import math
import os
import shutil
from matplotlib import pyplot as plt


def main():

    sift = cv2.xfeatures2d.SIFT_create()
    # sift = cv2.xfeatures2d.SURF_create()
    imreadModes = 1 #0 IMREAD_GRAYSCALE  1 IMREAD_COLOR
    img1 = cv2.imread("mask_logo.png",imreadModes)          # queryImage
    img2 = cv2.imread("test_frame.jpg",imreadModes)          # trainImage
    
    print(type(img2.shape))
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = []
    if not type(des1)== type(None) and  not type(des2)== type(None) and len(des1) >= 2 and len(des2) >= 2:
        matches = flann.knnMatch(des1,des2,k=2)

    # mts = []
    # for m,n in matches:
    #     mts.append(m)
    # ms = cv2.xfeatures2d.matchGMS((img1.shape[1],img1.shape[0]),(img2.shape[1],img2.shape[0]),kp1,kp2,mts,False,True,6.0)
    # print("ms ::::::::::::"+str(ms))

    good = []
    #通过distance 0.7 提取good 的match点 
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    print("good ::::::::::::"+str(good))            

if __name__ == '__main__':
    main()
