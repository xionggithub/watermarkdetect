#coding=utf-8
import cv2
import numpy as np
import time
import math
import os
import shutil
from matplotlib import pyplot as plt


def main():

    # sift = cv2.xfeatures2d.SIFT_create()
    # imreadModes = 1 #0 IMREAD_GRAYSCALE  1 IMREAD_COLOR
    # image = cv2.imread("test.jpg",imreadModes)          # trainImage
    # kp1, des1 = sift.detectAndCompute(image,None)
    # img = None
    # img = cv2.drawKeypoints(image,kp1,image,color=(255,0,0))
    # print(type(img))
    # cv2.imwrite("result.jpg",image)


    surf = cv2.xfeatures2d.SURF_create()
    imreadModes = 1 #0 IMREAD_GRAYSCALE  1 IMREAD_COLOR
    image = cv2.imread("test.jpg",imreadModes)          # trainImage
    kp1, des1 = surf.detectAndCompute(image,None)
    img = None
    img = cv2.drawKeypoints(image,kp1,image,color=(255,0,0))
    print(type(img))
    cv2.imwrite("result.jpg",image)

if __name__ == '__main__':
    main()
