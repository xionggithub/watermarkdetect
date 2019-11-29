
import cv2
import numpy as np
import time
import math
import os
import shutil
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

def  wmdr_print(des):
    # print(str(des)+"!")
    # print(str(des)+"\n")
    des = None

def wmdr_getSubFiles(file_dir):   
    fileArray = []
    for root, dirs, files in os.walk(file_dir):  
        #root 当前目录路径  dirs 当前路径下所有子目录  files当前路径下所有非目录子文件
        for name in files:
            file_dir = os.path.join(root, name)
            fileArray.append([file_dir, name])
        break
    return fileArray

def wmdr_drawRectInImageAndSaveImage(image, rect, writeImageDes):
    addRectTempImage = image.copy()
    cv2.rectangle(addRectTempImage,(int(rect[0]),int(rect[1])),(int(rect[2]),int(rect[3])),(255,255,0),2)
    if not len(writeImageDes)==0:
        cv2.imwrite(writeImageDes,addRectTempImage)

def wmdr_drawKeyPointsInImageAndSaveImage(image, keyPoints, writeImageDes):
    addFeatureTempImage = image.copy()
    addFeatureTempImage = cv2.drawKeypoints(addFeatureTempImage,keyPoints,addFeatureTempImage,color=(255,0,0))
    if not len(writeImageDes)==0: 
        cv2.imwrite(writeImageDes,addFeatureTempImage)


def wmdr_correctRect(rect, img1, img2):
    #image image for train 用于训练的模型或者定位水印的视频帧或者图片
    h1, w1 = img1.shape[0:2]
    h2, w2 = img2.shape[0:2]

    # 如果定位的位置超出了img2的大小 或者 宽高 过小  或者宽高比和模板的比例差别过大说明是错的
    if rect[0] < 0 or rect[1] < 0 or rect[2] > w2 or rect[3] > h2 or (rect[2] - rect[0] <= 2) or (rect[3] - rect[1] <= 2):
        rect = [0 ,0 , 0 , 0]
    else:
        h = float(rect[3] - rect[1])
        w = float(rect[2] - rect[0])
        if abs(h/w  - h1/w1) >= 0.2:
            rect = [0 , 0 , 0 , 0]
    return rect

def wmdr_getCombinRectFromRects(rects):
    combinRect = [0, 0 , 0 ,0]
    startX  = 0;
    endX    = 1000000;
    startY  = 0;
    endY    = 1000000;
    for rect in rects:
        if rect[0] > startX:    startX = rect[0]
        if rect[1] > startY:    startY = rect[1]
        if rect[2] < endX:    endX = rect[2]
        if rect[3] < endY:    endY = rect[3]
    combinRect = [startX, startY, endX, endY]
    if startX >= endX or startY >= endY:
        combinRect = [0, 0 , 0 ,0]
    return combinRect

def wmdr_getRectWithMindistanceCenter(rects):
    centers = []
    for rect in rects:
        centers.append([(rect[2]+rect[0])/2, (rect[3]+rect[1])/2])

    wmdr_print(centers)
    pointDistances = []
    for i in range(len(centers)):
        point1 = centers[i]
        distance = 0
        for j in range(len(centers)):
            if not i==j:
                point2 = centers[j]
                xoffset = abs(point1[0] - point2[0])
                yoffset = abs(point1[1] - point2[1])
                distance += (xoffset*xoffset + yoffset*yoffset)**0.5
        count = len(centers)-1
        if count <= 0:
            count = 1
        pointDistances.append("%.2f" % (distance/count))

    wmdr_print(pointDistances)
    minDistance = 1000000;
    index = 0
    for i in range(len(pointDistances)):
        distance = float(pointDistances[i])
        if distance < minDistance:
            minDistance = distance
            index = i
    wmdr_print("minDistance :" + str(minDistance))
    if minDistance >= 5:
        return [0, 0 , 0 ,0]
    return rects[index]

# 从获取到的规则矩形中统计分类 判断生成的矩形正确性
def wmdr_getMatchRectForRects(rects,frames_count,tempImage):
    newRects = []
    for rect in rects:
        if rect[3] - rect[1] > 0 and rect[2] - rect[0] > 0:
            newRects.append(rect)
            wmdr_drawRectInImageAndSaveImage(tempImage , rect,"./result/"+str(rect)+"标记水印位置图.jpg")

    findCount  =  len(newRects)
    if findCount <= 1:
        rect = [0,0,0,0]
        if findCount > 0:
            rect = newRects[0]
            print("检测到的水印正确率大概为："+str(50+int(findCount*(10.0/frames_count)))+" %")
        return rect
    #test
    minDistancecenterRect = wmdr_getRectWithMindistanceCenter(newRects)
    if minDistancecenterRect[3] - minDistancecenterRect[1] > 0:
        wmdr_drawRectInImageAndSaveImage(tempImage , minDistancecenterRect,"./result/最小center标记水印位置图.jpg")
    combinRect = wmdr_getCombinRectFromRects(newRects)
    if combinRect[3] - combinRect[1] > 0:
        wmdr_drawRectInImageAndSaveImage(tempImage , combinRect,"./result/交集标记水印位置图.jpg")
        print("检测到的水印正确率大概为："+str(50+int(findCount*(10.0/frames_count)))+" %")
    return combinRect

def wmdr_printMatch(kp1, kp2, matches):
    for m, n in matches:        
        print(str("%.4f" % m.distance) + "  "+str("%.4f" % n.distance) + "  rate: " +str("%.4f" %  (m.distance/n.distance)))

def wmdr_sortDistanceRate(distanceRates):
    for i in range(len(distanceRates) - 1): 
        for j in range(len(distanceRates) - i - 1):
            if distanceRates[j] > distanceRates[j + 1]:
                distanceRates[j], distanceRates[j + 1] = distanceRates[j + 1], distanceRates[j]
    return distanceRates


def wmdr_getDistanceRateForMatchCount(matches, count):
    maxDistaceRate = 0.7
    distanceRates = [] 
    for m, n in matches:        
        distanceRates.append(m.distance/n.distance)
    distanceRates = wmdr_sortDistanceRate(distanceRates)
    if len(distanceRates) >= count:
        maxDistaceRate = distanceRates[count-1]
    else:
        maxDistaceRate = distanceRates[len(distanceRates)-1]
    return maxDistaceRate

def test_draw_trainkp(trainKps, matches, trainImage, fileName, imreadModes):
    keyPoints = []
    for m in matches:
        keyPoints.append(trainKps[m.trainIdx])
    imageWritePath = fileName+"关键点.jpg"
    if imreadModes == 0:
        imageWritePath = fileName+"关键点gray.jpg"
    wmdr_drawKeyPointsInImageAndSaveImage(trainImage, keyPoints, imageWritePath)

def wmdr_find_watermark_for_get_good_matches(image1, image2, sift):
    img1 = image1
    img2 = image2
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    matches = []
    fast = True
    #快速匹配
    if fast == True:
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)
    else:
        #暴利匹配
        bf = cv2.BFMatcher()
        #返回k个最佳匹配
        matches = bf.knnMatch(des1, des2, k=2)

    mindistanceRate = 0.7
    maxdistanceRate = 0.7
    distanceRates = []
    for m,n in matches:
        distanceRates.append(m.distance/n.distance)

    if len(distanceRates) > 0:
        distanceRates = wmdr_sortDistanceRate(distanceRates)
        mindistanceRate = distanceRates[0]
        maxdistanceRate = distanceRates[len(distanceRates) - 1]
    # print("mindistanceRate :"+str(mindistanceRate) + ", maxdistanceRate :"+str(maxdistanceRate))

    good = []
    distanceRate = mindistanceRate + 0.3*(maxdistanceRate - mindistanceRate)
    for m,n in matches:
        distanceRates.append(m.distance/n.distance)
        if m.distance < distanceRate*n.distance:
            good.append(m)
            # print("m.distance :"+str(m.distance) + " n.distance :"+ str(n.distance) +"  "+ str(distanceRate))

    return kp1, kp2, good

    

def wmdr_findWaterRect(sift, img1, img2, img2Name, imreadModes):
    #img1 image for qurey 用于查找定位的水印图
    #img2 image for train 用于训练的模型或者定位水印的视频帧或者图片
    rect = [0, 0 , 0 , 0]
    kp11, kp21, goodMatches1 = wmdr_find_watermark_for_get_good_matches(img1, img2, sift)
    kp22, kp12, goodMatches2 = wmdr_find_watermark_for_get_good_matches(img2, img1, sift)
    good = []
    for m1 in goodMatches1:
        for m2 in goodMatches2:
            pt11 = kp11[m1.queryIdx].pt
            pt21 = kp21[m1.trainIdx].pt
            pt22 = kp22[m2.queryIdx].pt
            pt12 = kp12[m2.trainIdx].pt
            if abs(int(pt11[0]) - int(pt12[0])) < 0.001 and abs(int(pt11[1]) - int(pt12[1])) < 0.001 and abs(int(pt21[0]) - int(pt22[0])) < 0.001 and abs(int(pt21[1]) - int(pt22[1])) < 0.001:
                good.append(m1)

    wmdr_print("find_watermark_for_not_enough_matches find good len :"+str(len(good)))
    rect = [0 , 0 , 0 , 0]
    if len(good) > 0:
        goodKps1 = []
        goodKps2 = []
        scale = 0
        goodMatch = None
        mindistance = 10000
        for m in good:
            keyPoint1 = kp11[m.queryIdx]
            keyPoint2 = kp21[m.trainIdx]
            goodKps1.append(keyPoint1)
            goodKps2.append(keyPoint2)

            angle1 = keyPoint1.angle
            angle2 = keyPoint2.angle
            # print("  angle : "+str(angle1 - angle2))
            if abs(angle1 - angle2) < 10.0:
                if m.distance < mindistance:
                    mindistance = m.distance
                    goodMatch = m
                size1 = keyPoint1.size
                size2 = keyPoint2.size
                scale += size1/size2
                # print(" scale:  "+ str(size1/size2))
            
        scale /= len(good) 

        # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
        #                singlePointColor = None,
        #                matchesMask = None, # draw only inliers
        #                flags = 2)
        # img3 = cv2.drawMatches(img1,kp11,img2,kp21,good,None,**draw_params)
        # cv2.imwrite("./result/"+"旋转 "+str(rotationAngle*90)+" 度 "+"缩放 "+str(scale)+" 倍"+"的特征点mathce图.jpg",img3)
        wmdr_print("goodMatch :" + str(goodMatch))
        if goodMatch is not  None:
            point1 = kp11[goodMatch.queryIdx].pt
            point2 = kp21[goodMatch.trainIdx].pt
            h1, w1 = img1.shape[0:2]
            leftOffset1 = point1[0]
            topOffset1 =  point1[1]
            rightOffset1 = w1 - point1[0]
            bottomOffset1 = h1 - point1[1]
            rect = [int(point2[0] - leftOffset1/scale), int(point2[1] - topOffset1/scale), int(point2[0] + rightOffset1/scale), int(point2[1] + bottomOffset1/scale)]    
            rect = wmdr_correctRect(rect, img1, img2)
        else:
            rect = [0 , 0 , 0 , 0]

    return rect


def main():

    imreadModes = 1 #0 IMREAD_GRAYSCALE  1 IMREAD_COLOR
    mask_logo = cv2.imread("mask_logo.png",imreadModes)          # queryImage
    if mask_logo is None:
        wmdr_print("模板图片不存在！")
        return
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
     #获取提取到的视频帧
    test_frames_dir = "test_frames"
    test_frames_array = wmdr_getSubFiles(test_frames_dir)
    test_frames_count = len(test_frames_array)

    rects = []
    firstFrameImage = None
    for index in range(test_frames_count):
        imreadModes = 1
        wmdr_print("开始使用原色图检测到水印！")
        file_info_array = test_frames_array[index]
        file_dir = file_info_array[0]
        fileName = file_info_array[1]
        tempFrameImage = cv2.imread(file_dir,imreadModes)           # trainImage
        if  tempFrameImage is None:#type(tempFrameImage) == type(None):
            wmdr_print("视频图片不存在！")
        else:
            firstFrameImage = tempFrameImage
            rect =  wmdr_findWaterRect(sift, mask_logo, tempFrameImage, fileName, imreadModes)
            wmdr_print((fileName +"  检测到的水印位置：   ").rjust(40) + str(rect))
            if (rect[3] - rect[1]) > 0:
                rects.append(rect)
            else:
                wmdr_print("原色图未检测到水印！")
                wmdr_print("开始使用灰度图检测到水印！")
                #使用灰度图
                imreadModes = 0
                mask_logo_gray = cv2.imread("mask_logo.png",imreadModes)          # queryImage
                if type(mask_logo_gray) == type(None):
                    wmdr_print("灰度图模板图片不存在！")
                else:
                    tempFrameImageGray = cv2.imread(file_dir,imreadModes)           # trainImage
                    if type(tempFrameImageGray) == type(None):
                        wmdr_print("灰度图视频图片不存在！")
                    else:
                        rect =  wmdr_findWaterRect(sift, mask_logo_gray, tempFrameImageGray, fileName, imreadModes)
                        wmdr_print((fileName +"  检测到的水印位置：   ").rjust(40) + str(rect))
                        if (rect[3] - rect[1]) > 0:
                            rects.append(rect)
    # 校验获取到的规则矩形数组
    rect =  wmdr_getMatchRectForRects(rects, test_frames_count,firstFrameImage)
    print(str(rect)+str(rects))
    if (rect[3] - rect[1]) > 0:
        imreadModes = 1
        result_dir = "./result/"
        if imreadModes == 0:
            result_dir = "./result/grayResult/"
        wmdr_drawRectInImageAndSaveImage(firstFrameImage , rect,result_dir+"标记水印位置图.jpg")


if __name__ == '__main__':
    main()
