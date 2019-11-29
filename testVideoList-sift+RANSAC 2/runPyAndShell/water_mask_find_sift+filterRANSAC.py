
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

def wmdr_getKeypointArc(keyPoints):
    arc = 0
    canGetArc = False
    if len(keyPoints) == 3:
        A = [int(keyPoints[0][0]),int(keyPoints[0][1])]
        B = [int(keyPoints[1][0]),int(keyPoints[1][1])]
        C = [int(keyPoints[2][0]),int(keyPoints[2][1])]
        AB = ((A[0] - B[0])*(A[0] - B[0])+(A[1] - B[1])*(A[1] - B[1]))**0.5
        AC = ((A[0] - C[0])*(A[0] - C[0])+(A[1] - C[1])*(A[1] - C[1]))**0.5
        BC = ((B[0] - C[0])*(B[0] - C[0])+(B[1] - C[1])*(B[1] - C[1]))**0.5
        #cosA = (AB*AB + AC*AC - BC*BC ) / 2*AB*AC
        if AB > 0 and AC > 0 and BC > 0:
            cosA = (AB*AB + AC*AC - BC*BC ) / (2*AB*AC)
            if abs(cosA) > 1:
                if cosA < 0:
                    cosA = -1
                else:
                    cosA = 1
            arc = math.acos(cosA) * 180/ math.pi
            wmdr_print("arc :" +str("%.4f" % arc))
            canGetArc = True
        else:
            wmdr_print("A: "+str(A[0])+" "+str(A[1])+" B: "+str(B[0])+" "+str(B[1])+" C: "+str(C[0])+" "+str(C[1]))
            wmdr_print("AB: "+str(AB)+" AC: "+str(AC)+" BC: "+str(BC))
            canGetArc = False
            arc = 0
    else:
        canGetArc = False
        arc = 0

    return arc, canGetArc

def wmdr_ifDstIsRight(dst):
    # 如果定位的位置不是个矩形说定位的是错的
    isRight = True
    if not len(dst) == 4:
        isRight = False
        return isRight

    arc1, canGetArc1 = wmdr_getKeypointArc([dst[0][0],dst[1][0],dst[3][0]])
    arc2, canGetArc2 = wmdr_getKeypointArc([dst[1][0],dst[2][0],dst[0][0]])
    arc3, canGetArc3 = wmdr_getKeypointArc([dst[2][0],dst[3][0],dst[1][0]])
    arc4, canGetArc4 = wmdr_getKeypointArc([dst[3][0],dst[0][0],dst[2][0]])
    totalArc = arc1 + arc2 + arc3 + arc4

    if abs(arc1 - 90) < 10 and abs(arc2 - 90) < 10 and abs(arc3 - 90) < 10 and abs(arc4 - 90) < 10 and abs(totalArc - 360) < 5:
        isRight = True
    else:
        isRight = False

    return isRight



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

# 从目标点获取水印的矩形
def wmdr_getWaterRect(pts, dst, img1, img2):
    rect = [0, 0 , 0 , 0]
    #校验给定的目标点是否能生成一个规则矩形
    isRight = wmdr_ifDstIsRight(dst)
    if isRight == False:
        return rect
    rows = []
    cols = []
    for array in dst:
        rows.append(array[0][1])
        cols.append(array[0][0])
    minRow = min(rows)
    maxRow = max(rows)
    minCol = min(cols)
    maxCol = max(cols)
    rect = [int(round(minCol-0.5)), int(round(minRow-0.5)), int(round(maxCol+0.5)), int(round(maxRow+0.5))]
    rect = wmdr_correctRect(rect, img1, img2)
    return rect


# 从获取到的规则矩形中统计分类 判断生成的矩形正确性
def wmdr_getMatchRectForRects(rects,frames_count):
    findCount  =  len(rects)
    if findCount <= 1:
        rect = [0,0,0,0]
        if findCount > 0:
            rect = rects[0]
            print("检测到的水印正确率大概为："+str(50+int(findCount*(50.0/frames_count)))+" %")
        return rect

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
    if minDistance >= 5:
        return [0, 0 , 0 ,0]
    else:
        print("检测到的水印正确率大概为："+str(50+int(findCount*(50.0/frames_count)))+" %")

    return rects[index]

def wmdr_printMatch(kp1, kp2, matches):
    for m, n in matches:        
        print(str("%.4f" % m.distance) + "  "+str("%.4f" % n.distance) + "  rate: " +str("%.4f" %  (m.distance/n.distance)))

def wmdr_sortDistanceRate(distanceRates):
    for i in range(len(distanceRates) - 1): 
        for j in range(len(distanceRates) - i - 1):
            if distanceRates[j] > distanceRates[j + 1]:
                distanceRates[j], distanceRates[j + 1] = distanceRates[j + 1], distanceRates[j]
    return distanceRates


def test_draw_trainkp(trainKps, matches, trainImage, fileName, imreadModes):
    keyPoints = []
    for m in matches:
        keyPoints.append(trainKps[m.trainIdx])
    # minDistance = 10000
    # for m in matches:
    #     if m.distance < minDistance:
    #         minDistance = m.distance
    #         keyPoints= [trainKps[m.trainIdx]]
    imageWritePath = fileName+"关键点.jpg"
    if imreadModes == 0:
        imageWritePath = fileName+"关键点gray.jpg"
    wmdr_drawKeyPointsInImageAndSaveImage(trainImage, keyPoints, imageWritePath)


def wmdr_findWaterRect(sift, img1, img2, img2Name, imreadModes, needDrawCmpLine):
    #img1 image for qurey 用于查找定位的水印图
    #img2 image for train 用于训练的模型或者定位水印的视频帧或者图片
    rect = [0, 0 , 0 , 0]
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = []
    if not type(des1)== type(None) and  not type(des2)== type(None) and len(des1) >= 2 and len(des2) >= 2:
        matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    #通过distance 0.7 提取good 的match点 
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    # test_draw_trainkp(kp2, good, img2, img2Name, imreadModes)
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        # findHomography 函数解析
        # 第三个参数 Method used to computed a homography matrix. The following methods are possible:
        #0 - a regular method using all the points
        #CV_RANSAC - RANSAC-based robust method
        #CV_LMEDS - Least-Median robust method
        # 第四个参数取值范围在 1 到 10 , 绝一个点对的阈值。原图像的点经过变换后点与目标图像上对应点的误差
        # 超过误差就认为是 outlier
        # 返回值中 H 为变换矩阵。mask是掩模，online的点
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape[0:2]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

        if not type(M) == type(None) :
            #通过变换得到目标点集
            dst = cv2.perspectiveTransform(pts,M)
            # 将检测到的水印圈出来
            img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
            #获取定位到的矩形位置
            rect = wmdr_getWaterRect(pts, dst, img1,img2)
        else:
            rect = [0, 0 , 0 , 0]
            matchesMask = None
    else:
        wmdr_print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        rect = [0, 0 , 0 , 0]
        matchesMask = None

    # 绘制对比连线图
    result_dir = "./result/"
    if imreadModes == 0:
        result_dir = "./result/grayResult/"
    if needDrawCmpLine == True:
        # Finally we draw our inliers (if successfully found the object) or matching keypoints (if failed).
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                           singlePointColor = None,
                           matchesMask = matchesMask, # draw only inliers
                           flags = 2)
        
        wmdr_print(result_dir)
        wmdr_print("find matchesMask :" + str(matchesMask))
        if  matchesMask is not None and len(matchesMask) > 0 and (rect[3] - rect[1]) > 0:
            img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
            cv2.imwrite(result_dir+img2Name+"标记水印位置和特征点连线图.jpg",img3)
    # cv2.imwrite(result_dir+img2Name+"标记水印位置图.jpg",img2)
    return  rect


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
            rect =  wmdr_findWaterRect(sift, mask_logo, tempFrameImage, fileName, imreadModes, False)
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
                        rect =  wmdr_findWaterRect(sift, mask_logo_gray, tempFrameImageGray, fileName, imreadModes, False)
                        wmdr_print((fileName +"  检测到的水印位置：   ").rjust(40) + str(rect))
                        if (rect[3] - rect[1]) > 0:
                            rects.append(rect)
    # 校验获取到的规则矩形数组
    rect =  wmdr_getMatchRectForRects(rects, test_frames_count)
    print(str(rect)+str(rects))
    if (rect[3] - rect[1]) > 0:
        imreadModes = 1
        result_dir = "./result/"
        if imreadModes == 0:
            result_dir = "./result/grayResult/"
        for index in range(test_frames_count):
            imreadModes = 1
            file_info_array = test_frames_array[index]
            file_dir = file_info_array[0]
            fileName = file_info_array[1]
            tempFrameImage = cv2.imread(file_dir,imreadModes) 
            if not type(tempFrameImage) == type(None):
                wmdr_drawRectInImageAndSaveImage(tempFrameImage , rect,result_dir+"标记水印位置图.jpg")
                break

if __name__ == '__main__':
    main()
