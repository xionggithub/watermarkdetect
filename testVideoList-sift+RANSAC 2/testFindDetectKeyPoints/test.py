
import numpy as np
import cv2
from matplotlib import pyplot as plt

#https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html

MIN_MATCH_COUNT = 10

def testPointRotation(rotationAngle, scale):
    w = 200 
    h = 200
    pts = np.float32([ [0,0],[0,h],[w,h],[w,0] ]).reshape(-1,1,2)
    print("原始矩形：\n"+str(pts)+ "\n")

    ang=np.pi*rotationAngle/2
    
    size = [w,h]
    rot_mat = np.array([[np.cos(0), np.sin(0), 0], [-np.sin(0), np.cos(0), 0], [0, 0 , scale]])
    
    if rotationAngle == 0 or rotationAngle == 4:
        size = [w,h]
        rot_mat = np.array([[np.cos(ang), np.sin(ang), 0], [-np.sin(ang), np.cos(ang), 0], [0, 0 , scale]])
    if rotationAngle == 1:
        size = [h,w]
        rot_mat = np.array([[np.cos(ang), np.sin(ang), 0], [-np.sin(ang), np.cos(ang), size[1]], [0, 0 , scale]])
        
    if rotationAngle == 2:
        size = [w,h]
        rot_mat = np.array([[np.cos(ang), np.sin(ang), size[0]], [-np.sin(ang), np.cos(ang), size[1]], [0, 0 , scale]])
        
    if rotationAngle == 3:
        size = [h,w]
        rot_mat = np.array([[np.cos(ang), np.sin(ang), size[0]], [-np.sin(ang), np.cos(ang), 0], [0, 0 , scale]])

    print("转换矩阵：\n"+str(rot_mat)+ "\n")

    dst = cv2.perspectiveTransform(pts,rot_mat)

    for points in dst:
        point = points[0]
        point = [int(point[0]), int(point[1])]
        points[0] = point
    print("转换后原始矩形：\n"+str(dst)+ "\n")



def test_image_crop_withRect(img,cropRect,imreadMode):
    rows = int(cropRect[3] - cropRect[1])
    cols = int(cropRect[2] - cropRect[0])
    imgTmp = np.zeros((rows ,cols, 3),np.uint8)#生成一个空彩色图像
    if imreadMode == 0:
        imgTmp = np.zeros((rows ,cols),np.uint8)#生成一个空彩色图像
    else:
        imgTmp = np.zeros((rows ,cols, 3),np.uint8)#生成一个空彩色图像
    imgTmp[0:imgTmp.shape[0],0:imgTmp.shape[1]] = img[cropRect[1]:cropRect[3], cropRect[0]:cropRect[2]]
    return imgTmp


def test_image_rotation(img,rotationAngle,scale):
    ang=np.pi*rotationAngle/2
    size = [img.shape[1]*scale, img.shape[0]*scale]
    rot_mat = np.array([[np.cos(0)*scale, np.sin(0), 0], [-np.sin(0), np.cos(0)*scale, 0]])
    
    if rotationAngle == 0 or rotationAngle == 4:
        size = [img.shape[1]*scale, img.shape[0]*scale]
        rot_mat = np.array([[np.cos(ang)*scale, np.sin(ang),        0], [-np.sin(ang), np.cos(ang)*scale,       0]])
    
    if rotationAngle == 1:
        size = [img.shape[0]*scale,img.shape[1]*scale]
        rot_mat = np.array([[np.cos(ang), np.sin(ang)*scale,        0], [-np.sin(ang)*scale, np.cos(ang),  size[1]]])
        
    if rotationAngle == 2:
        size = [img.shape[1]*scale,img.shape[0]*scale]
        rot_mat = np.array([[np.cos(ang)*scale, np.sin(ang),  size[0]], [-np.sin(ang), np.cos(ang)*scale,  size[1]]])
        
    if rotationAngle == 3:
        size = [img.shape[0]*scale,img.shape[1]*scale]
        rot_mat = np.array([[np.cos(ang), np.sin(ang)*scale,  size[0]], [-np.sin(ang)*scale, np.cos(ang),       0]])
    
    rot_img = cv2.warpAffine(img, rot_mat, (int(size[0]), int(size[1])))
    
    return rot_img
       

def test():

    imreadMode = 1
    img = cv2.imread("logo_in_scene.jpg",imreadMode)
    rows = img.shape[0]
    cols = img.shape[1]
    crop_image = test_image_crop_withRect(img, [0, 0, cols, int(rows*0.2)],imreadMode)
    plt.figure(1)
    plt.imshow(crop_image, 'gray')
    plt.show()
    return

    # imreadMode = 1
    # angles = [0, 1, 2, 3 , 4]
    # img1 = cv2.imread("logo.png",imreadMode)
    # rot_img = test_image_rotation(img1,angles[1], 0.5)
    # plt.figure(1)
    # plt.imshow(rot_img, 'gray')
    # plt.show()
    # return
    return

# 


def wmdr_sortDistanceRate(distanceRates):
    for i in range(len(distanceRates) - 1): 
        for j in range(len(distanceRates) - i - 1):
            if distanceRates[j] > distanceRates[j + 1]:
                distanceRates[j], distanceRates[j + 1] = distanceRates[j + 1], distanceRates[j]
    return distanceRates

def find_watermark_for_get_good_matches(image1, image2, sift):
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
    print("mindistanceRate :"+str(mindistanceRate) + ", maxdistanceRate :"+str(maxdistanceRate))

    good = []
    distanceRate = mindistanceRate + 0.3*(maxdistanceRate - mindistanceRate)
    for m,n in matches:
        distanceRates.append(m.distance/n.distance)
        if m.distance < distanceRate*n.distance:
            good.append(m)
            print("m.distance :"+str(m.distance) + " n.distance :"+ str(n.distance) +"  "+ str(distanceRate))

    return kp1, kp2, good

def find_watermark_for_not_enough_matches(img1, img2, sift, rotationAngle, scale):
    print("---find_watermark_for_not_enough_matches----")
    kp11, kp21, goodMatches1 = find_watermark_for_get_good_matches(img1, img2, sift)
    kp22, kp12, goodMatches2 = find_watermark_for_get_good_matches(img2, img1, sift)
    # wmdr_drawKeyPointsInImageAndSaveImage(img1, kp11, "./result/"+"旋转 "+str(rotationAngle*90)+" 度 "+"缩放 "+str(scale)+" 倍"+"的logo图.jpg")
    # wmdr_drawKeyPointsInImageAndSaveImage(img2, kp21, "./result/"+"旋转 "+str(rotationAngle*90)+" 度 "+"缩放 "+str(scale)+" 倍"+"的sence图.jpg")
    good = []
    for m1 in goodMatches1:
        for m2 in goodMatches2:
            pt11 = kp11[m1.queryIdx].pt
            pt21 = kp21[m1.trainIdx].pt
            pt22 = kp22[m2.queryIdx].pt
            pt12 = kp12[m2.trainIdx].pt
            if abs(int(pt11[0]) - int(pt12[0])) < 0.001 and abs(int(pt11[1]) - int(pt12[1])) < 0.001 and abs(int(pt21[0]) - int(pt22[0])) < 0.001 and abs(int(pt21[1]) - int(pt22[1])) < 0.001:
                good.append(m1)

    print("find_watermark_for_not_enough_matches find good len :"+str(len(good)))
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
            if abs(angle1 - angle2) < 10.0:
                if m.distance < mindistance:
                    mindistance = m.distance
                    goodMatch = m
                size1 = keyPoint1.size
                size2 = keyPoint2.size
                scale += size1/size2
                print(" scale:  "+ str(size1/size2) + "  angle : "+str(angle1 - angle2))
            
        scale /= len(good) 
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = None, # draw only inliers
                       flags = 2)
        img3 = cv2.drawMatches(img1,kp11,img2,kp21,good,None,**draw_params)
        # cv2.imwrite("./result/"+"旋转 "+str(rotationAngle*90)+" 度 "+"缩放 "+str(scale)+" 倍"+"的特征点mathce图.jpg",img3)

        point1 = kp11[goodMatch.queryIdx].pt
        point2 = kp21[goodMatch.trainIdx].pt
        h1, w1 = img1.shape[0:2]
        leftOffset1 = point1[0]
        topOffset1 =  point1[1]
        rightOffset1 = w1 - point1[0]
        bottomOffset1 = h1 - point1[1]
        rect = [int(point2[0] - leftOffset1/scale), int(point2[1] - topOffset1/scale), int(point2[0] + rightOffset1/scale), int(point2[1] + bottomOffset1/scale)]
        wmdr_drawRectInImageAndSaveImage(img2, rect, "./result/"+"旋转 "+str(rotationAngle*90)+" 度 "+"缩放 "+str(scale)+" 倍"+"的特征点mathce图.jpg")
    
    return


def find_watermark(sift, rotationAngle, figure, scale):
    print("------------------------------"+str(rotationAngle*90)+" 度-----------"+str(scale)+"-------------------------")
    imreadMode = 1 # 0 gray  1 color
    img1 = cv2.imread("logo.png",imreadMode)          # queryImage
    img2 = cv2.imread("logo_in_scene.jpg",imreadMode) # trainImage
    #裁剪图2
    rows = img2.shape[0]
    cols = img2.shape[1]
    # cropRect = [0, 0 , cols, rows]
    # cropRect = [0, int(rows*0.2), cols, 2*int(rows*0.2)]
    cropRect = [0, 0, int(cols*1.0), int(rows*0.5)]
    img2 = test_image_crop_withRect(img2, cropRect,imreadMode)
    #图1 做旋转缩放
    img1 = test_image_rotation(img1,rotationAngle, scale)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # wmdr_drawKeyPointsInImageAndSaveImage(img2, kp2, "./result/"+"旋转 "+str(rotationAngle*90)+" 度 "+"缩放 "+str(scale)+" 倍"+"的特征点.jpg")

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

    print("find matches :"+str(len(matches))+"; kp1 len: " + str(len(kp1))+" kp2 len: " + str(len(kp2)))

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    print("find good :"+str(len(good)))

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape[0:2]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None
        find_watermark_for_not_enough_matches(img1, img2, sift, rotationAngle, scale)

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    plt.figure(figure*90+2*scale)
    plt.imshow(img3, 'gray')
    return

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


def find_sift_kps(sift, rotationAngle, scale):
    imreadMode = 1 # 0 gray  1 color
    img1 = cv2.imread("logo.png",imreadMode)          # queryImage
    img1 = test_image_rotation(img1,rotationAngle, scale)
    kp1, des1 = sift.detectAndCompute(img1,None)
    print("---"+str(rotationAngle*90)+" 度----"+str(scale)+"----")
    print(len(kp1))
    print(len(des1))
    print("-------------------")
    wmdr_drawKeyPointsInImageAndSaveImage(img1, kp1, "./result/"+"旋转 "+str(rotationAngle*90)+" 度 "+"缩放 "+str(scale)+" 倍"+"的特征点.jpg")
    return


def test_water_mark():
    sift = cv2.xfeatures2d.SIFT_create()
    angles = [0, 1, 2, 3 , 4]
    angles = [0]
    scales = [0.5, 1, 2]
    scales = [1]
    for angle in angles:
        for scale in scales:
            find_watermark(sift, angle, angle, scale)
            print("\n")
            print("\n")
    plt.show()

    # sift = cv2.xfeatures2d.SIFT_create()
    # angles = [0, 1, 2, 3 , 4]
    # scales = [0.5, 1, 2]
    # for angle in angles:
    #     for scale in scales:
    #         find_sift_kps(sift, angle, scale)
    return



if __name__ == '__main__':
    # Initiate SIFT detector
    # testM = True
    testM = False
    if testM == True:
        test()
    else:
        test_water_mark()
