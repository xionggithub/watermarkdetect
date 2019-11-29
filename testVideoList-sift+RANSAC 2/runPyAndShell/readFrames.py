import numpy as np
import cv2

cap = cv2.VideoCapture('test.mp4')

fps = cap.get(cv2.CAP_PROP_FPS)

frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

test_frames_dir = "test_frames/"

#读取第0帧
frameIndex = 0
cap.set(cv2.CAP_PROP_POS_FRAMES,frameIndex)  #设置要获取的帧号
ret, frame=cap.read()
if ret==True:
    cv2.imwrite(test_frames_dir+"test_frame"+str(frameIndex)+".jpg",frame)


#读取中间帧
frameIndex = round(frameCount/4)
cap.set(cv2.CAP_PROP_POS_FRAMES,frameIndex)  #设置要获取的帧号
ret, frame=cap.read()
if ret==True:
    cv2.imwrite(test_frames_dir+"test_frame"+str(frameIndex)+".jpg",frame)

#读取中间帧
frameIndex = round(frameCount/2)
cap.set(cv2.CAP_PROP_POS_FRAMES,frameIndex)  #设置要获取的帧号
ret, frame=cap.read()
if ret==True:
    cv2.imwrite(test_frames_dir+"test_frame"+str(frameIndex)+".jpg",frame)


#读取中间帧
frameIndex = round(frameCount*3/4)
cap.set(cv2.CAP_PROP_POS_FRAMES,frameIndex)  #设置要获取的帧号
ret, frame=cap.read()
if ret==True:
    cv2.imwrite(test_frames_dir+"test_frame"+str(frameIndex)+".jpg",frame)

#读取最后帧
frameIndex = frameCount-1
cap.set(cv2.CAP_PROP_POS_FRAMES,frameIndex)  #设置要获取的帧号
ret, frame=cap.read()
if ret==True:
    cv2.imwrite(test_frames_dir+"test_frame"+str(frameIndex)+".jpg",frame)



# while(cap.isOpened()):
#    ret, frame = cap.read()
#    if ret==True:
#        frameIndex += 1
#        cv2.imwrite(test_frames_dir+"test_frame"+str(frameIndex)+".jpg",frame)
#        break
#    else:
#        print("截取失败!")
#        break


cap.release()
cv2.destroyAllWindows()

