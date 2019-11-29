
# -*- coding: utf-8 -*-
import cv2
import numpy
import time

# start = time.time()
# src = cv2.imread('test.jpg')
# mask = cv2.imread('mask.jpg')
# save = numpy.zeros(src.shape, numpy.uint8) #创建一张空图像用于保存

# for row in range(src.shape[0]):
# 	for col in range(src.shape[1]):
# 		for channel in range(src.shape[2]):
# 			if mask[row, col, channel] == 0:
# 				val = 0
# 			else:
# 				reverse_val = 255 - src[row, col, channel]
# 				val = 255 - reverse_val * 256 / mask[row, col, channel]
# 			if val < 0: val = 0

# 			save[row, col, channel] = val

# cv2.imwrite('result.jpg', save)
# end = time.time()
# print("耗时：")
# print(end - start)

# mask = cv2.imread('mask.png')
# for row in range(mask.shape[0]):
# 		for col in range(mask.shape[1]):
# 			for channel in range(mask.shape[2]):
# 				print(mask[row, col, channel])



start = time.time()
imgs = ['test_frame1.jpg','test_frame2.jpg','test_frame3.jpg','test_frame4.jpg','test_frame5.jpg']
mask = None
for imgName in imgs:
	src = cv2.imread(imgName)
	if mask is None:
		mask = numpy.zeros(src.shape, numpy.uint8) #创建一张空图像用于保存
		for row in range(mask.shape[0]):
			for col in range(mask.shape[1]):
				for channel in range(mask.shape[2]):
					mask[row, col, channel] = 255

	save = numpy.zeros(src.shape, numpy.uint8) #创建一张空图像用于保存
	for row in range(src.shape[0]):
		for col in range(src.shape[1]):
			for channel in range(src.shape[2]):
				if mask[row, col, channel] == 0:
					val = 0
				else:
					reverse_val = 255 - src[row, col, channel]
					val = 255 - reverse_val * 256 / mask[row, col, channel]
				if val < 0: val = 0

				save[row, col, channel] = val
	mask = save
	cv2.imwrite(imgName+'result.jpg', mask)


cv2.imwrite('result.jpg', mask)
end = time.time()
print("耗时：")
print(end - start)
















