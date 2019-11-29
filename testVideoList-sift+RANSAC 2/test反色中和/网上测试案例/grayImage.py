import cv2

img = cv2.imread('mask.png')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imwrite('gray_mask.jpg',gray)

