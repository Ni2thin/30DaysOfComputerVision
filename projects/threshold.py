import cv2
img_path="handwritten.png"
img=cv2.imread(img_path)
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
'''ret,thresh=cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
thresh=cv2.blur(thresh,(5,5))
ret,thresh1=cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)'''
adaptive_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 30)
ret, simple_thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY)
cv2.imshow('img', img)
cv2.imshow('adaptive_thresh', adaptive_thresh)
cv2.waitKey(0)