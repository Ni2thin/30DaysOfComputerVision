import cv2 
img_path ="IMG_20240706_103203_555.jpg"
img=cv2.imread(img_path)
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
cv2.imshow("hsv",img_hsv)
cv2.imshow("gray",img_gray)
cv2.imshow("image",img_rgb)
cv2.waitKey(0)