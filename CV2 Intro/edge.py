import cv2
img_path = "IMG_20240706_103203_555.jpg"
img = cv2.imread(img_path)
edges=cv2.Canny(img,150,100)
eroded=cv2.erode(edges,(3,3),iterations=1)
edges1=cv2.Laplacian(img,cv2.CV_64F)
edges2=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
cv2.imshow("sobel",edges2)
cv2.imshow("laplacian",edges1)
cv2.imshow("edges",eroded)
cv2.imshow("image",img)
cv2.waitKey(0) 
