import cv2
img_path="noisy.jpg"
image=cv2.imread(img_path)
img_blur=cv2.blur(image,(7,7))
img_gaussian=cv2.GaussianBlur(image,(7,7),5)
img_median=cv2.medianBlur(image,7)  
cv2.imshow("median",img_median)
cv2.imshow("gaussian",img_gaussian)
cv2.imshow("blur",img_blur)
cv2.imshow("image",image)
cv2.waitKey(0)