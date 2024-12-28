import cv2 
img_path = "birds.jpg"
img = cv2.imread(img_path) 
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
contours , hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for i in contours:
    if cv2.contourArea(i) > 200:
        # cv2.drawContours(img, [i], -1, (0, 0, 255), 3)
        x, y, w, h = cv2.boundingRect(i)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
cv2.imshow("contours", img)
cv2.imshow("image", thresh)
cv2.waitKey(0)
