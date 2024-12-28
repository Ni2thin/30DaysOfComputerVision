import cv2 
img_path = "whiteboard.png"
img = cv2.imread(img_path) 
cv2.line(img, (0, 0), (150, 150), (255, 0, 0), 5)
cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), 3)
cv2.circle(img, (100, 100), 50, (0, 0, 255), 2)
cv2.putText(img, "Hello World", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
cv2.imshow("image", img)
cv2.waitKey(0)
