import cv2 
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
video_path="/Users/nitthin/Documents/Computer vision/yolov8 obj detection/cat.mp4"
cap=cv2.VideoCapture(video_path)
ret= True
while ret:
    ret,frame=cap.read()
    results = model(frame)
    frame2=results[0].plot()
    cv2.imshow('frame',frame2)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break   
cap.release()
cv2.destroyAllWindows()
