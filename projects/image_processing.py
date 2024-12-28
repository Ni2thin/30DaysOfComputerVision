import cv2
image_path = "/Users/nitthin/Documents/Computer vision/output.jpg"
img = cv2.imread(image_path)
if img is None:
    print(f"Error: The file {image_path} does not exist or could not be read.")
else:
    print("Image read successfully.")
    cv2.imshow("Image", img)
    print(img.shape)
    resized_img = cv2.resize(img, (500, 500))
    print(resized_img.shape)
    cv2.imshow("Resized Image", resized_img)
    cropped_img = img[100:300, 200:400]
    cv2.imshow("Cropped Image", cropped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""video_path = "test.mp4"
vid = cv2.VideoCapture(video_path)
if vid is None:
    print(f"Error: The file {video_path} does not exist or could not be read.")
else:
    print("video read successfully.")"""

"""ret=True 
while ret:
    ret,frame=vid.read()
    if ret:
        cv2.imshow("frame",frame)
        cv2.waitKey(40)
    cv2.destroyAllWindows()    """
"""webcam=cv2.VideoCapture(0)
while True:
    ret,frame=webcam.read()
    flipped=cv2.flip(frame,1)
    cv2.imshow("frame",flipped)
    if cv2.waitKey(40) & 0xFF == ord('a'):
        break
cv2.release()    
cv2.destroyAllWindows() """   