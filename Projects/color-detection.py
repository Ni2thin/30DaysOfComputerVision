import cv2
import numpy as np

# Function to get HSV limits for a given BGR color
def get_limits(color):
    c = np.uint8([[color]])  # BGR values as input
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)  # Convert to HSV

    hue, saturation, value = hsvC[0][0]  # Extract HSV components

    if saturation < 50:  # Low saturation indicates grayscale
        lowerLimit = np.array([0, 0, max(0, value - 40)], dtype=np.uint8)
        upperLimit = np.array([180, 50, min(255, value + 40)], dtype=np.uint8)
    else:
        # Normal case: Handle hue range
        if hue >= 170:  # Upper limit for red hue wrap-around
            lowerLimit = np.array([hue - 10, 50, 50], dtype=np.uint8)
            upperLimit = np.array([180, 255, 255], dtype=np.uint8)
        elif hue <= 10:  # Lower limit for red hue wrap-around
            lowerLimit = np.array([0, 50, 50], dtype=np.uint8)
            upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
        else:  # General case for other colors
            lowerLimit = np.array([hue - 10, 50, 50], dtype=np.uint8)
            upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)

    return lowerLimit, upperLimit


# Define red color in BGR
red_bgr = [255, 255, 255]

# Get HSV limits for red color
lowerLimit, upperLimit = get_limits(red_bgr)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:  # Check if frame was read successfully
        print("Failed to grab frame")
        break

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(frame, (21, 21), 0)

    # Convert frame to HSV
    hsvImage = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Create a mask for red color
    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

    # Perform morphological operations to clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)# Remove noise
    mask = cv2.dilate(mask, kernel, iterations=2)  # Dilate to fill gaps
    mask = cv2.erode(mask, kernel, iterations=2)  # Erode to smooth edges


    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around detected contours
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Set a minimum area threshold to ignore small noise
            x, y, w, h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    flipped_img = cv2.flip(frame, 1) 
    cv2.imshow('frame', flipped_img)       

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
