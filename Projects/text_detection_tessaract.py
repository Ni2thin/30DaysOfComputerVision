
import cv2
import pytesseract
from pytesseract import Output
# Set the Tesseract executable path (modify this path as per your installation)
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

# Function to detect text in an image
def detect_text_in_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Preprocess the image using thresholding for better results
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Perform dilation to emphasize the text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # Detect text using pytesseract
    custom_config = r'--oem 3 --psm 6'  # Use OCR Engine Mode 3 and page segmentation mode 6
    text_data = pytesseract.image_to_data(dilated, config=custom_config, output_type=Output.DICT)
    
    # Draw rectangles around detected text
    n_boxes = len(text_data['text'])
    for i in range(n_boxes):
        if int(text_data['conf'][i]) > 50:  # Confidence threshold
            (x, y, w, h) = (text_data['left'][i], text_data['top'][i], 
                            text_data['width'][i], text_data['height'][i])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return image, text_data['text']

# Capture text from an image file
def detect_text_from_image_file(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image from {image_path}.")
        return
    
    # Process the image for text detection
    processed_image, detected_text = detect_text_in_image(image)
    
    # Display the processed image
    cv2.imshow("Text Detection - Image", processed_image)
    print("Detected Text from Image:", " ".join(detected_text))
    
    # Wait for key press to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Capture text from webcam in real-time
def detect_text_from_webcam():
    cap = cv2.VideoCapture(0)  # Use the default webcam
    print("Press 'q' to quit the webcam feed.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break
        
        # Process the frame for text detection
        processed_frame, detected_text = detect_text_in_image(frame)
        
        # Display the processed frame
        cv2.imshow("Text Detection - Webcam", processed_frame)
        print("Detected Text:", " ".join(detected_text))
        
        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Detect text from an image file")
    print("2. Detect text from the webcam (real-time)")
    choice = int(input("Enter your choice (1 or 2): "))
    
    if choice == 1:
        image_path = input("Enter the path to the image file: ")
        detect_text_from_image_file(image_path)
    elif choice == 2:
        detect_text_from_webcam()
    else:
        print("Invalid choice. Please enter 1 or 2.")
