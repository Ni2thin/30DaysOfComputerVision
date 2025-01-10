import cv2
import torch
from ultralytics import YOLOv10 as YOLO

# Load the YOLOv10 model with the trained weights
model = YOLO('./runs/detect/train/weights/best.pt')  # Path to the trained model weights (last.pt , best.pt)

# Paths to input and output video files
input_video_path = 'video.mp4'
output_video_path = 'output.mp4'

# Open the input video using OpenCV
video_capture = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

# Define codec and create VideoWriter for saving the processed video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Initialize frame counter
frame_count = 0

# Process each frame
while video_capture.isOpened():
    ret, frame = video_capture.read()  # Read a single frame
    if not ret:
        break  # Exit loop if no frame is returned
    
    # Apply YOLOv10 object detection
    results = model(frame)[0]
    
    # Iterate through detections and draw bounding boxes with labels
    for result in results.boxes.data.tolist():  # Format: [x1, y1, x2, y2, conf, class]
        x1, y1, x2, y2, conf, cls = result[:6]
        if conf > 0.5:  # Only consider detections with confidence > 0.5
            label = f'{model.names[int(cls)]} {conf:.2f}'  # Label with class name and confidence
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            
            # Draw label above the bounding box
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Write the processed frame to the output video
    out_video.write(frame)
    
    # Print progress
    frame_count += 1
    print(f'Processed frame {frame_count}/{total_frames}')

# Release resources
video_capture.release()
out_video.release()
cv2.destroyAllWindows()

print(f'Output video saved to {output_video_path}')
