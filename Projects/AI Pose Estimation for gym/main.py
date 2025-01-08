# 0. Install and Import Dependencies
import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize Mediapipe pose and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 1. Function to Calculate Angles
def calculate_angle(a, b, c):
    """
    Calculate the angle between three points.
    :param a: First point (x, y)
    :param b: Middle point (x, y)
    :param c: Last point (x, y)
    :return: Angle in degrees
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# 2. Video Feed for Pose Detection with Features
cap = cv2.VideoCapture(0)
counter = {"left": 0, "right": 0}  # Counters for left and right arm
stage = {"left": None, "right": None}  # Stages for left and right arm
paused = False  # Pause functionality
prev_frame_time = 0  # For FPS calculation

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        if paused:
            key = cv2.waitKey(10)
            if key & 0xFF == ord('p'):  # Resume
                paused = False
            elif key & 0xFF == ord('q'):  # Quit
                break
            continue

        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from webcam.")
            break

        # Recolor frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the image and get pose detections
        results = pose.process(image)

        # Recolor back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            # Extract landmarks
            landmarks = results.pose_landmarks.landmark

            # Get coordinates for both arms
            joints = {
                "left": {
                    "shoulder": [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
                    "elbow": [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
                    "wrist": [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                },
                "right": {
                    "shoulder": [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
                    "elbow": [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y],
                    "wrist": [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                }
            }

            # Loop through both arms for calculations
            for side in ["left", "right"]:
                shoulder, elbow, wrist = joints[side].values()

                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)

                # Display angle on the video feed
                cv2.putText(image, f"{side.upper()}:{int(angle)}",
                            tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Curl counter logic
                if angle > 160:
                    stage[side] = "down"
                if angle < 30 and stage[side] == "down":
                    stage[side] = "up"
                    counter[side] += 1
                    print(f"{side.capitalize()} Arm Reps: {counter[side]}")

        except Exception as e:
            print("Error:", e)

        # Calculate FPS
        new_frame_time = time.time()
        fps = int(1 / (new_frame_time - prev_frame_time))
        prev_frame_time = new_frame_time

        # Render additional info on the video feed
        cv2.rectangle(image, (0, 0), (400, 150), (245, 117, 16), -1)
        cv2.putText(image, f"LEFT REPS: {counter['left']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"RIGHT REPS: {counter['right']}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"FPS: {fps}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "Press P to Pause, Q to Quit", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Render pose landmarks on the video feed
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # Display the video feed
        cv2.imshow('Enhanced Mediapipe Feed', image)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('p'):  # Pause
            paused = True

# Save reps data to file
with open("reps_data.txt", "w") as f:
    f.write(f"Left Arm Reps: {counter['left']}\n")
    f.write(f"Right Arm Reps: {counter['right']}\n")

cap.release()
cv2.destroyAllWindows()
