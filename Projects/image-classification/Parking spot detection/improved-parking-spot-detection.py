import cv2
import numpy as np
import argparse
import logging
from util import get_parking_spots_bboxes, empty_or_not

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def calc_diff(im1, im2):
    """Calculate the absolute mean difference between two images."""
    return np.abs(np.mean(im1) - np.mean(im2))

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Parking Spot Detection")
    parser.add_argument("--mask", required=True, help="Path to the parking area mask image.")
    parser.add_argument("--video", required=True, help="Path to the video file.")
    parser.add_argument("--step", type=int, default=30, help="Frame processing step size.")
    parser.add_argument("--diff_threshold", type=float, default=0.4, help="Threshold for spot change detection.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load mask and video
    mask = cv2.imread(args.mask, 0)
    video_path = args.video
    step = args.step
    diff_threshold = args.diff_threshold

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logging.error("Unable to open video file.")
        return

    connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    spots = get_parking_spots_bboxes(connected_components)

    logging.info(f"Detected {len(spots)} parking spots.")
    spots_status = [None for _ in spots]
    diffs = [None for _ in spots]

    previous_frame = None
    frame_nmr = 0
    ret = True

    while ret:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_nmr % step == 0 and previous_frame is not None:
            for spot_idx, spot in enumerate(spots):
                x1, y1, w, h = spot
                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                diffs[spot_idx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

            logging.debug(f"Diffs: {diffs}")

        if frame_nmr % step == 0:
            arr_ = (
                range(len(spots)) if previous_frame is None
                else [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > diff_threshold]
            )

            for spot_idx in arr_:
                spot = spots[spot_idx]
                x1, y1, w, h = spot
                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                spots_status[spot_idx] = empty_or_not(spot_crop)

        if frame_nmr % step == 0:
            previous_frame = frame.copy()

        for spot_idx, spot in enumerate(spots):
            spot_status = spots_status[spot_idx]
            x1, y1, w, h = spots[spot_idx]
            color = (0, 255, 0) if spot_status else (0, 0, 255)
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

        cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
        available_spots = sum(spot is True for spot in spots_status)
        cv2.putText(frame, f'Available spots: {available_spots} / {len(spots)}',
                    (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.namedWindow('Parking Spot Detection', cv2.WINDOW_NORMAL)
        cv2.imshow('Parking Spot Detection', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        frame_nmr += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
