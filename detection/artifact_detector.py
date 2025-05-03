import cv2
import numpy as np

def detect_artifacts(current_frame, previous_frame, threshold=35):
    if previous_frame is None:
        return 0, None

    # Convert to grayscale
    gray_curr = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray_prev = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference
    diff = cv2.absdiff(gray_curr, gray_prev)
    mean_diff = np.mean(diff)

    artifact_detected = mean_diff > threshold
    return mean_diff, diff if artifact_detected else None
