import cv2
from scipy.spatial import distance as dist

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def detect_liveness(shape):
    left_eye = shape[42:48]
    right_eye = shape[36:42]

    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)

    avg_ear = (left_ear + right_ear) / 2.0
    return avg_ear
