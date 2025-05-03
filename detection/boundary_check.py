import cv2
import numpy as np

def detect_face_boundaries(shape_np, frame):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    points = np.array(shape_np, dtype=np.int32)
    cv2.fillConvexPoly(mask, points, 255)

    skin_region = cv2.bitwise_and(frame, frame, mask=mask)
    edge = cv2.Canny(skin_region, 100, 200)

    boundary_score = np.sum(edge) / 255  # Number of edge pixels
    return boundary_score, edge
