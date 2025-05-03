from collections import deque
import time

class BlinkAnalyzer:
    def __init__(self, ear_threshold=0.21, consecutive_frames=3):
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.counter = 0
        self.blink_count = 0
        self.blink_times = deque()

    def update(self, landmarks):
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]

        left_ear = self._eye_aspect_ratio(left_eye)
        right_ear = self._eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        blink_detected = False

        if ear < self.ear_threshold:
            self.counter += 1
        else:
            if self.counter >= self.consecutive_frames:
                self.blink_count += 1
                self.blink_times.append(time.time())
                blink_detected = True
            self.counter = 0

        # Remove old blink times (>10 seconds ago)
        current_time = time.time()
        while self.blink_times and current_time - self.blink_times[0] > 10:
            self.blink_times.popleft()

        return self.blink_count, blink_detected

    def get_blink_rate(self):
        return len(self.blink_times) / 10.0  # Blinks per second (10-sec window)

    def _eye_aspect_ratio(self, eye):
        import numpy as np
        from scipy.spatial import distance as dist

        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear
