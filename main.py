import cv2
import dlib
from detection.liveness_detection import detect_liveness
from detection.boundary_check import detect_face_boundaries
from detection.artifact_detector import detect_artifacts
from detection.blink_analysis import BlinkAnalyzer
from utils.logger import log_event, view_log_file

PREDICTOR_PATH = "models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

EAR_THRESHOLD = 0.21
CONSEC_FRAMES = 3
blink_analyzer = BlinkAnalyzer(ear_threshold=EAR_THRESHOLD, consecutive_frames=CONSEC_FRAMES)

def run_toolkit():
    print("[INFO] DeepFake Defense Toolkit starting...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not access the camera.")
        return

    print("[INFO] Camera started successfully.")
    previous_frame = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            print(f"[INFO] Detected {len(faces)} face(s).")

            for face in faces:
                shape = predictor(gray, face)
                shape_np = [(p.x, p.y) for p in shape.parts()]

                # Liveness Detection
                ear = detect_liveness(shape_np)

                # Blink Pattern Analysis
                blink_count, blink_detected = blink_analyzer.update(shape_np)
                blink_rate = blink_analyzer.get_blink_rate()
                cv2.putText(frame, f"Blinks: {blink_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
                cv2.putText(frame, f"Blink Rate: {blink_rate:.2f}/sec", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

                if blink_rate < 0.15:
                    cv2.putText(frame, "⚠️ Irregular Blink Pattern", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
                    log_event("Blink Anomaly", f"Low blink rate detected: {blink_rate:.2f}/sec")

                # Boundary Check
                boundary_score, _ = detect_face_boundaries(shape_np, frame)
                cv2.putText(frame, f"Boundary Score: {boundary_score:.0f}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if boundary_score < 500:
                    cv2.putText(frame, "⚠️ Face Boundary Mismatch", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    log_event("Boundary Anomaly", f"Low boundary score: {boundary_score:.0f}")

            # Artifact Detection
            artifact_score, _ = detect_artifacts(frame, previous_frame)
            cv2.putText(frame, f"Artifact Score: {artifact_score:.1f}", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if artifact_score > 35:
                cv2.putText(frame, "⚠️ Video Artifact Detected", (10, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                log_event("Artifact Anomaly", f"High artifact score: {artifact_score:.1f}")

            previous_frame = frame.copy()
            cv2.imshow("DeepFake Defense Toolkit", frame)
            print("[INFO] Frame displayed.")

            # Press 'q' to quit live feed (OPTIONAL—useful fallback)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] 'q' pressed. Exiting Toolkit...")
                break

    except KeyboardInterrupt:
        print("\n[INFO] CTRL+C detected. Stopping toolkit...")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Toolkit stopped. Camera released.")

def view_logs():
    print("\n=== Event Log ===")
    view_log_file()

def main_menu():
    while True:
        print("\n==== DeepFake Defense Toolkit ====")
        print("1. Start Defense Toolkit")
        print("2. View Logs")
        print("3. Exit")
        choice = input("Choose an option: ").strip()

        if choice == '1':
            run_toolkit()
        elif choice == '2':
            view_logs()
        elif choice == '3':
            print("Goodbye.")
            break
        else:
            print("Invalid option. Please choose 1, 2, or 3.")

if __name__ == "__main__":
    main_menu()
