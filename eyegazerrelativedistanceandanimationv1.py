import cv2
import numpy as np
import mediapipe as mp
import time
from screeninfo import get_monitors

# Mediapipe face mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Automatically detect screen resolution
monitor = get_monitors()[0]
SCREEN_WIDTH = monitor.width
SCREEN_HEIGHT = monitor.height

# Calibration parameters
calibration_points = [
    (100, 100),
    (SCREEN_WIDTH - 100, 100),
    (SCREEN_WIDTH - 100, SCREEN_HEIGHT - 100),
    (100, SCREEN_HEIGHT - 100),
    (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2),
]
calibration_data = []  # Store (normalized eye landmarks, screen point) pairs
current_calibration_index = 0
calibration_complete = False
dwell_time = 2  # Increased dwell time for better accuracy
start_time = None

# Animation parameters
circle_radius = 50  # Initial circle size
circle_decrement = 1  # How much the circle shrinks per frame


def get_eye_position(landmarks, face_center_x, face_center_y, face_width, face_height):
    """Extract normalized eye position (average of both irises)."""
    left_iris = landmarks[468]  # Left iris center landmark
    right_iris = landmarks[473]  # Right iris center landmark
    eye_x = (left_iris.x + right_iris.x) / 2
    eye_y = (left_iris.y + right_iris.y) / 2

    # Normalize based on face dimensions
    normalized_x = (eye_x - face_center_x) / face_width
    normalized_y = (eye_y - face_center_y) / face_height
    return normalized_x, normalized_y


def map_gaze_to_screen(normalized_x, normalized_y, calibration_data):
    """Map normalized eye positions to screen coordinates using calibration."""
    if len(calibration_data) < 3:
        raise ValueError("[ERROR] Insufficient calibration data for mapping.")

    eye_positions = np.array([data[0] for data in calibration_data])
    screen_points = np.array([data[1] for data in calibration_data])

    # Fit a linear regression model (Least Squares) for mapping
    coeffs_x, _, _, _ = np.linalg.lstsq(
        np.c_[eye_positions, np.ones(len(eye_positions))], screen_points[:, 0], rcond=None
    )
    coeffs_y, _, _, _ = np.linalg.lstsq(
        np.c_[eye_positions, np.ones(len(eye_positions))], screen_points[:, 1], rcond=None
    )

    # Predict screen coordinates
    screen_x = normalized_x * coeffs_x[0] + normalized_y * coeffs_x[1] + coeffs_x[2]
    screen_y = normalized_x * coeffs_y[0] + normalized_y * coeffs_y[1] + coeffs_y[2]

    # Ensure the red dot stays within the screen bounds
    screen_x = max(0, min(SCREEN_WIDTH, screen_x))
    screen_y = max(0, min(SCREEN_HEIGHT, screen_y))

    return int(screen_x), int(screen_y)


# Start webcam capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

print("[INFO] Starting Eye Gaze Tracker...")
print("[INFO] Waiting 2 seconds before calibration...")
time.sleep(2)  # Buffer time

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("[ERROR] Failed to capture frame from webcam.")
        break

    # Flip the frame for a mirror-like effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe
    results = face_mesh.process(rgb_frame)

    # Calibration phase
    if not calibration_complete:
        if current_calibration_index < len(calibration_points):
            # Get the current calibration point
            target_x, target_y = calibration_points[current_calibration_index]
            cv2.circle(frame, (target_x, target_y), circle_radius, (255, 0, 0), 2)

            print(f"[CALIBRATION] Target point {current_calibration_index + 1}/{len(calibration_points)} at ({target_x}, {target_y})")

            # Shrink the circle for animation
            circle_radius -= circle_decrement
            if circle_radius <= 0:
                circle_radius = 50  # Reset the circle radius

            # Process eye landmarks
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Calculate face dimensions and center
                    face_center_x = (face_landmarks.landmark[454].x + face_landmarks.landmark[234].x) / 2
                    face_center_y = (face_landmarks.landmark[10].y + face_landmarks.landmark[152].y) / 2
                    face_width = abs(face_landmarks.landmark[454].x - face_landmarks.landmark[234].x)
                    face_height = abs(face_landmarks.landmark[10].y - face_landmarks.landmark[152].y)

                    # Normalize eye position
                    normalized_x, normalized_y = get_eye_position(
                        face_landmarks.landmark, face_center_x, face_center_y, face_width, face_height
                    )

                    print(f"[CALIBRATION] Detected normalized eye position: ({normalized_x:.4f}, {normalized_y:.4f})")

                    # Start dwell timer if gaze is near the target
                    if start_time is None:
                        start_time = time.time()

                    # Check if gaze is stable near the calibration point
                    if time.time() - start_time > dwell_time:
                        print(f"[CALIBRATION] Successfully captured point {current_calibration_index + 1}")
                        calibration_data.append(((normalized_x, normalized_y), (target_x, target_y)))
                        current_calibration_index += 1
                        start_time = None
                        break
        else:
            calibration_complete = True
            print("[INFO] Calibration complete.")
            print("[INFO] Tracking gaze...")

    # Tracking phase
    else:
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Calculate face dimensions and center
                face_center_x = (face_landmarks.landmark[454].x + face_landmarks.landmark[234].x) / 2
                face_center_y = (face_landmarks.landmark[10].y + face_landmarks.landmark[152].y) / 2
                face_width = abs(face_landmarks.landmark[454].x - face_landmarks.landmark[234].x)
                face_height = abs(face_landmarks.landmark[10].y - face_landmarks.landmark[152].y)

                # Normalize eye position
                normalized_x, normalized_y = get_eye_position(
                    face_landmarks.landmark, face_center_x, face_center_y, face_width, face_height
                )

                # Map gaze to screen coordinates
                try:
                    screen_x, screen_y = map_gaze_to_screen(normalized_x, normalized_y, calibration_data)
                    cv2.circle(frame, (screen_x, screen_y), 15, (0, 0, 255), -1)
                    print(f"[TRACKING] Gaze mapped to ({screen_x}, {screen_y})")
                except ValueError as e:
                    print(e)

    # Show the frame
    cv2.imshow("Eye Gaze Tracker", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        print("[INFO] Exiting program.")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
