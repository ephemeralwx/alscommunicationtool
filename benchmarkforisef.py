import cv2
import numpy as np
import mediapipe as mp
import time
from screeninfo import get_monitors
import random
from gaze_tracking import GazeTracking
import os

# Define regions for top, left, bottom, and right directions
monitor = get_monitors()[0]
SCREEN_WIDTH = monitor.width
SCREEN_HEIGHT = monitor.height

# Define regions for top, left, bottom, and right directions
TOP_BOUNDS = (SCREEN_WIDTH // 4, SCREEN_WIDTH * 3 // 4, 0, SCREEN_HEIGHT // 4)
LEFT_BOUNDS = (0, SCREEN_WIDTH // 4, SCREEN_HEIGHT // 4, SCREEN_HEIGHT * 3 // 4)
BOTTOM_BOUNDS = (SCREEN_WIDTH // 4, SCREEN_WIDTH * 3 // 4, SCREEN_HEIGHT * 3 // 4, SCREEN_HEIGHT)
RIGHT_BOUNDS = (SCREEN_WIDTH * 3 // 4, SCREEN_WIDTH, SCREEN_HEIGHT // 4, SCREEN_HEIGHT * 3 // 4)

# Define positions for top, left, bottom, and right
top_position = (SCREEN_WIDTH // 2 - 100, 50)        # Top-center
left_position = (50, SCREEN_HEIGHT // 2)           # Left-center
bottom_position = (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT - 50)  # Bottom-center
right_position = (SCREEN_WIDTH - 250, SCREEN_HEIGHT // 2)        # Right-center

direction_timer = {"up": 0, "left": 0, "down": 0, "right": 0}  # Track duration for each direction
AGREEMENT_THRESHOLD = 1  # Seconds of agreement required

def is_in_bounds(x, y, bounds):
    """Checks if a point (x, y) is within given bounds."""
    x_min, x_max, y_min, y_max = bounds
    return x_min <= x <= x_max and y_min <= y <= y_max

# Mediapipe face mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Calibration parameters for screen calibration
calibration_points = [
    (100, 100),
    (SCREEN_WIDTH - 100, 100),
    (SCREEN_WIDTH - 100, SCREEN_HEIGHT - 100),
    (100, SCREEN_HEIGHT - 100),
    (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2),
]
calibration_data = []
current_calibration_index = 0
calibration_complete = False

# Vertical calibration parameters
gaze = GazeTracking()
directions = ["up", "center", "down"]
current_direction_index = 0

# Miscellaneous parameters
dwell_time = 3.2
start_time = None
circle_radius = 50
circle_decrement = 1
phase = "vertical_calibration"  # Switch to "screen_calibration" after vertical calibration
blob_position = None
smoothing_factor = 0.2
blob_radius = 50

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

def draw_blob(frame, position, radius, color):
    """Draw a morphing blob at the given position."""
    noise = random.randint(-5, 5)  # Add slight random variation to radius
    radius += noise
    cv2.circle(frame, position, max(radius, 10), color, -1)  # Ensure radius doesn't go below 10

def draw_progress_bar(frame, position, progress, max_progress, bar_length=200, bar_height=20, color=(0, 255, 0)):
    """
    Draws a progress bar on the frame.

    Args:
        frame (numpy.ndarray): The current video frame.
        position (tuple): Top-left corner of the progress bar (x, y).
        progress (float): Current progress (e.g., seconds elapsed).
        max_progress (float): Maximum progress required (e.g., 2 seconds).
        bar_length (int): Length of the progress bar in pixels.
        bar_height (int): Height of the progress bar in pixels.
        color (tuple): Color of the progress bar in BGR.
    """
    x, y = position
    end_x = x + int((progress / max_progress) * bar_length)
    cv2.rectangle(frame, (x, y), (x + bar_length, y + bar_height), (255, 255, 255), 2)  # Outer rectangle
    cv2.rectangle(frame, (x, y), (end_x, y + bar_height), color, -1)  # Filled progress bar

# Define opposite directions
opposite_directions = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left"
}

# Start webcam capture for calibration
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

# Initialize calibration data
print("[INFO] Starting calibration phase...")
print("[INFO] Please follow the on-screen instructions for calibration.")

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

    # Draw bounding boxes for each region
    cv2.rectangle(frame, (TOP_BOUNDS[0], TOP_BOUNDS[2]), (TOP_BOUNDS[1], TOP_BOUNDS[3]), (0, 255, 0), 2)  # Green for TOP
    cv2.rectangle(frame, (LEFT_BOUNDS[0], LEFT_BOUNDS[2]), (LEFT_BOUNDS[1], LEFT_BOUNDS[3]), (255, 0, 0), 2)  # Blue for LEFT
    cv2.rectangle(frame, (BOTTOM_BOUNDS[0], BOTTOM_BOUNDS[2]), (BOTTOM_BOUNDS[1], BOTTOM_BOUNDS[3]), (0, 0, 255), 2)  # Red for BOTTOM
    cv2.rectangle(frame, (RIGHT_BOUNDS[0], RIGHT_BOUNDS[2]), (RIGHT_BOUNDS[1], RIGHT_BOUNDS[3]), (255, 255, 0), 2)  # Yellow for RIGHT

    # Phase: Vertical Calibration
    if phase == "vertical_calibration":
        direction = directions[current_direction_index]
        cv2.putText(
            frame,
            f"Look {direction} and press 'c'",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(1)
        if key == ord("c"):
            gaze.calibrate_vertical(frame, direction)
            current_direction_index += 1
            if current_direction_index >= len(directions):
                phase = "screen_calibration"
                print("Vertical calibration complete! Starting screen calibration.")
        continue

    # Phase: Screen Calibration
    elif phase == "screen_calibration":
        if current_calibration_index < len(calibration_points):
            target_x, target_y = calibration_points[current_calibration_index]
            cv2.circle(frame, (target_x, target_y), circle_radius, (255, 0, 0), 2)
            circle_radius -= circle_decrement
            if circle_radius <= 0:
                circle_radius = 50

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    face_center_x = (face_landmarks.landmark[454].x + face_landmarks.landmark[234].x) / 2
                    face_center_y = (face_landmarks.landmark[10].y + face_landmarks.landmark[152].y) / 2
                    face_width = abs(face_landmarks.landmark[454].x - face_landmarks.landmark[234].x)
                    face_height = abs(face_landmarks.landmark[10].y - face_landmarks.landmark[152].y)

                    normalized_x, normalized_y = get_eye_position(
                        face_landmarks.landmark, face_center_x, face_center_y, face_width, face_height
                    )

                    if start_time is None:
                        start_time = time.time()
                    if time.time() - start_time > dwell_time:
                        calibration_data.append(((normalized_x, normalized_y), (target_x, target_y)))
                        current_calibration_index += 1
                        start_time = None
        else:
            calibration_complete = True
            phase = "tracking"
            print("Screen calibration complete! Starting tracking phase.")
            cv2.destroyWindow("Calibration")
            break

    # Display the calibration frame in fullscreen
    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Calibration", frame)

    # Exit calibration on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Exiting calibration.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

# Release webcam resources after calibration
cap.release()
cv2.destroyAllWindows()

# Ensure calibration data is sufficient
if not calibration_data:
    print("[ERROR] Calibration data is insufficient. Exiting.")
    exit()

print("[INFO] Starting tracking phase using images from 'pics' folder...")

# Path to the 'pics' folder
pics_folder = 'pics'

# Verify that the 'pics' folder exists
if not os.path.exists(pics_folder):
    print(f"[ERROR] The folder '{pics_folder}' does not exist. Please create it and add jpg images.")
    exit()

# Get list of jpg images in the 'pics' folder
image_files = [f for f in os.listdir(pics_folder) if f.lower().endswith('.jpg')]

if not image_files:
    print(f"[ERROR] No jpg images found in the folder '{pics_folder}'.")
    exit()

# Process each image in the 'pics' folder
for image_file in image_files:
    image_path = os.path.join(pics_folder, image_file)
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"[WARNING] Failed to load image: {image_path}")
        continue

    h, w, _ = frame.shape

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe
    results = face_mesh.process(rgb_frame)

    # Draw bounding boxes for each region
    cv2.rectangle(frame, (TOP_BOUNDS[0], TOP_BOUNDS[2]), (TOP_BOUNDS[1], TOP_BOUNDS[3]), (0, 255, 0), 2)  # Green for TOP
    cv2.rectangle(frame, (LEFT_BOUNDS[0], LEFT_BOUNDS[2]), (LEFT_BOUNDS[1], LEFT_BOUNDS[3]), (255, 0, 0), 2)  # Blue for LEFT
    cv2.rectangle(frame, (BOTTOM_BOUNDS[0], BOTTOM_BOUNDS[2]), (BOTTOM_BOUNDS[1], BOTTOM_BOUNDS[3]), (0, 0, 255), 2)  # Red for BOTTOM
    cv2.rectangle(frame, (RIGHT_BOUNDS[0], RIGHT_BOUNDS[2]), (RIGHT_BOUNDS[1], RIGHT_BOUNDS[3]), (255, 255, 0), 2)  # Yellow for RIGHT

    # Phase: Tracking
    gaze.refresh(frame)
    annotated_frame = gaze.annotated_frame()
    horizontal_ratio = gaze.horizontal_ratio()
    calibrated_vertical_direction = gaze.get_calibrated_vertical_direction()

    # Determine horizontal direction
    horizontal_direction = ""
    if horizontal_ratio is not None:
        if horizontal_ratio <= 0.35:
            horizontal_direction = "left"
        elif horizontal_ratio >= 0.65:
            horizontal_direction = "right"
        else:
            horizontal_direction = "center horizontally"

    # Combine directions for display
    if calibrated_vertical_direction != "uncalibrated" and horizontal_direction:
        text = f"Looking {horizontal_direction} and {calibrated_vertical_direction}"
    elif calibrated_vertical_direction != "uncalibrated":
        text = f"Looking {calibrated_vertical_direction}"
    else:
        text = "Calibrate to detect gaze"

    cv2.putText(annotated_frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Process gaze data with Mediapipe landmarks
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_center_x = (face_landmarks.landmark[454].x + face_landmarks.landmark[234].x) / 2
            face_center_y = (face_landmarks.landmark[10].y + face_landmarks.landmark[152].y) / 2
            face_width = abs(face_landmarks.landmark[454].x - face_landmarks.landmark[234].x)
            face_height = abs(face_landmarks.landmark[10].y - face_landmarks.landmark[152].y)

            normalized_x, normalized_y = get_eye_position(
                face_landmarks.landmark, face_center_x, face_center_y, face_width, face_height
            )

            try:
                screen_x, screen_y = map_gaze_to_screen(normalized_x, normalized_y, calibration_data)

                if blob_position is None:
                    blob_position = (screen_x, screen_y)
                else:
                    blob_position = (
                        int(blob_position[0] * (1 - smoothing_factor) + screen_x * smoothing_factor),
                        int(blob_position[1] * (1 - smoothing_factor) + screen_y * smoothing_factor),
                    )

                draw_blob(annotated_frame, blob_position, blob_radius, (0, 0, 255))

                # Determine regions for agreement logic
                in_top = is_in_bounds(blob_position[0], blob_position[1], TOP_BOUNDS)
                in_left = is_in_bounds(blob_position[0], blob_position[1], LEFT_BOUNDS)
                in_bottom = is_in_bounds(blob_position[0], blob_position[1], BOTTOM_BOUNDS)
                in_right = is_in_bounds(blob_position[0], blob_position[1], RIGHT_BOUNDS)

                # Define gaze directions
                gaze_vertical = calibrated_vertical_direction  # 'up', 'center', 'down'
                gaze_horizontal = horizontal_direction  # 'left', 'center', 'right'

                # Top Region Logic
                if in_top:
                    if gaze_vertical != opposite_directions["up"]:  # Not looking down
                        direction_timer["up"] += 1 / 30  # Assuming 30 FPS or similar timing
                        draw_progress_bar(annotated_frame, (SCREEN_WIDTH // 2 - 100, 10), direction_timer["up"], AGREEMENT_THRESHOLD)
                        if direction_timer["up"] >= AGREEMENT_THRESHOLD:
                            print(f"[IMAGE: {image_file}] Gaze Direction: UP")
                            direction_timer["up"] = 0  # Reset the timer
                    else:
                        direction_timer["up"] = 0  # Reset if looking opposite
                else:
                    direction_timer["up"] = 0  # Reset if not in top region

                # Bottom Region Logic
                if in_bottom:
                    if gaze_vertical != opposite_directions["down"]:  # Not looking up
                        direction_timer["down"] += 1 / 30
                        draw_progress_bar(annotated_frame, (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT - 30), direction_timer["down"], AGREEMENT_THRESHOLD)
                        if direction_timer["down"] >= AGREEMENT_THRESHOLD:
                            print(f"[IMAGE: {image_file}] Gaze Direction: DOWN")
                            direction_timer["down"] = 0
                    else:
                        direction_timer["down"] = 0
                else:
                    direction_timer["down"] = 0

                # Left Region Logic
                if in_left:
                    if gaze_horizontal != opposite_directions["left"]:  # Not looking right
                        direction_timer["left"] += 1 / 30
                        draw_progress_bar(annotated_frame, (10, SCREEN_HEIGHT // 2 - 10), direction_timer["left"], AGREEMENT_THRESHOLD)
                        if direction_timer["left"] >= AGREEMENT_THRESHOLD:
                            print(f"[IMAGE: {image_file}] Gaze Direction: LEFT")
                            direction_timer["left"] = 0
                    else:
                        direction_timer["left"] = 0
                else:
                    direction_timer["left"] = 0

                # Right Region Logic
                if in_right:
                    if gaze_horizontal != opposite_directions["right"]:  # Not looking left
                        direction_timer["right"] += 1 / 30
                        draw_progress_bar(annotated_frame, (SCREEN_WIDTH - 220, SCREEN_HEIGHT // 2 - 10), direction_timer["right"], AGREEMENT_THRESHOLD)
                        if direction_timer["right"] >= AGREEMENT_THRESHOLD:
                            print(f"[IMAGE: {image_file}] Gaze Direction: RIGHT")
                            direction_timer["right"] = 0
                    else:
                        direction_timer["right"] = 0
                else:
                    direction_timer["right"] = 0

                # Determine the final gaze direction based on timers
                directions_detected = []
                for direction, timer in direction_timer.items():
                    if timer >= AGREEMENT_THRESHOLD:
                        directions_detected.append(direction.upper())

                if directions_detected:
                    print(f"[IMAGE: {image_file}] Predicted Gaze Directions: {', '.join(directions_detected)}")
                else:
                    print(f"[IMAGE: {image_file}] Predicted Gaze Directions: NONE")

            except ValueError:
                print(f"[IMAGE: {image_file}] Error in mapping gaze to screen.")
                pass

    # Display the annotated frame (optional)
    cv2.imshow("Tracking", annotated_frame)
    cv2.waitKey(0)  # Wait for key press to move to the next image

# Close all OpenCV windows after processing
cv2.destroyAllWindows()
print("[INFO] Tracking phase completed.")
