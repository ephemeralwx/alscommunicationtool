import cv2
#ATTNETION. BETTER ONE IS ON MAC, ACTUALLY TRAINED ML MODEL ON THERE.
import numpy as np
import mediapipe as mp
import time
from screeninfo import get_monitors
import pyttsx3
import math
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from filterpy.kalman import KalmanFilter

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Define screen dimensions
monitor = get_monitors()[0]
SCREEN_WIDTH = monitor.width
SCREEN_HEIGHT = monitor.height

# Define regions for top and bottom directions
TOP_BOUNDS = (SCREEN_WIDTH // 4, SCREEN_WIDTH * 3 // 4, 0, SCREEN_HEIGHT // 4)
BOTTOM_BOUNDS = (SCREEN_WIDTH // 4, SCREEN_WIDTH * 3 // 4, SCREEN_HEIGHT * 3 // 4, SCREEN_HEIGHT)

# Define fixed left and right quarter bounds
LEFT_QUARTER_BOUNDS = (0, SCREEN_WIDTH // 4, 0, SCREEN_HEIGHT)  # Left Quarter (Leftmost 25%)
RIGHT_QUARTER_BOUNDS = (SCREEN_WIDTH * 3 // 4, SCREEN_WIDTH, 0, SCREEN_HEIGHT)  # Right Quarter (Rightmost 25%)

# Define calibration points at the four corners and center of the screen
calibration_points = [
    (100, 100),  # Top-Left
    (SCREEN_WIDTH - 100, 100),  # Top-Right
    (SCREEN_WIDTH - 100, SCREEN_HEIGHT - 100),  # Bottom-Right
    (100, SCREEN_HEIGHT - 100),  # Bottom-Left
    (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)  # Center
]

calibration_data = []
current_calibration_index = 0
calibration_complete = False
CALIBRATION_HOLD_TIME = 0.5  # seconds
CALIBRATION_SAMPLE_COUNT = 50  # Increased sample count for better accuracy

# Mediapipe face mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Eye tracking parameters
SMOOTHING_FACTOR = 0.3
blob_position = None

# Gesture detection parameters
GESTURE_HOLD_TIME = 0.5  # seconds
gesture_timers = {
    "selection": 0,
}

# Drawing parameters
current_selection = []
selected_key = ""
drawn_text = ""

# Initialize letter lists
full_letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
current_letters = full_letters.copy()

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

# Window Initialization
cv2.namedWindow("Eye Keyboard", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Eye Keyboard", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def speak(text):
    """Uses pyttsx3 to speak the given text."""
    engine.say(text)
    engine.runAndWait()

def calibrate_gaze(landmarks, calibration_point):
    """Calibrates the gaze by mapping eye positions to screen coordinates."""
    # Access the landmarks to determine face metrics
    face_center_x = (landmarks[454].x + landmarks[234].x) / 2
    face_center_y = (landmarks[10].y + landmarks[152].y) / 2
    face_width = abs(landmarks[454].x - landmarks[234].x)
    face_height = abs(landmarks[10].y - landmarks[152].y)

    # Extract iris landmarks for higher accuracy
    left_iris = landmarks[468]  # Left iris center
    right_iris = landmarks[473]  # Right iris center

    # Compute the average iris position
    iris_x = (left_iris.x + right_iris.x) / 2
    iris_y = (left_iris.y + right_iris.y) / 2

    # Normalize based on face dimensions
    normalized_x = (iris_x - face_center_x) / face_width
    normalized_y = (iris_y - face_center_y) / face_height

    # Optionally, include head pose data here if available
    # For simplicity, this example does not include head pose

    # Append normalized iris position and corresponding screen point
    calibration_data.append(((normalized_x, normalized_y), calibration_point))

def map_gaze_to_screen_ransac(normalized_x, normalized_y, model_x, model_y):
    """Maps normalized gaze coordinates to screen coordinates using RANSAC-based polynomial regression."""
    # Predict screen coordinates using the fitted models
    screen_x = model_x.predict([[normalized_x, normalized_y]])[0]
    screen_y = model_y.predict([[normalized_x, normalized_y]])[0]

    # Clamp values to screen boundaries
    screen_x = max(0, min(SCREEN_WIDTH - 1, int(screen_x)))
    screen_y = max(0, min(SCREEN_HEIGHT - 1, int(screen_y)))

    return (screen_x, screen_y)

def train_ransac_models(calibration_data):
    """Trains RANSAC-based polynomial regression models for X and Y mappings."""
    # Prepare data
    X = np.array([data[0] for data in calibration_data])  # normalized_x, normalized_y
    y_x = np.array([data[1][0] for data in calibration_data])  # screen_x
    y_y = np.array([data[1][1] for data in calibration_data])  # screen_y

    # Define polynomial degree
    degree = 2

    # Create polynomial regression pipelines
    model_x = make_pipeline(PolynomialFeatures(degree), RANSACRegressor())
    model_y = make_pipeline(PolynomialFeatures(degree), RANSACRegressor())

    # Fit models
    model_x.fit(X, y_x)
    model_y.fit(X, y_y)

    return model_x, model_y

def draw_blob(frame, position, radius, color):
    """Draws a blob at the given position."""
    cv2.circle(frame, position, radius, color, -1)

def reset_selection():
    """Resets the current selection."""
    global current_selection, selected_key, gesture_timers, current_letters
    current_selection = []
    selected_key = ""
    gesture_timers["selection"] = 0
    current_letters = full_letters.copy()

def display_keyboard(frame, current_letters):
    """
    Displays the hierarchical keyboard based on current letters.

    Args:
        frame (numpy.ndarray): The current video frame.
        current_letters (list): The list of current possible letters.
    """
    num_letters = len(current_letters)
    if num_letters == 0:
        return

    # If only one letter is left, display it
    if num_letters == 1:
        letter = current_letters[0]
        cv2.putText(frame, f"Selected: {letter}", (SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        return

    # Split the current_letters into two halves
    mid = num_letters // 2
    left_letters = current_letters[:mid]
    right_letters = current_letters[mid:]

    # Always use fixed regions: Left Quarter and Right Quarter
    left_region = LEFT_QUARTER_BOUNDS
    right_region = RIGHT_QUARTER_BOUNDS

    # Draw left region
    cv2.rectangle(frame, (left_region[0], left_region[2]), (left_region[1], left_region[3]), (255, 0, 0), 2)
    left_label = ''.join(left_letters) if len(left_letters) <= 10 else ''.join(left_letters[:10]) + '...'
    cv2.putText(frame, left_label,  # Use the actual letters instead of "Left (number)"
                (left_region[0] + 10, left_region[2] + 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Draw right region
    cv2.rectangle(frame, (right_region[0], right_region[2]), (right_region[1], right_region[3]), (0, 255, 0), 2)
    right_label = ''.join(right_letters) if len(right_letters) <= 10 else ''.join(right_letters[:10]) + '...'
    cv2.putText(frame, right_label,  # Use the actual letters instead of "Right (number)"
                (right_region[0] + 10, right_region[2] + 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def draw_progress_bar(frame, position, progress, max_progress, bar_length=150, bar_height=20, color=(0, 255, 0)):
    """
    Draws a progress bar on the frame.

    Args:
        frame (numpy.ndarray): The current video frame.
        position (tuple): Top-left corner of the progress bar (x, y).
        progress (float): Current progress (e.g., seconds elapsed).
        max_progress (float): Maximum progress required (e.g., 0.5 seconds).
        bar_length (int): Length of the progress bar in pixels.
        bar_height (int): Height of the progress bar in pixels.
        color (tuple): Color of the progress bar in BGR.
    """
    x, y = position
    end_x = x + int((progress / max_progress) * bar_length)
    cv2.rectangle(frame, (x, y), (x + bar_length, y + bar_height), (255, 255, 255), 2)  # Outer rectangle
    cv2.rectangle(frame, (x, y), (end_x, y + bar_height), color, -1)  # Filled progress bar

def recognize_selection():
    """Recognizes the selected key based on current_selection."""
    global selected_key, drawn_text, current_letters

    # If only one letter is left, select it
    if len(current_letters) == 1:
        selected_key = current_letters[0]
        drawn_text += selected_key
        speak(f"You selected {selected_key}")
        reset_selection()

# Initialize Kalman Filter for smoothing
kalman = KalmanFilter(dim_x=2, dim_z=2)
kalman.x = np.array([0., 0.])  # initial state
kalman.P = np.eye(2) * 1000.  # initial uncertainty
kalman.F = np.eye(2)  # state transition matrix
kalman.H = np.eye(2)  # Measurement function
kalman.R = np.eye(2) * 5  # measurement uncertainty
kalman.Q = np.eye(2)  # process uncertainty

# Placeholder for RANSAC models
model_x = None
model_y = None

# Main loop
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("[ERROR] Failed to capture frame from webcam.")
        break

    frame = cv2.flip(frame, 1)  # Mirror effect
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Calibration phase
    if not calibration_complete:
        if current_calibration_index < len(calibration_points):
            target_x, target_y = calibration_points[current_calibration_index]
            # Animate the calibration point with pulsating effect
            pulsate_radius = 15 + 5 * math.sin(time.time() * 2)  # Pulsating effect
            cv2.circle(frame, (target_x, target_y), int(pulsate_radius), (0, 255, 0), -1)
            cv2.putText(frame, f"Look at point {current_calibration_index + 1}", 
                        (target_x - 150, target_y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    calibrate_gaze(face_landmarks.landmark, (target_x, target_y))

            # Check if enough calibration samples are collected
            if len(calibration_data) >= (current_calibration_index + 1) * CALIBRATION_SAMPLE_COUNT:
                # Brief pause with animation before moving to next point
                pause_start = time.time()
                while time.time() - pause_start < 0.5:  # 0.5 second pause
                    frame_copy = frame.copy()
                    pulsate_radius = 20 + 5 * math.sin(time.time() * 2)
                    cv2.circle(frame_copy, (target_x, target_y), int(pulsate_radius), (255, 0, 0), -1)
                    cv2.putText(frame_copy, "Calibrating...", 
                                (target_x - 150, target_y + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.imshow("Eye Keyboard", frame_copy)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                current_calibration_index += 1
                time.sleep(0.2)  # Short delay before next point
                if current_calibration_index >= len(calibration_points):
                    # Train RANSAC models with calibration data
                    model_x, model_y = train_ransac_models(calibration_data)
                    calibration_complete = True
                    print("Calibration complete!")
                    speak("Calibration complete!")
        else:
            # Train RANSAC models with calibration data
            if model_x is None or model_y is None:
                model_x, model_y = train_ransac_models(calibration_data)
            calibration_complete = True
            print("Calibration complete!")
            speak("Calibration complete!")
    else:
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Access the landmarks
                landmarks = face_landmarks.landmark
                # Compute face metrics
                face_center_x = (landmarks[454].x + landmarks[234].x) / 2
                face_center_y = (landmarks[10].y + landmarks[152].y) / 2
                face_width = abs(landmarks[454].x - landmarks[234].x)
                face_height = abs(landmarks[10].y - landmarks[152].y)

                # Extract iris landmarks for higher accuracy
                left_iris = landmarks[468]  # Left iris center
                right_iris = landmarks[473]  # Right iris center

                # Compute the average iris position
                iris_x = (left_iris.x + right_iris.x) / 2
                iris_y = (left_iris.y + right_iris.y) / 2

                # Normalize based on face dimensions
                normalized_x = (iris_x - face_center_x) / face_width
                normalized_y = (iris_y - face_center_y) / face_height

                # Map gaze to screen using RANSAC models
                if model_x and model_y:
                    screen_coords = map_gaze_to_screen_ransac(normalized_x, normalized_y, model_x, model_y)
                    if screen_coords:
                        screen_x, screen_y = screen_coords

                        # Apply Kalman filter for smoothing
                        kalman.predict()
                        kalman.update(np.array([screen_x, screen_y]))
                        filtered_x, filtered_y = kalman.x
                        filtered_x = int(filtered_x)
                        filtered_y = int(filtered_y)
                        blob_position = (filtered_x, filtered_y)

                        # Draw the blob
                        draw_blob(frame, blob_position, 20, (0, 0, 255))

                        # Determine which region the gaze is in
                        in_left_quarter = (LEFT_QUARTER_BOUNDS[0] <= filtered_x <= LEFT_QUARTER_BOUNDS[1]) and (LEFT_QUARTER_BOUNDS[2] <= filtered_y <= LEFT_QUARTER_BOUNDS[3])
                        in_right_quarter = (RIGHT_QUARTER_BOUNDS[0] <= filtered_x <= RIGHT_QUARTER_BOUNDS[1]) and (RIGHT_QUARTER_BOUNDS[2] <= filtered_y <= RIGHT_QUARTER_BOUNDS[3])

                        selected_region = None
                        region_color = (0, 0, 0)  # Default color

                        if in_left_quarter:
                            selected_region = "left"
                            region_color = (255, 0, 0)  # Blue for left
                        elif in_right_quarter:
                            selected_region = "right"
                            region_color = (0, 255, 0)  # Green for right

                        # Handle gesture selection
                        if selected_region:
                            gesture_timers["selection"] += 1 / 30  # Assuming ~30 FPS
                            # Draw bounding box
                            if selected_region == "left":
                                zone = LEFT_QUARTER_BOUNDS
                                current_subset = ''.join(current_letters[:len(current_letters)//2])
                            elif selected_region == "right":
                                zone = RIGHT_QUARTER_BOUNDS
                                current_subset = ''.join(current_letters[len(current_letters)//2:])
                            else:
                                zone = (0, 0, 0, 0)  # Invalid zone

                            cv2.rectangle(frame, (zone[0], zone[2]), (zone[1], zone[3]), region_color, 2)
                            # Draw progress bar
                            bar_position = (zone[0], zone[3] + 10)
                            draw_progress_bar(frame, bar_position, gesture_timers["selection"], GESTURE_HOLD_TIME, 
                                              bar_length=150, bar_height=20, color=region_color)

                            if gesture_timers["selection"] >= GESTURE_HOLD_TIME:
                                gesture_timers["selection"] = 0
                                current_selection.append(selected_region)
                                speak(f"{selected_region.capitalize()} quarter selected")

                                # Split the current_letters into two halves
                                mid = len(current_letters) // 2
                                left_half = current_letters[:mid]
                                right_half = current_letters[mid:]

                                # Update current_letters based on selection
                                if selected_region == "left":
                                    current_letters = left_half
                                elif selected_region == "right":
                                    current_letters = right_half

                                # Recognize selection if only one letter is left
                                recognize_selection()
                        else:
                            gesture_timers["selection"] = 0

    # Display the keyboard
    display_keyboard(frame, current_letters)

    # Display the drawn text at the bottom of the screen
    cv2.putText(frame, drawn_text, (50, SCREEN_HEIGHT - 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    # Show the frame
    cv2.imshow("Eye Keyboard", frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("[INFO] Exiting program.")
        speak("Exiting program.")
        break
    elif key == ord('c') and not calibration_complete:
        # Start calibration when 'c' is pressed
        if current_calibration_index < len(calibration_points):
            target_x, target_y = calibration_points[current_calibration_index]
            # Collect calibration data
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    calibrate_gaze(face_landmarks.landmark, (target_x, target_y))
            # Check if enough calibration samples are collected
            if len(calibration_data) >= (current_calibration_index + 1) * CALIBRATION_SAMPLE_COUNT:
                current_calibration_index += 1
                if current_calibration_index >= len(calibration_points):
                    # Train RANSAC models with calibration data
                    model_x, model_y = train_ransac_models(calibration_data)
                    calibration_complete = True
                    print("Calibration complete!")
                    speak("Calibration complete!")

    # Reset selection if user presses 'r'
    if key == ord('r'):
        reset_selection()
        speak("Selection reset.")

# Release resources
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
