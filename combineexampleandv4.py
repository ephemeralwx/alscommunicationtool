import cv2
import numpy as np
import mediapipe as mp
import time
from screeninfo import get_monitors
import random
from gaze_tracking import GazeTracking
import tkinter as tk
from tkinter import Button, Label
from threading import Thread
import collections
import contextlib
import os
import wave
import pyaudio
import webrtcvad
import whisper
import openai
import pyttsx3

# Define regions for top, left, bottom, and right directions
monitor = get_monitors()[0]
SCREEN_WIDTH = monitor.width
SCREEN_HEIGHT = monitor.height

# Define regions for top, left, bottom, and right directions
TOP_BOUNDS = (SCREEN_WIDTH // 4, SCREEN_WIDTH * 3 // 4, 0, SCREEN_HEIGHT // 4)
LEFT_BOUNDS = (0, SCREEN_WIDTH // 4, SCREEN_HEIGHT // 4, SCREEN_HEIGHT * 3 // 4)
BOTTOM_BOUNDS = (SCREEN_WIDTH // 4, SCREEN_WIDTH * 3 // 4, SCREEN_HEIGHT * 3 // 4, SCREEN_HEIGHT)
RIGHT_BOUNDS = (SCREEN_WIDTH * 3 // 4, SCREEN_WIDTH, SCREEN_HEIGHT // 4, SCREEN_HEIGHT * 3 // 4)

arrayOfResponses = []
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1              # Mono audio
RATE = 16000              # Sampling rate
CHUNK_DURATION_MS = 30    # Duration of a chunk in milliseconds
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)  # Chunk size
SILENCE_THRESHOLD = 3     # Silence threshold in seconds
RECORDING_TIMEOUT = 60    # Maximum recording duration in seconds
monitor = get_monitors()[0]
SCREEN_WIDTH = monitor.width
SCREEN_HEIGHT = monitor.height

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

class AudioRecorder:
    def __init__(self):
        self.vad = webrtcvad.Vad(1)  # Set aggressiveness mode
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=FORMAT, channels=CHANNELS,
                                      rate=RATE, input=True,
                                      frames_per_buffer=CHUNK_SIZE)
        self.frames = []
        self.silence_start = None
        self.start_time = time.time()

    def record(self):
        print("Listening for speech...")
        while True:
            chunk = self.stream.read(CHUNK_SIZE)
            active = self.vad.is_speech(chunk, RATE)

            if active:
                self.frames.append(chunk)
                self.silence_start = None  # Reset silence timer
            else:
                if self.silence_start is None:
                    self.silence_start = time.time()
                elif time.time() - self.silence_start > SILENCE_THRESHOLD:
                    print("Silence detected. Stopping recording.")
                    break

            # Stop recording after the timeout
            if time.time() - self.start_time > RECORDING_TIMEOUT:
                print("Recording timeout reached. Stopping recording.")
                break

        self.save_audio()

    def save_audio(self):
        with contextlib.closing(wave.open("output.wav", 'wb')) as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(self.frames))
        print("Audio saved as 'output.wav'.")

    def terminate(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

def transcribe_audio(filename):
    """Transcribes the given audio file using OpenAI's Whisper model."""
    model = whisper.load_model("tiny")
    result = model.transcribe(filename)
    return result["text"]

def transcription_phase():
    """Records audio, transcribes it, and returns the text."""
    print("[INFO] Starting transcription phase...")
    context = None
    recorder = AudioRecorder()
    try:
        recorder.record()  # Record audio
        context = transcribe_audio("output.wav")  # Transcribe audio
        print("[INFO] Transcription completed:")
        print(context)
    except KeyboardInterrupt:
        print("[INFO] Transcription interrupted by user.")
    finally:
        recorder.terminate()  # Ensure resources are cleaned up
    return context

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
calibration_phase_flag = True
directions = ["up", "center", "down"]
current_direction_index = 0

# Miscellaneous parameters
dwell_time = 3.2
start_time = None
buffer_start_time = None
post_calibration_buffer = 0.3
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

# Start webcam capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

# Transcription phase
context = transcription_phase()
print(f"Transcription context: {context}")  # Print or use the context as needed

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define the model and prompt
model = "gpt-4o-mini"

# Create a chat completion
for i in range(4):
    chat_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Given this question, {context}, Give me one example of what someone might answer with. Only say the one actual answer. IE, Do not say, How about sushi, just say sushi. Do not say pizza sounds great, just say pizza. Vary your answers. In your answer do not say any of these items: {arrayOfResponses}. If the question is how are you feeling, don't respond with im feeling fine or im feeling anxious, just say fine/anxious. "}                    
        ]
    )
    arrayOfResponses.append(chat_response.choices[0].message.content.strip())

# Extract and print the assistant's reply
print(arrayOfResponses)

# Proceed to Eye Gaze Tracking
print("[INFO] Starting Eye Gaze Tracker...")
print("[INFO] Waiting 2 seconds before calibration...")
time.sleep(2)

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
        cv2.putText(
            frame,
            f"Look {directions[current_direction_index]} and press 'c'",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Eye Gaze Tracker", frame)
        key = cv2.waitKey(1)
        if key == ord("c"):
            gaze.calibrate_vertical(frame, directions[current_direction_index])
            current_direction_index += 1
            if current_direction_index >= len(directions):
                phase = "screen_calibration"
                print("Vertical calibration complete!")
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
            print("Screen calibration complete!")
            continue

    # Phase: Tracking
    elif phase == "tracking":
        gaze.refresh(frame)
        frame = gaze.annotated_frame()
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

        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display arrayOfResponses on screen
        if len(arrayOfResponses) == 4:  # Ensure there are 4 responses
            cv2.putText(frame, arrayOfResponses[0], top_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, arrayOfResponses[1], left_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, arrayOfResponses[2], bottom_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, arrayOfResponses[3], right_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

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


                if gaze.frame_landmarks:
            # Get image dimensions
                    h, w, _ = frame.shape

                    # Extract outer_corner and inner_corner for the left eye
                    outer_corner = (
                        int(face_landmarks.landmark[33].x * w),
                        int(face_landmarks.landmark[33].y * h)
                    )  # Left eye outer corner

                    inner_corner = (
                        int(face_landmarks.landmark[133].x * w),
                        int(face_landmarks.landmark[133].y * h)
                    )  # Left eye inner corner

                    # Calculate eye_width
                    eye_width = abs(outer_corner[0] - inner_corner[0])

                    # Get self.left_pupil[0] from GazeTracking
                    left_pupil_x = gaze.left_pupil[0] if gaze.left_pupil else None

                    # Extract lower_lid
                    lower_lid = (
                        int(face_landmarks.landmark[145].x * w),
                        int(face_landmarks.landmark[145].y * h)
                    )  # Lower eyelid

                    # Get iris_center from GazeTracking
                    iris_center = gaze.left_pupil

                    # Calculate eye_height
                    upper_lid = (
                        int(face_landmarks.landmark[159].x * w),
                        int(face_landmarks.landmark[159].y * h)
                    )  # Upper eyelid

                    eye_height = abs(upper_lid[1] - lower_lid[1])

                    # Get left_eye_vertical[0] (y-coordinate of upper eyelid)
                    left_eye_vertical_y = upper_lid[1]

                    # Get normalized_distance from GazeTracking
                    normalized_distance = (
                        gaze.normalized_pupil_to_bottom_distance()["left"]
                        if gaze.normalized_pupil_to_bottom_distance()
                        else None
                    )

                    # Print the extracted values
                    print("outer_corner:", outer_corner)
                    print("inner_corner:", inner_corner)
                    print("eye_width:", eye_width)
                    print("self.left_pupil[0]:", left_pupil_x)
                    print("lower_lid:", lower_lid)
                    print("iris_center:", iris_center)
                    print("eye_height:", eye_height)
                    print("left_eye_vertical[0]:", left_eye_vertical_y)
                    print("normalized_distance:", normalized_distance)

                        # Print coordinates of specific landmarks
                    landmark_468 = face_landmarks.landmark[468]
                    landmark_159 = face_landmarks.landmark[159]
                    landmark_145 = face_landmarks.landmark[145]

                    print(f"Point 468: (x={landmark_468.x:.4f}, y={landmark_468.y:.4f}, z={landmark_468.z:.4f})")
                    print(f"Point 159: (x={landmark_159.x:.4f}, y={landmark_159.y:.4f}, z={landmark_159.z:.4f})") 
                    print(f"Point 145: (x={landmark_145.x:.4f}, y={landmark_145.y:.4f}, z={landmark_145.z:.4f})")


                try:
                    screen_x, screen_y = map_gaze_to_screen(normalized_x, normalized_y, calibration_data)

                    if blob_position is None:
                        blob_position = (screen_x, screen_y)
                    else:
                        blob_position = (
                            int(blob_position[0] * (1 - smoothing_factor) + screen_x * smoothing_factor),
                            int(blob_position[1] * (1 - smoothing_factor) + screen_y * smoothing_factor),
                        )

                    draw_blob(frame, blob_position, blob_radius, (0, 0, 255))

                   
                    # Determine regions for agreement logic
                    in_top = is_in_bounds(blob_position[0], blob_position[1], TOP_BOUNDS)
                    in_left = is_in_bounds(blob_position[0], blob_position[1], LEFT_BOUNDS)
                    in_bottom = is_in_bounds(blob_position[0], blob_position[1], BOTTOM_BOUNDS)
                    in_right = is_in_bounds(blob_position[0], blob_position[1], RIGHT_BOUNDS)

                    # ==== Modified Agreement Logic Start ====
                    # Define gaze directions
                    gaze_vertical = calibrated_vertical_direction  # 'up', 'center', 'down'
                    gaze_horizontal = horizontal_direction  # 'left', 'center', 'right'

                    # Top Region Logic
                    if in_top:
                        if gaze_vertical != opposite_directions["up"]:  # Not looking down
                            direction_timer["up"] += 1 / 30  # Assuming 30 FPS
                            draw_progress_bar(frame, (SCREEN_WIDTH // 2 - 100, 10), direction_timer["up"], AGREEMENT_THRESHOLD)
                            if direction_timer["up"] >= AGREEMENT_THRESHOLD:
                                print(f"User is looking at the TOP element: {arrayOfResponses[0]}")
                                direction_timer["up"] = 0  # Reset the timer
                                
                                # Define the prompt for top
                                prompt = (
                                    f"Given this question, {context}. "
                                    f"Here is a word that someone answered the question with: {arrayOfResponses[0]}. "
                                    f"Please provide a sample response sentence that uses the one word answer to respond question. For example if the question is how are you feeling and the one word answer is happy, you should output I'm feeling happy. Just respond with the answer short and brief, do not ask anything else. It is supposed to be an answer only. If the question is what do you want for dinner and the one word answer is sushi, do not say 'Sushi sounds great, do you have a specific type of sushi in mind, like rolls or sashimi?' Instead, say, 'Sushi sounds great.' Keep it short and brief and only answer the question as best you can and keep it at that."
                                )
                                print(f"Prompt being sent to ChatGPT: {prompt}")  # Debug

                                # Create a chat completion
                                response = client.chat.completions.create(
                                    model=model,
                                    messages=[
                                        {"role": "system", "content": "You are a helpful assistant."},
                                        {"role": "user", "content": prompt}
                                    ]
                                )

                                # Extract and print the assistant's reply
                                assistant_reply = response.choices[0].message.content
                                print(assistant_reply)
                                engine = pyttsx3.init()
                                engine.say(assistant_reply)
                                engine.runAndWait()
                        else:
                            direction_timer["up"] = 0  # Reset if looking opposite

                    else:
                        direction_timer["up"] = 0  # Reset if not in top region

                    # Bottom Region Logic
                    if in_bottom:
                        if gaze_vertical != opposite_directions["down"]:  # Not looking up
                            direction_timer["down"] += 1 / 30  # Assuming 30 FPS
                            draw_progress_bar(frame, (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT - 30), direction_timer["down"], AGREEMENT_THRESHOLD)
                            if direction_timer["down"] >= AGREEMENT_THRESHOLD:
                                print(f"User is looking at the BOTTOM element: {arrayOfResponses[2]}")
                                direction_timer["down"] = 0
                                
                                # Define the prompt for bottom
                                prompt = (
                                    f"Given this question, {context}. "
                                    f"Here is a word that someone answered the question with: {arrayOfResponses[2]}. "
                                    f"Please provide a sample response sentence that uses the one word answer to respond question. For example if the question is how are you feeling and the one word answer is happy, you should output I'm feeling happy. Just respond with the answer short and brief, do not ask anything else. It is supposed to be an answer only. If the question is what do you want for dinner and the one word answer is sushi, do not say 'Sushi sounds great, do you have a specific type of sushi in mind, like rolls or sashimi?' Instead, say, 'Sushi sounds great.' Keep it short and brief and only answer the question as best you can and keep it at that."
                                )
                                print(f"Prompt being sent to ChatGPT: {prompt}")  # Debug

                                # Create a chat completion
                                response = client.chat.completions.create(
                                    model=model,
                                    messages=[
                                        {"role": "system", "content": "You are a helpful assistant."},
                                        {"role": "user", "content": prompt}
                                    ]
                                )

                                # Extract and print the assistant's reply
                                assistant_reply = response.choices[0].message.content
                                print(assistant_reply)
                                engine = pyttsx3.init()
                                engine.say(assistant_reply)
                                engine.runAndWait()
                        else:
                            direction_timer["down"] = 0  # Reset if looking opposite

                    else:
                        direction_timer["down"] = 0  # Reset if not in bottom region

                    # Left Region Logic
                    if in_left:
                        if gaze_horizontal != opposite_directions["left"]:  # Not looking right
                            direction_timer["left"] += 1 / 30  # Assuming 30 FPS
                            draw_progress_bar(frame, (10, SCREEN_HEIGHT // 2 - 10), direction_timer["left"], AGREEMENT_THRESHOLD)
                            if direction_timer["left"] >= AGREEMENT_THRESHOLD:
                                print(f"User is looking at the LEFT element: {arrayOfResponses[1]}")
                                direction_timer["left"] = 0
                                
                                # Define the prompt for left
                                prompt = (
                                    f"Given this question, {context}. "
                                    f"Here is a word that someone answered the question with: {arrayOfResponses[1]}. "
                                    f"Please provide a sample response sentence that uses the one word answer to respond question. For example if the question is how are you feeling and the one word answer is happy, you should output I'm feeling happy. Just respond with the answer short and brief, do not ask anything else. It is supposed to be an answer only. If the question is what do you want for dinner and the one word answer is sushi, do not say 'Sushi sounds great, do you have a specific type of sushi in mind, like rolls or sashimi?' Instead, say, 'Sushi sounds great.' Keep it short and brief and only answer the question as best you can and keep it at that."
                                )
                                print(f"Prompt being sent to ChatGPT: {prompt}")  # Debug

                                # Create a chat completion
                                response = client.chat.completions.create(
                                    model=model,
                                    messages=[
                                        {"role": "system", "content": "You are a helpful assistant."},
                                        {"role": "user", "content": prompt}
                                    ]
                                )

                                # Extract and print the assistant's reply
                                assistant_reply = response.choices[0].message.content
                                print(assistant_reply)
                                engine = pyttsx3.init()
                                engine.say(assistant_reply)
                                engine.runAndWait()
                        else:
                            direction_timer["left"] = 0  # Reset if looking opposite

                    else:
                        direction_timer["left"] = 0  # Reset if not in left region

                    # Right Region Logic
                    if in_right:
                        if gaze_horizontal != opposite_directions["right"]:  # Not looking left
                            direction_timer["right"] += 1 / 30  # Assuming 30 FPS
                            draw_progress_bar(frame, (SCREEN_WIDTH - 220, SCREEN_HEIGHT // 2 - 10), direction_timer["right"], AGREEMENT_THRESHOLD)
                            if direction_timer["right"] >= AGREEMENT_THRESHOLD:
                                print(f"User is looking at the RIGHT element: {arrayOfResponses[3]}")
                                direction_timer["right"] = 0
                                
                                # Define the prompt for right
                                prompt = (
                                    f"Given this question, {context}. "
                                    f"Here is a word that someone answered the question with: {arrayOfResponses[3]}. "
                                    f"Please provide a sample response sentence that uses the one word answer to respond question. For example if the question is how are you feeling and the one word answer is happy, you should output I'm feeling happy. Just respond with the answer short and brief, do not ask anything else. It is supposed to be an answer only. If the question is what do you want for dinner and the one word answer is sushi, do not say 'Sushi sounds great, do you have a specific type of sushi in mind, like rolls or sashimi?' Instead, say, 'Sushi sounds great.' Keep it short and brief and only answer the question as best you can and keep it at that."
                                )
                                print(f"Prompt being sent to ChatGPT: {prompt}")  # Debug

                                # Create a chat completion
                                response = client.chat.completions.create(
                                    model=model,
                                    messages=[
                                        {"role": "system", "content": "You are a helpful assistant."},
                                        {"role": "user", "content": prompt}
                                    ]
                                )

                                # Extract and print the assistant's reply
                                assistant_reply = response.choices[0].message.content
                                print(assistant_reply)
                                engine = pyttsx3.init()
                                engine.say(assistant_reply)
                                engine.runAndWait()
                        else:
                            direction_timer["right"] = 0  # Reset if looking opposite

                    else:
                        direction_timer["right"] = 0  # Reset if not in right region
                    # ==== Modified Agreement Logic End ====

                except ValueError:
                    pass

    # Display the frame in fullscreen
    cv2.namedWindow("Eye Gaze Tracker", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Eye Gaze Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Eye Gaze Tracker", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        print("[INFO] Exiting program.")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
