import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
from screeninfo import get_monitors
import random
from gaze_tracking import GazeTracking
import contextlib
import os
import wave
import pyaudio
import webrtcvad
import whisper
import openai
import pyttsx3
import io
from PIL import Image

# -----------------------
# Set page config for Streamlit
st.set_page_config(layout="wide")

# Initialize session state variables
if 'phase' not in st.session_state:
    st.session_state.phase = "vertical_calibration"  # start phase
if 'directions' not in st.session_state:
    st.session_state.directions = ["up", "center", "down"]
if 'current_direction_index' not in st.session_state:
    st.session_state.current_direction_index = 0
if 'current_calibration_index' not in st.session_state:
    st.session_state.current_calibration_index = 0
if 'calibration_data' not in st.session_state:
    st.session_state.calibration_data = []
if 'calibration_complete' not in st.session_state:
    st.session_state.calibration_complete = False
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'context' not in st.session_state:
    st.session_state.context = None
if 'arrayOfResponses' not in st.session_state:
    st.session_state.arrayOfResponses = []
if 'blob_position' not in st.session_state:
    st.session_state.blob_position = None
if 'direction_timer' not in st.session_state:
    st.session_state.direction_timer = {"up": 0, "left": 0, "down": 0, "right": 0}

# Add buttons for user actions
col1, col2, col3 = st.columns(3)
with col1:
    c_pressed = st.button("Press 'c'")
with col2:
    q_pressed = st.button("Press 'q' (Quit)")
with col3:
    start_run = st.button("Start/Restart Application")

# If user clicks start/restart, reinitialize some states
if start_run:
    st.session_state.phase = "vertical_calibration"
    st.session_state.current_direction_index = 0
    st.session_state.current_calibration_index = 0
    st.session_state.calibration_data = []
    st.session_state.calibration_complete = False
    st.session_state.start_time = None
    st.session_state.context = None
    st.session_state.arrayOfResponses = []
    st.session_state.blob_position = None
    st.session_state.direction_timer = {"up": 0, "left": 0, "down": 0, "right": 0}

# -------------- Original Code with Modifications --------------

monitor = get_monitors()[0]
SCREEN_WIDTH = monitor.width
SCREEN_HEIGHT = monitor.height

TOP_BOUNDS = (SCREEN_WIDTH // 4, SCREEN_WIDTH * 3 // 4, 0, SCREEN_HEIGHT // 4)
LEFT_BOUNDS = (0, SCREEN_WIDTH // 4, SCREEN_HEIGHT // 4, SCREEN_HEIGHT * 3 // 4)
BOTTOM_BOUNDS = (SCREEN_WIDTH // 4, SCREEN_WIDTH * 3 // 4, SCREEN_HEIGHT * 3 // 4, SCREEN_HEIGHT)
RIGHT_BOUNDS = (SCREEN_WIDTH * 3 // 4, SCREEN_WIDTH, SCREEN_HEIGHT // 4, SCREEN_HEIGHT * 3 // 4)

arrayOfResponses = st.session_state.arrayOfResponses
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_DURATION_MS = 30
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)
SILENCE_THRESHOLD = 3
RECORDING_TIMEOUT = 60
top_position = (SCREEN_WIDTH // 2 - 100, 50)
left_position = (50, SCREEN_HEIGHT // 2)
bottom_position = (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT - 50)
right_position = (SCREEN_WIDTH - 250, SCREEN_HEIGHT // 2)
direction_timer = st.session_state.direction_timer
AGREEMENT_THRESHOLD = 1

def is_in_bounds(x, y, bounds):
    x_min, x_max, y_min, y_max = bounds
    return x_min <= x <= x_max and y_min <= y <= y_max

class AudioRecorder:
    def __init__(self):
        self.vad = webrtcvad.Vad(1)
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
                self.silence_start = None
            else:
                if self.silence_start is None:
                    self.silence_start = time.time()
                elif time.time() - self.silence_start > SILENCE_THRESHOLD:
                    print("Silence detected. Stopping recording.")
                    break

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
    model = whisper.load_model("tiny")
    result = model.transcribe(filename)
    return result["text"]

def transcription_phase():
    print("[INFO] Starting transcription phase...")
    context = None
    recorder = AudioRecorder()
    try:
        recorder.record()
        context = transcribe_audio("output.wav")
        print("[INFO] Transcription completed:")
        print(context)
    except KeyboardInterrupt:
        print("[INFO] Transcription interrupted by user.")
    finally:
        recorder.terminate()
    return context

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

calibration_points = [
    (100, 100),
    (SCREEN_WIDTH - 100, 100),
    (SCREEN_WIDTH - 100, SCREEN_HEIGHT - 100),
    (100, SCREEN_HEIGHT - 100),
    (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2),
]

calibration_data = st.session_state.calibration_data
current_calibration_index = st.session_state.current_calibration_index
calibration_complete = st.session_state.calibration_complete

gaze = GazeTracking()
calibration_phase_flag = True
directions = st.session_state.directions
current_direction_index = st.session_state.current_direction_index

dwell_time = 3.2
start_time = st.session_state.start_time
post_calibration_buffer = 0.3
circle_radius = 50
circle_decrement = 1
phase = st.session_state.phase
blob_position = st.session_state.blob_position
smoothing_factor = 0.2
blob_radius = 50

def get_eye_position(landmarks, face_center_x, face_center_y, face_width, face_height):
    left_iris = landmarks[468]
    right_iris = landmarks[473]
    eye_x = (left_iris.x + right_iris.x) / 2
    eye_y = (left_iris.y + right_iris.y) / 2
    normalized_x = (eye_x - face_center_x) / face_width
    normalized_y = (eye_y - face_center_y) / face_height
    return normalized_x, normalized_y

def map_gaze_to_screen(normalized_x, normalized_y, calibration_data):
    if len(calibration_data) < 3:
        raise ValueError("[ERROR] Insufficient calibration data for mapping.")

    eye_positions = np.array([data[0] for data in calibration_data])
    screen_points = np.array([data[1] for data in calibration_data])

    coeffs_x, _, _, _ = np.linalg.lstsq(
        np.c_[eye_positions, np.ones(len(eye_positions))], screen_points[:, 0], rcond=None
    )
    coeffs_y, _, _, _ = np.linalg.lstsq(
        np.c_[eye_positions, np.ones(len(eye_positions))], screen_points[:, 1], rcond=None
    )

    screen_x = normalized_x * coeffs_x[0] + normalized_y * coeffs_x[1] + coeffs_x[2]
    screen_y = normalized_x * coeffs_y[0] + normalized_y * coeffs_y[1] + coeffs_y[2]

    screen_x = max(0, min(SCREEN_WIDTH, screen_x))
    screen_y = max(0, min(SCREEN_HEIGHT, screen_y))

    return int(screen_x), int(screen_y)

def draw_blob(frame, position, radius, color):
    noise = random.randint(-5, 5)
    radius += noise
    cv2.circle(frame, position, max(radius, 10), color, -1)

# Only run the transcription phase and GPT calls once
if st.session_state.context is None:
    # Transcription phase
    st.session_state.context = transcription_phase()
    context = st.session_state.context
    print(f"Transcription context: {context}")

    client = openai
    client.api_key = os.getenv("OPENAI_API_KEY")

    model = "gpt-4o-mini"
    for i in range(4):
        chat_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Given this question, {context}, Give me one example of what someone might answer with. Only say the one actual answer. IE, Do not say, How about sushi, just say sushi. Do not say pizza sounds great, just say pizza. Vary your answers. In your answer do not say any of these items: {arrayOfResponses}. If the question is how are you feeling, don't respond with im feeling fine or im feeling anxious, just say fine/anxious."}
            ]
        )
        st.session_state.arrayOfResponses.append(chat_response.choices[0].message.content.strip())
    arrayOfResponses = st.session_state.arrayOfResponses
    print(arrayOfResponses)

print("[INFO] Starting Eye Gaze Tracker...")

def draw_progress_bar(frame, position, progress, max_progress, bar_length=200, bar_height=20, color=(0, 255, 0)):
    x, y = position
    end_x = x + int((progress / max_progress) * bar_length)
    cv2.rectangle(frame, (x, y), (x + bar_length, y + bar_height), (255, 255, 255), 2)
    cv2.rectangle(frame, (x, y), (end_x, y + bar_height), color, -1)

opposite_directions = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left"
}

# Start capturing video frames
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

# Main processing loop
frame_placeholder = st.empty()

while True:
    if q_pressed:
        print("[INFO] Exiting program.")
        break

    success, frame = cap.read()
    if not success:
        print("[ERROR] Failed to capture frame from webcam.")
        break

    # Flip frame
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    cv2.rectangle(frame, (TOP_BOUNDS[0], TOP_BOUNDS[2]), (TOP_BOUNDS[1], TOP_BOUNDS[3]), (0, 255, 0), 2)
    cv2.rectangle(frame, (LEFT_BOUNDS[0], LEFT_BOUNDS[2]), (LEFT_BOUNDS[1], LEFT_BOUNDS[3]), (255, 0, 0), 2)
    cv2.rectangle(frame, (BOTTOM_BOUNDS[0], BOTTOM_BOUNDS[2]), (BOTTOM_BOUNDS[1], BOTTOM_BOUNDS[3]), (0, 0, 255), 2)
    cv2.rectangle(frame, (RIGHT_BOUNDS[0], RIGHT_BOUNDS[2]), (RIGHT_BOUNDS[1], RIGHT_BOUNDS[3]), (255, 255, 0), 2)

    # Phase logic
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
        if c_pressed:
            gaze.calibrate_vertical(frame, directions[current_direction_index])
            st.session_state.current_direction_index += 1
            current_direction_index = st.session_state.current_direction_index
            if current_direction_index >= len(directions):
                st.session_state.phase = "screen_calibration"
                phase = "screen_calibration"
                print("Vertical calibration complete!")
        # Display frame
        frame_placeholder.image(frame, channels="BGR")
        continue

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
                        st.session_state.start_time = time.time()
                        start_time = st.session_state.start_time
                    if time.time() - start_time > dwell_time:
                        st.session_state.calibration_data.append(((normalized_x, normalized_y), (target_x, target_y)))
                        calibration_data = st.session_state.calibration_data
                        st.session_state.current_calibration_index += 1
                        current_calibration_index = st.session_state.current_calibration_index
                        st.session_state.start_time = None
                        start_time = None
        else:
            st.session_state.calibration_complete = True
            st.session_state.phase = "tracking"
            phase = "tracking"
            print("Screen calibration complete!")
            frame_placeholder.image(frame, channels="BGR")
            continue

        frame_placeholder.image(frame, channels="BGR")
        # No continue here, let loop iterate

    elif phase == "tracking":
        gaze.refresh(frame)
        frame = gaze.annotated_frame()
        horizontal_ratio = gaze.horizontal_ratio()
        calibrated_vertical_direction = gaze.get_calibrated_vertical_direction()

        horizontal_direction = ""
        if horizontal_ratio is not None:
            if horizontal_ratio <= 0.35:
                horizontal_direction = "left"
            elif horizontal_ratio >= 0.65:
                horizontal_direction = "right"
            else:
                horizontal_direction = "center horizontally"

        if calibrated_vertical_direction != "uncalibrated" and horizontal_direction:
            text = f"Looking {horizontal_direction} and {calibrated_vertical_direction}"
        elif calibrated_vertical_direction != "uncalibrated":
            text = f"Looking {calibrated_vertical_direction}"
        else:
            text = "Calibrate to detect gaze"

        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if len(arrayOfResponses) == 4:
            cv2.putText(frame, arrayOfResponses[0], top_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, arrayOfResponses[1], left_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, arrayOfResponses[2], bottom_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, arrayOfResponses[3], right_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

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
                    # Extract points
                    outer_corner = (
                        int(face_landmarks.landmark[33].x * w),
                        int(face_landmarks.landmark[33].y * h)
                    )
                    inner_corner = (
                        int(face_landmarks.landmark[133].x * w),
                        int(face_landmarks.landmark[133].y * h)
                    )
                    eye_width = abs(outer_corner[0] - inner_corner[0])
                    left_pupil_x = gaze.left_pupil[0] if gaze.left_pupil else None
                    lower_lid = (
                        int(face_landmarks.landmark[145].x * w),
                        int(face_landmarks.landmark[145].y * h)
                    )
                    iris_center = gaze.left_pupil
                    upper_lid = (
                        int(face_landmarks.landmark[159].x * w),
                        int(face_landmarks.landmark[159].y * h)
                    )
                    eye_height = abs(upper_lid[1] - lower_lid[1])
                    left_eye_vertical_y = upper_lid[1]
                    normalized_distance = (gaze.normalized_pupil_to_bottom_distance()["left"]
                                           if gaze.normalized_pupil_to_bottom_distance()
                                           else None)

                    print("outer_corner:", outer_corner)
                    print("inner_corner:", inner_corner)
                    print("eye_width:", eye_width)
                    print("self.left_pupil[0]:", left_pupil_x)
                    print("lower_lid:", lower_lid)
                    print("iris_center:", iris_center)
                    print("eye_height:", eye_height)
                    print("left_eye_vertical[0]:", left_eye_vertical_y)
                    print("normalized_distance:", normalized_distance)

                    landmark_468 = face_landmarks.landmark[468]
                    landmark_159 = face_landmarks.landmark[159]
                    landmark_145 = face_landmarks.landmark[145]

                    print(f"Point 468: (x={landmark_468.x:.4f}, y={landmark_468.y:.4f}, z={landmark_468.z:.4f})")
                    print(f"Point 159: (x={landmark_159.x:.4f}, y={landmark_159.y:.4f}, z={landmark_159.z:.4f})")
                    print(f"Point 145: (x={landmark_145.x:.4f}, y={landmark_145.y:.4f}, z={landmark_145.z:.4f})")

                try:
                    screen_x, screen_y = map_gaze_to_screen(normalized_x, normalized_y, calibration_data)
                    if blob_position is None:
                        st.session_state.blob_position = (screen_x, screen_y)
                        blob_position = st.session_state.blob_position
                    else:
                        st.session_state.blob_position = (
                            int(blob_position[0] * (1 - smoothing_factor) + screen_x * smoothing_factor),
                            int(blob_position[1] * (1 - smoothing_factor) + screen_y * smoothing_factor),
                        )
                        blob_position = st.session_state.blob_position

                    draw_blob(frame, blob_position, blob_radius, (0, 0, 255))

                    in_top = is_in_bounds(blob_position[0], blob_position[1], TOP_BOUNDS)
                    in_left = is_in_bounds(blob_position[0], blob_position[1], LEFT_BOUNDS)
                    in_bottom = is_in_bounds(blob_position[0], blob_position[1], BOTTOM_BOUNDS)
                    in_right = is_in_bounds(blob_position[0], blob_position[1], RIGHT_BOUNDS)

                    gaze_vertical = calibrated_vertical_direction
                    gaze_horizontal = horizontal_direction

                    def respond_and_speak(response_word):
                        # Construct prompt
                        prompt = (
                            f"Given this question, {st.session_state.context}. "
                            f"Here is a word that someone answered the question with: {response_word}. "
                            "Please provide a sample response sentence that uses the one word answer to respond question. "
                            "For example if the question is how are you feeling and the one word answer is happy, "
                            "you should output I'm feeling happy. Just respond with the answer short and brief, "
                            "do not ask anything else. It is supposed to be an answer only. "
                            "If the question is what do you want for dinner and the one word answer is sushi, "
                            "do not say 'Sushi sounds great, do you have a specific type of sushi in mind, like rolls or sashimi?' "
                            "Instead, say, 'Sushi sounds great.' Keep it short and brief and only answer the question "
                            "as best you can and keep it at that."
                        )
                        print(f"Prompt being sent to ChatGPT: {prompt}")

                        response = openai.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": prompt}
                            ]
                        )

                        assistant_reply = response.choices[0].message.content
                        print(assistant_reply)
                        engine = pyttsx3.init()
                        engine.say(assistant_reply)
                        engine.runAndWait()

                    # TOP logic
                    if in_top:
                        if gaze_vertical != opposite_directions["up"]:
                            direction_timer["up"] += 1/30
                            draw_progress_bar(frame, (SCREEN_WIDTH // 2 - 100, 10), direction_timer["up"], AGREEMENT_THRESHOLD)
                            if direction_timer["up"] >= AGREEMENT_THRESHOLD:
                                print(f"User is looking at the TOP element: {arrayOfResponses[0]}")
                                direction_timer["up"] = 0
                                respond_and_speak(arrayOfResponses[0])
                        else:
                            direction_timer["up"] = 0
                    else:
                        direction_timer["up"] = 0

                    # BOTTOM logic
                    if in_bottom:
                        if gaze_vertical != opposite_directions["down"]:
                            direction_timer["down"] += 1/30
                            draw_progress_bar(frame, (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT - 30), direction_timer["down"], AGREEMENT_THRESHOLD)
                            if direction_timer["down"] >= AGREEMENT_THRESHOLD:
                                print(f"User is looking at the BOTTOM element: {arrayOfResponses[2]}")
                                direction_timer["down"] = 0
                                respond_and_speak(arrayOfResponses[2])
                        else:
                            direction_timer["down"] = 0
                    else:
                        direction_timer["down"] = 0

                    # LEFT logic
                    if in_left:
                        if gaze_horizontal != opposite_directions["left"]:
                            direction_timer["left"] += 1/30
                            draw_progress_bar(frame, (10, SCREEN_HEIGHT // 2 - 10), direction_timer["left"], AGREEMENT_THRESHOLD)
                            if direction_timer["left"] >= AGREEMENT_THRESHOLD:
                                print(f"User is looking at the LEFT element: {arrayOfResponses[1]}")
                                direction_timer["left"] = 0
                                respond_and_speak(arrayOfResponses[1])
                        else:
                            direction_timer["left"] = 0
                    else:
                        direction_timer["left"] = 0

                    # RIGHT logic
                    if in_right:
                        if gaze_horizontal != opposite_directions["right"]:
                            direction_timer["right"] += 1/30
                            draw_progress_bar(frame, (SCREEN_WIDTH - 220, SCREEN_HEIGHT // 2 - 10), direction_timer["right"], AGREEMENT_THRESHOLD)
                            if direction_timer["right"] >= AGREEMENT_THRESHOLD:
                                print(f"User is looking at the RIGHT element: {arrayOfResponses[3]}")
                                direction_timer["right"] = 0
                                respond_and_speak(arrayOfResponses[3])
                        else:
                            direction_timer["right"] = 0
                    else:
                        direction_timer["right"] = 0

                except ValueError:
                    pass

        frame_placeholder.image(frame, channels="BGR")
    else:
        # If no recognized phase, just show frame
        frame_placeholder.image(frame, channels="BGR")

cap.release()
