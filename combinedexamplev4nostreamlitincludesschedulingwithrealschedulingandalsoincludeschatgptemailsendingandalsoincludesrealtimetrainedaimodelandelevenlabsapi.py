import cv2
import numpy as np
import mediapipe as mp
import time
from screeninfo import get_monitors
import random
from gaze_tracking import GazeTracking
import tkinter as tk
from tkinter import Button, Label, Entry, StringVar, END, Listbox, Scrollbar, Toplevel, Checkbutton, IntVar, Radiobutton, filedialog
from threading import Thread, Event
import collections
import contextlib
import os
import wave
import pyaudio
import webrtcvad
import whisper
import openai  # Ensure you have the correct OpenAI package installed
import pyttsx3
from datetime import datetime, timedelta
import smtplib
from dotenv import load_dotenv
from email.mime.text import MIMEText
from elevenlabs import ElevenLabs, play
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression  # Import Linear Regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ---------------------------
# Setup Logging
# ---------------------------
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------
# Load Environment Variables
# ---------------------------
load_dotenv()
APP_PASSWORD = os.getenv("APP_PASSWORD")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------------------
# Initialize ElevenLabs Client
# ---------------------------
client_eleven = ElevenLabs(api_key=ELEVEN_API_KEY)

# ---------------------------
# Initialize OpenAI Client and Model
# ---------------------------
openai.api_key = OPENAI_API_KEY
model = "gpt-4"  # Ensure this model exists or replace with a valid one

# ---------------------------
# Initialize Text-to-Speech Engine
# ---------------------------
engine_pyttsx3 = pyttsx3.init()

# ---------------------------
# ==== EMERGENCY EMAIL FEATURE START ====
# Initialize the emergency_email variable
emergency_email = None

def send_email_for_help(emergency_email):
    """
    Sends an email to the emergency email address indicating that assistance is needed.
    """
    subject = "Help Needed"
    body = "The ALS patient has been determined to be in need of assistance. Please check on them immediately."
    sender = "kevinx8017@gmail.com"  # Replace with your sender email
    recipients = [sender, emergency_email]
                  
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender, APP_PASSWORD)
            smtp.sendmail(sender, recipients, msg.as_string())
        logging.info("Email sent successfully.")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")
# ==== EMERGENCY EMAIL FEATURE END ==== 

# ---------------------------
# ==== SCHEDULED QUESTION FEATURE START ====
# ---------------------------
class ScheduledQuestion:
    def __init__(self, question, scheduled_time, recurring):
        """
        Initializes a ScheduledQuestion instance.

        Args:
            question (str): The question text.
            scheduled_time (datetime): The time at which to ask the question.
            recurring (bool): Whether the question should recur daily.
        """
        self.question = question
        self.scheduled_time = scheduled_time
        self.recurring = recurring

    def reschedule(self):
        """
        Reschedules the question to the next day if it's recurring.
        """
        if self.recurring:
            self.scheduled_time += timedelta(days=1)

class Scheduler:
    def __init__(self):
        self.scheduled_questions = []
        self.lock = collections.defaultdict(Thread)
        self.stop_event = Event()

    def add_question(self, scheduled_question):
        self.scheduled_questions.append(scheduled_question)

    def remove_question(self, index):
        if 0 <= index < len(self.scheduled_questions):
            del self.scheduled_questions[index]

    def run(self):
        while not self.stop_event.is_set():
            now = datetime.now()
            for sq in list(self.scheduled_questions):  # Create a copy to avoid modification during iteration
                if sq.scheduled_time <= now:
                    # Trigger the scheduled question
                    trigger_scheduled_question(sq)
                    if sq.recurring:
                        sq.reschedule()
                    else:
                        self.scheduled_questions.remove(sq)
            time.sleep(1)  # Check every second

    def stop(self):
        self.stop_event.set()

def trigger_scheduled_question(sq):
    """
    Triggers the scheduled question: alerts the user, vocalizes the question,
    fetches responses from ChatGPT, and displays them.

    Args:
        sq (ScheduledQuestion): The scheduled question to trigger.
    """
    logging.info("Triggering scheduled question.")
    speak("Alert, time for scheduled question.")

    speak(sq.question)

    # Fetch responses from ChatGPT using the same logic as normal usage
    arrayOfResponses.clear()  # Clear any existing responses
    global context
    context = sq.question
    logging.info(f"Scheduled Question: {context}")

    # Set the flag to indicate a scheduled question is active
    global scheduled_question_active
    scheduled_question_active = True

    for i in range(4):
        try:
            chat_response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Given this question, {context}, Give me one example of what someone might answer with. Only say the one actual answer. IE, Do not say, How about sushi, just say sushi. Do not say pizza sounds great, just say pizza. Vary your answers. In your answer do not say any of these items: {arrayOfResponses}. If the question is how are you feeling, don't respond with im feeling fine or im feeling anxious, just say fine/anxious."}
                ]
            )
            response_text = chat_response.choices[0].message.content.strip()
            arrayOfResponses.append(response_text)
            logging.info(f"Received response {i+1}: {response_text}")
        except Exception as e:
            logging.error(f"ChatGPT API call failed: {e}")
            break

    logging.info(f"All scheduled responses: {arrayOfResponses}")
# ==== SCHEDULED QUESTION FEATURE END ==== 

# ---------------------------
# Define regions for top, left, bottom, and right directions
# ---------------------------
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

# ---------------------------
# ==== Audio Upload and Voice Cloning FEATURE START ====
# ---------------------------
# Initialize the cloned_voice variable
cloned_voice = None

def upload_voice_sample():
    """
    Allows the user to upload a voice sample and clones it using ElevenLabs API.
    """
    global cloned_voice
    file_path = filedialog.askopenfilename(title="Select Voice Sample", filetypes=[("Audio Files", "*.mp3 *.wav")])
    if file_path:
        try:
            logging.info(f"Cloning voice from: {file_path}")
            speak("Cloning voice. Please wait.")
            cloned_voice = client_eleven.clone(
                name="PatientVoice",
                description="Cloned voice of the patient.",
                files=[file_path],  # Path to the voice sample
            )
            speak("Voice cloned successfully.")
            logging.info("Voice cloned successfully.")
        except Exception as e:
            logging.error(f"Failed to clone voice: {e}")
            speak("Failed to clone voice. Please try again.")
# ==== Audio Upload and Voice Cloning FEATURE END ==== 

# ---------------------------
# ==== SCHEDULED QUESTION FEATURE START ====
# ---------------------------
scheduler = Scheduler()
scheduler_thread = Thread(target=scheduler.run, daemon=True)
scheduler_thread.start()

# Initialize the scheduled_question_active flag
scheduled_question_active = False

global context
context = None  # Initialize globally

def open_schedule_window():
    """
    Opens a new window to schedule a question.
    """
    schedule_window = Toplevel(root)
    schedule_window.title("Schedule a Question")

    # Question Text
    Label(schedule_window, text="Question:").grid(row=0, column=0, padx=10, pady=5, sticky='e')
    question_var = StringVar()
    Entry(schedule_window, textvariable=question_var, width=50).grid(row=0, column=1, padx=10, pady=5)

    # Schedule Type
    Label(schedule_window, text="Schedule Type:").grid(row=1, column=0, padx=10, pady=5, sticky='e')
    schedule_type_var = StringVar(value="time")
    Radiobutton(schedule_window, text="Specific Time", variable=schedule_type_var, value="time").grid(row=1, column=1, padx=10, pady=5, sticky='w')
    Radiobutton(schedule_window, text="Countdown Timer (seconds)", variable=schedule_type_var, value="timer").grid(row=2, column=1, padx=10, pady=5, sticky='w')

    # Time Entry
    time_var = StringVar()
    timer_var = StringVar()

    def update_time_fields(*args):
        if schedule_type_var.get() == "time":
            time_entry.grid(row=3, column=1, padx=10, pady=5, sticky='w')
            timer_entry.grid_remove()
        else:
            timer_entry.grid(row=3, column=1, padx=10, pady=5, sticky='w')
            time_entry.grid_remove()

    schedule_type_var.trace_add('write', update_time_fields)

    # Specific Time Inputs
    Label(schedule_window, text="Hour (0-23):").grid(row=3, column=0, padx=10, pady=5, sticky='e')
    time_entry = Entry(schedule_window, textvariable=time_var, width=10)
    time_entry.grid(row=3, column=1, padx=10, pady=5, sticky='w')
    # Remove initially if not needed
    timer_entry = Entry(schedule_window, textvariable=timer_var, width=10)
    timer_entry.grid(row=3, column=1, padx=10, pady=5, sticky='w')
    timer_entry.grid_remove()

    # Recurring Daily
    recurring_var = IntVar()
    Checkbutton(schedule_window, text="Repeat Daily", variable=recurring_var).grid(row=4, column=1, padx=10, pady=5, sticky='w')

    # Add Button
    def add_scheduled_question():
        question = question_var.get().strip()
        if not question:
            logging.error("Question cannot be empty.")
            speak("Question cannot be empty.")
            return

        schedule_type = schedule_type_var.get()
        if schedule_type == "time":
            try:
                hour = int(time_var.get())
                if not (0 <= hour <= 23):
                    raise ValueError
                minute = 0  # You can extend this to allow minutes input
                scheduled_time = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
                if scheduled_time < datetime.now():
                    scheduled_time += timedelta(days=1)
            except ValueError:
                logging.error("Invalid time format.")
                speak("Invalid time format.")
                return
        else:
            try:
                countdown = int(timer_var.get())
                if countdown <= 0:
                    raise ValueError
                scheduled_time = datetime.now() + timedelta(seconds=countdown)
            except ValueError:
                logging.error("Invalid countdown timer format.")
                speak("Invalid countdown timer format.")
                return

        recurring = bool(recurring_var.get())
        sq = ScheduledQuestion(question, scheduled_time, recurring)
        scheduler.add_question(sq)
        logging.info(f"Scheduled question added: '{question}' at {scheduled_time} | Recurring: {recurring}")
        speak("Scheduled question added successfully.")
        schedule_window.destroy()

    Button(schedule_window, text="Add Scheduled Question", command=add_scheduled_question).grid(row=5, column=1, padx=10, pady=10, sticky='e')

def schedule_gui():
    """
    Creates the main GUI for scheduling questions and setting emergency email.
    """
    global root
    root = tk.Tk()
    root.title("Eye Gaze Tracker Scheduler")

    Button(root, text="Schedule a Question", command=open_schedule_window).pack(padx=20, pady=20)

    # ==== EMERGENCY EMAIL FEATURE START ====
    # Emergency Email Section
    Label(root, text="Emergency Email:").pack(padx=10, pady=5)
    email_var = StringVar()
    email_entry = Entry(root, textvariable=email_var, width=50)
    email_entry.pack(padx=10, pady=5)
    Button(root, text="Set Emergency Email", command=lambda: set_emergency_email(email_var, email_entry)).pack(padx=10, pady=5)
    # ==== EMERGENCY EMAIL FEATURE END====

    # ==== VOICE CLONING FEATURE START ====
    # Voice Cloning Section
    Label(root, text="Voice Cloning:").pack(padx=10, pady=5)
    Button(root, text="Upload Voice Sample", command=upload_voice_sample).pack(padx=10, pady=5)
    # ==== VOICE CLONING FEATURE END ====

    # Display Scheduled Questions
    Label(root, text="Scheduled Questions:").pack(padx=10, pady=5)
    scheduled_listbox = Listbox(root, selectmode=tk.SINGLE, width=80)
    scheduled_listbox.pack(padx=10, pady=10)
    scrollbar = Scrollbar(root)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    scheduled_listbox.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=scheduled_listbox.yview)

    def refresh_scheduled_list():
        scheduled_listbox.delete(0, END)
        for idx, sq in enumerate(scheduler.scheduled_questions):
            time_str = sq.scheduled_time.strftime("%Y-%m-%d %H:%M:%S")
            recurring_str = "Yes" if sq.recurring else "No"
            scheduled_listbox.insert(END, f"{idx+1}. Question: '{sq.question}' | Time: {time_str} | Recurring: {recurring_str}")
        root.after(1000, refresh_scheduled_list)  # Refresh every second

    def remove_selected_question():
        selected = scheduled_listbox.curselection()
        if selected:
            index = selected[0]
            scheduler.remove_question(index)
            logging.info(f"Removed scheduled question at index {index+1}.")
            speak("Scheduled question removed successfully.")

    Button(root, text="Remove Selected Question", command=remove_selected_question).pack(padx=10, pady=5)

    refresh_scheduled_list()
    return root
# ==== SCHEDULED QUESTION FEATURE END ==== 

# ---------------------------
# ==== Voice Cloning and TTS FUNCTION START ====
# ---------------------------
def speak(text):
    """
    Speaks the given text using ElevenLabs cloned voice if available,
    else uses pyttsx3.
    """
    if cloned_voice:
        try:
            audio = client_eleven.generate(
                text=text,
                voice=cloned_voice
            )
            play(audio)
        except Exception as e:
            logging.error(f"Failed to generate or play audio with ElevenLabs: {e}")
            # Fallback to pyttsx3
            engine_pyttsx3.say(text)
            engine_pyttsx3.runAndWait()
    else:
        engine_pyttsx3.say(text)
        engine_pyttsx3.runAndWait()
# ==== Voice Cloning and TTS FUNCTION END ==== 

# ---------------------------
# ==== EMERGENCY EMAIL FEATURE START ====
# ---------------------------
def set_emergency_email(email_var, email_entry):
    """
    Sets the emergency email from the GUI input.

    Args:
        email_var (StringVar): The StringVar associated with the email entry.
        email_entry (Entry): The Entry widget for the email.
    """
    global emergency_email
    email = email_var.get().strip()
    if email:
        # Basic email format validation
        if "@" in email and "." in email:
            emergency_email = email
            logging.info(f"Emergency email set to: {emergency_email}")
            speak("Emergency email set successfully.")
            email_entry.delete(0, END)
        else:
            logging.error("Invalid email format.")
            speak("Please enter a valid email address.")
    else:
        logging.error("Email field is empty.")
        speak("Please enter an email address.")
# ==== EMERGENCY EMAIL FEATURE END ==== 

# ---------------------------
# ==== Real-Time AI Training FEATURE START ====
# ---------------------------
def train_models(screen_calibration_data):
    """
    Trains multiple regression models for X and Y screen coordinates with extended features.

    Args:
        screen_calibration_data (list): List of tuples containing normalized features and screen points.

    Returns:
        dict: Dictionary containing trained models.
    """
    logging.info("Starting training of all regression models with extended features...")
    start_time = time.time()
    
    # Prepare data
    X = np.array([data[0] for data in screen_calibration_data])  # Extended normalized features
    y_x = np.array([data[1][0] for data in screen_calibration_data])  # screen_x
    y_y = np.array([data[1][1] for data in screen_calibration_data])  # screen_y

    logging.info(f"Feature matrix shape: {X.shape}")
    logging.info(f"Target vectors shape: X={y_x.shape}, Y={y_y.shape}")

    models_trained = {}

    # 1. Support Vector Regression (SVR)
    logging.info("Training Support Vector Regression (SVR) models...")
    svr_pipeline_x = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1))
    svr_pipeline_y = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1))
    svr_pipeline_x.fit(X, y_x)
    svr_pipeline_y.fit(X, y_y)
    models_trained['SVR_X'] = svr_pipeline_x
    models_trained['SVR_Y'] = svr_pipeline_y
    logging.info("SVR models trained.")

    # 2. Random Forest Regressor (RF)
    logging.info("Training Random Forest Regressor (RF) models...")
    rf_pipeline_x = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42))
    rf_pipeline_y = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42))
    rf_pipeline_x.fit(X, y_x)
    rf_pipeline_y.fit(X, y_y)
    models_trained['RF_X'] = rf_pipeline_x
    models_trained['RF_Y'] = rf_pipeline_y
    logging.info("Random Forest models trained.")

    # 3. Gradient Boosting Regressor (GBR)
    logging.info("Training Gradient Boosting Regressor (GBR) models...")
    gbr_pipeline_x = make_pipeline(StandardScaler(), GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    gbr_pipeline_y = make_pipeline(StandardScaler(), GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    gbr_pipeline_x.fit(X, y_x)
    gbr_pipeline_y.fit(X, y_y)
    models_trained['GBR_X'] = gbr_pipeline_x
    models_trained['GBR_Y'] = gbr_pipeline_y
    logging.info("Gradient Boosting models trained.")

    # 4. K-Nearest Neighbors Regressor (KNN)
    logging.info("Training K-Nearest Neighbors Regressor (KNN) models...")
    knn_pipeline_x = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5))
    knn_pipeline_y = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5))
    knn_pipeline_x.fit(X, y_x)
    knn_pipeline_y.fit(X, y_y)
    models_trained['KNN_X'] = knn_pipeline_x
    models_trained['KNN_Y'] = knn_pipeline_y
    logging.info("K-Nearest Neighbors models trained.")

    # 5. Linear Regression (Least Squares)
    logging.info("Training Linear Regression (Least Squares) models...")
    lr_pipeline_x = make_pipeline(StandardScaler(), LinearRegression())
    lr_pipeline_y = make_pipeline(StandardScaler(), LinearRegression())
    lr_pipeline_x.fit(X, y_x)
    lr_pipeline_y.fit(X, y_y)
    models_trained['LR_X'] = lr_pipeline_x
    models_trained['LR_Y'] = lr_pipeline_y
    logging.info("Linear Regression models trained.")

    end_time = time.time()
    logging.info(f"All regression models trained in {end_time - start_time:.2f} seconds.")

    return models_trained

def map_gaze_to_screen(models, feature_vector):
    """
    Maps normalized gaze coordinates to screen coordinates using all models.

    Args:
        models (dict): Dictionary containing trained models.
        feature_vector (list): Normalized feature vector extracted from facial landmarks.

    Returns:
        dict: Dictionary containing predictions from each model.
    """
    input_features = np.array([feature_vector])

    predictions = {}

    # SVR Predictions
    svr_x_pred = models['SVR_X'].predict(input_features)[0]
    svr_y_pred = models['SVR_Y'].predict(input_features)[0]
    predictions['SVR'] = (int(np.clip(svr_x_pred, 0, SCREEN_WIDTH - 1)),
                           int(np.clip(svr_y_pred, 0, SCREEN_HEIGHT - 1)))

    # Random Forest Predictions
    rf_x_pred = models['RF_X'].predict(input_features)[0]
    rf_y_pred = models['RF_Y'].predict(input_features)[0]
    predictions['RF'] = (int(np.clip(rf_x_pred, 0, SCREEN_WIDTH - 1)),
                         int(np.clip(rf_y_pred, 0, SCREEN_HEIGHT - 1)))

    # Gradient Boosting Predictions
    gbr_x_pred = models['GBR_X'].predict(input_features)[0]
    gbr_y_pred = models['GBR_Y'].predict(input_features)[0]
    predictions['GBR'] = (int(np.clip(gbr_x_pred, 0, SCREEN_WIDTH - 1)),
                          int(np.clip(gbr_y_pred, 0, SCREEN_HEIGHT - 1)))

    # K-Nearest Neighbors Predictions
    knn_x_pred = models['KNN_X'].predict(input_features)[0]
    knn_y_pred = models['KNN_Y'].predict(input_features)[0]
    predictions['KNN'] = (int(np.clip(knn_x_pred, 0, SCREEN_WIDTH - 1)),
                           int(np.clip(knn_y_pred, 0, SCREEN_HEIGHT - 1)))

    # Linear Regression Predictions
    lr_x_pred = models['LR_X'].predict(input_features)[0]
    lr_y_pred = models['LR_Y'].predict(input_features)[0]
    predictions['LR'] = (int(np.clip(lr_x_pred, 0, SCREEN_WIDTH - 1)),
                          int(np.clip(lr_y_pred, 0, SCREEN_HEIGHT - 1)))

    logging.debug(f"Predictions: {predictions}")

    return predictions

def draw_blobs(frame, predictions):
    """
    Draws blobs on the frame based on predictions from all models.

    Args:
        frame (numpy.ndarray): The current video frame.
        predictions (dict): Dictionary containing predictions from each model.
    """
    # Define colors for each model (B, G, R)
    colors = {
        'SVR': (0, 0, 255),    # Red
        'RF': (0, 255, 0),     # Green
        'GBR': (255, 0, 0),    # Blue
        'KNN': (0, 255, 255),  # Yellow
        'LR': (255, 255, 0)    # Cyan
    }

    for model_name, pos in predictions.items():
        if pos:
            cv2.circle(frame, pos, 10, colors[model_name], -1)
            # Label the blobs
            cv2.putText(frame, model_name, (pos[0]+12, pos[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[model_name], 1)

def reset_selection():
    """Resets the current selection."""
    global current_selection, selected_key, direction_timer, current_letters
    current_selection = []
    selected_key = ""
    direction_timer["selection"] = 0
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
    left_region = LEFT_BOUNDS
    right_region = RIGHT_BOUNDS

    # Draw left region
    cv2.rectangle(frame, (left_region[0], left_region[2]), (left_region[1], left_region[3]), (255, 0, 0), 2)
    left_label = ''.join(left_letters) if len(left_letters) <= 10 else ''.join(left_letters[:10]) + '...'
    cv2.putText(frame, left_label, 
                (left_region[0] + 10, left_region[2] + 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Draw right region
    cv2.rectangle(frame, (right_region[0], right_region[2]), (right_region[1], right_region[3]), (0, 255, 0), 2)
    right_label = ''.join(right_letters) if len(right_letters) <= 10 else ''.join(right_letters[:10]) + '...'
    cv2.putText(frame, right_label, 
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
        logging.info(f"Letter selected: {selected_key}")
        reset_selection()

# ---------------------------
# ==== Transcription and Recording FEATURE START ====
# ---------------------------
class AudioRecorder:
    def __init__(self):
        self.format = FORMAT
        self.channels = CHANNELS
        self.rate = RATE
        self.chunk = CHUNK_SIZE
        self.record_seconds = RECORDING_TIMEOUT
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.stream = None
        self.is_recording = False

    def record(self):
        """Starts recording audio."""
        self.stream = self.audio.open(format=self.format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer=self.chunk)
        self.is_recording = True
        logging.info("Recording started.")
        for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
            if not self.is_recording:
                break
            data = self.stream.read(self.chunk)
            self.frames.append(data)
        self.is_recording = False
        logging.info("Recording stopped.")
        self.save_wave_file("output.wav")

    def save_wave_file(self, filename):
        """Saves the recorded audio to a WAV file."""
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        logging.info(f"Audio saved to {filename}.")

    def terminate(self):
        """Terminates the audio stream and PyAudio."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        logging.info("Audio resources terminated.")
# ==== Transcription and Recording FEATURE END ==== 

# ---------------------------
# ==== Calibration and Eye Tracking FEATURE START ====
# ---------------------------
# Mediapipe face mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Calibration parameters for screen calibration
calibration_points = [
    (100, 100),
    (SCREEN_WIDTH - 100, 100),
    (SCREEN_WIDTH - 100, SCREEN_HEIGHT - 100),
    (100, SCREEN_HEIGHT - 100),
    (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2),
]
screen_calibration_data = []
current_calibration_index = 0
calibration_complete = False

# Vertical calibration parameters
gaze = GazeTracking()
calibration_phase_flag = True
directions = ["up", "center", "down"]
current_direction_index = 0

# Miscellaneous parameters
dwell_time = 3.2
start_time_calibration = None
circle_radius = 50
circle_decrement = 1
phase = "vertical_calibration"  # Switch to "screen_calibration" after vertical calibration
blob_position = None
smoothing_factor = 0.2
blob_radius = 50

full_letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
current_letters = full_letters.copy()
current_selection = []
selected_key = ""
drawn_text = ""

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

def calibrate_gaze(landmarks, calibration_point):
    """
    Calibrates the gaze by mapping eye positions and additional facial landmarks to screen coordinates.

    Args:
        landmarks (list): List of facial landmarks.
        calibration_point (tuple): (x, y) coordinates on the screen to map the gaze to.
    """
    # Define the landmarks of interest
    selected_landmarks = [
        33,   # Left eye inner corner
        133,  # Left eye outer corner
        362,  # Right eye inner corner
        263,  # Right eye outer corner
        468,  # Left iris center
        473,  # Right iris center
        159, 145,  # Left upper and lower eyelids
        386, 374,  # Right upper and lower eyelids
        70, 63, 105, 66, 107,  # Left eyebrow
        336, 296, 334, 293, 300,  # Right eyebrow
        1,    # Nose tip
        78, 308  # Mouth corners
    ]
    
    # Extract and normalize the selected landmarks
    features = []
    for idx in selected_landmarks:
        x = landmarks[idx].x
        y = landmarks[idx].y
        z = landmarks[idx].z  # Depth information (optional)
        features.extend([x, y, z])
    
    # Normalize features based on face dimensions
    # Assuming landmarks 454 and 234 define the horizontal bounds
    # and landmarks 10 and 152 define the vertical bounds
    face_center_x = (landmarks[454].x + landmarks[234].x) / 2
    face_center_y = (landmarks[10].y + landmarks[152].y) / 2
    face_width = abs(landmarks[454].x - landmarks[234].x)
    face_height = abs(landmarks[10].y - landmarks[152].y)
    
    normalized_features = []
    for i in range(0, len(features), 3):
        norm_x = (features[i] - face_center_x) / face_width
        norm_y = (features[i+1] - face_center_y) / face_height
        # norm_z = features[i+2] / face_width  # Example normalization
        normalized_features.extend([norm_x, norm_y])

    # Append normalized features and corresponding screen point
    screen_calibration_data.append((normalized_features, calibration_point))
    logging.debug(f"Calibration data appended: Features {normalized_features} -> Screen {calibration_point}")

def validate_calibration_data(calibration_data):
    """
    Validates that all entries in calibration_data are tuples of two numerical values.

    Args:
        calibration_data (list): List of calibration data tuples.

    Returns:
        bool: True if valid, False otherwise.
    """
    for entry in calibration_data:
        features, screen_point = entry
        if not (isinstance(screen_point, tuple) and len(screen_point) == 2):
            logging.error(f"Invalid screen_point format: {screen_point}")
            return False
        if not all(isinstance(coord, (int, float)) for coord in screen_point):
            logging.error(f"Non-numerical screen_point values: {screen_point}")
            return False
    return True

def transcription_phase():
    """Records audio, transcribes it, and returns the text."""
    logging.info("Starting transcription phase...")
    context_transcription = None
    recorder = AudioRecorder()
    try:
        recorder.record()  # Record audio
        context_transcription = transcribe_audio("output.wav")  # Transcribe audio
        logging.info("Transcription completed.")
        logging.info(f"Transcribed Text: {context_transcription}")
    except KeyboardInterrupt:
        logging.info("Transcription interrupted by user.")
    finally:
        recorder.terminate()  # Ensure resources are cleaned up
    return context_transcription

def transcribe_audio(filename):
    """Transcribes the given audio file using OpenAI's Whisper model."""
    model_whisper = whisper.load_model("tiny")
    result = model_whisper.transcribe(filename)
    return result["text"]

# ---------------------------
# Initialize Webcam
# ---------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logging.error("Cannot open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

# ---------------------------
# Initialize and start the scheduler GUI in a separate thread
# ---------------------------
schedule_thread = Thread(target=lambda: schedule_gui().mainloop(), daemon=True)
schedule_thread.start()

# Initialize the scheduled_question_active flag
scheduled_question_active = False

# ---------------------------
# Define Helper Functions
# ---------------------------
def handle_question():
    """
    Handles the process of listening for a question, transcribing it,
    sending it to ChatGPT, and populating the arrayOfResponses.
    """
    global arrayOfResponses
    arrayOfResponses = []  # Clear previous responses
    global context
    context = transcription_phase()
    logging.info(f"Transcribed Question: {context}")
    
    for i in range(4):
        try:
            chat_response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Given this question, {context}, Give me one example of what someone might answer with. Only say the one actual answer. IE, Do not say, How about sushi, just say sushi. Do not say pizza sounds great, just say pizza. Vary your answers. In your answer do not say any of these items: {arrayOfResponses}. If the question is how are you feeling, don't respond with im feeling fine or im feeling anxious, just say fine/anxious."}
                ]
            )
            response_text = chat_response.choices[0].message.content.strip()
            arrayOfResponses.append(response_text)
            logging.info(f"Received response {i+1}: {response_text}")
        except Exception as e:
            logging.error(f"ChatGPT API call failed: {e}")
            break

    logging.info(f"All responses: {arrayOfResponses}")

def evaluate_models(models, X_test, y_test):
    """
    Evaluates the trained models on test data and returns their accuracies.

    Args:
        models (dict): Trained models.
        X_test (np.ndarray): Test feature matrix.
        y_test (np.ndarray): Test target vectors.

    Returns:
        dict: Model accuracies.
    """
    accuracies = {}
    for model_name, model_obj in models.items():
        if 'X' in model_name:
            y_pred = model_obj.predict(X_test)
            mse = np.mean((y_pred - y_test) ** 2)
            accuracies[model_name] = mse
    return accuracies
# ==== Real-Time AI Training FEATURE END ==== 

# ---------------------------
# Main Loop
# ---------------------------
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        logging.error("Failed to capture frame from webcam.")
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
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Handle vertical calibration without appending to screen_calibration_data
                    normalized_x, normalized_y = get_eye_position(
                        face_landmarks.landmark, 
                        (face_landmarks.landmark[454].x + face_landmarks.landmark[234].x) / 2,
                        (face_landmarks.landmark[10].y + face_landmarks.landmark[152].y) / 2,
                        abs(face_landmarks.landmark[454].x - face_landmarks.landmark[234].x),
                        abs(face_landmarks.landmark[10].y - face_landmarks.landmark[152].y)
                    )
                    # Store the normalized_y for the direction
                    # You can implement your logic here based on normalized_y
                    # For example, you might store calibration data for vertical directions separately
                    # Currently, we are not using vertical_calibration_data for training
                    logging.debug(f"Vertical Calibration: {directions[current_direction_index]} -> {normalized_y}")
            current_direction_index += 1
            if current_direction_index >= len(directions):
                phase = "screen_calibration"
                logging.info("Vertical calibration complete!")
                speak("Vertical calibration complete!")
        continue

    # Phase: Screen Calibration
    elif phase == "screen_calibration":
        if current_calibration_index < len(calibration_points):
            target_x, target_y = calibration_points[current_calibration_index]
            # Animate the calibration point with pulsating effect
            pulsate_radius = 15 + 5 * np.sin(time.time() * 2)  # Pulsating effect
            cv2.circle(frame, (target_x, target_y), int(pulsate_radius), (0, 255, 0), -1)
            cv2.putText(frame, f"Look at point {current_calibration_index + 1}", 
                        (target_x - 150, target_y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    calibrate_gaze(face_landmarks.landmark, (target_x, target_y))

            # Check if enough calibration samples are collected
            # Increased from 100 to 300 samples per point for better calibration
            if len(screen_calibration_data) >= (current_calibration_index + 1) * 300:  # Assuming 300 samples per point
                # Brief pause with animation before moving to next point
                pause_start = time.time()
                while time.time() - pause_start < 0.5:  # 0.5 second pause
                    frame_copy = frame.copy()
                    pulsate_radius = 20 + 5 * np.sin(time.time() * 2)
                    cv2.circle(frame_copy, (target_x, target_y), int(pulsate_radius), (255, 0, 0), -1)
                    cv2.putText(frame_copy, "Calibrating...", 
                                (target_x - 150, target_y + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.imshow("Eye Gaze Tracker", frame_copy)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                current_calibration_index += 1
                time.sleep(0.2)  # Short delay before next point
                if current_calibration_index >= len(calibration_points):
                    # Validate calibration data before training
                    if validate_calibration_data(screen_calibration_data):
                        # Train all models with screen calibration data
                        models_trained = train_models(screen_calibration_data)
                        calibration_complete = True
                        phase = "tracking"  # Update phase to tracking
                        logging.info("Screen calibration complete!")
                        speak("Screen calibration complete!")
                    else:
                        logging.error("Calibration data validation failed. Please recalibrate.")
                        speak("Calibration data validation failed. Please recalibrate.")
        else:
            calibration_complete = True
            phase = "tracking"  # Update phase to tracking
            logging.info("Screen calibration complete!")
            speak("Screen calibration complete!")
            continue

    # Phase: Tracking
    elif phase == "tracking":
        if not models_trained:
            logging.warning("Models not trained yet.")
            cv2.imshow("Eye Gaze Tracker", frame)
            continue

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract normalized features
                normalized_x, normalized_y = get_eye_position(
                    face_landmarks.landmark, 
                    (face_landmarks.landmark[454].x + face_landmarks.landmark[234].x) / 2,
                    (face_landmarks.landmark[10].y + face_landmarks.landmark[152].y) / 2,
                    abs(face_landmarks.landmark[454].x - face_landmarks.landmark[234].x),
                    abs(face_landmarks.landmark[10].y - face_landmarks.landmark[152].y)
                )

                # Extract extended features for ML models
                selected_landmarks = [
                    33,   # Left eye inner corner
                    133,  # Left eye outer corner
                    362,  # Right eye inner corner
                    263,  # Right eye outer corner
                    468,  # Left iris center
                    473,  # Right iris center
                    159, 145,  # Left upper and lower eyelids
                    386, 374,  # Right upper and lower eyelids
                    70, 63, 105, 66, 107,  # Left eyebrow
                    336, 296, 334, 293, 300,  # Right eyebrow
                    1,    # Nose tip
                    78, 308  # Mouth corners
                ]
                features = []
                for idx in selected_landmarks:
                    x = face_landmarks.landmark[idx].x
                    y = face_landmarks.landmark[idx].y
                    z = face_landmarks.landmark[idx].z  # Depth information (optional)
                    features.extend([x, y, z])

                # Normalize based on face dimensions
                face_center_x = (face_landmarks.landmark[454].x + face_landmarks.landmark[234].x) / 2
                face_center_y = (face_landmarks.landmark[10].y + face_landmarks.landmark[152].y) / 2
                face_width = abs(face_landmarks.landmark[454].x - face_landmarks.landmark[234].x)
                face_height = abs(face_landmarks.landmark[10].y - face_landmarks.landmark[152].y)
                
                normalized_features_extended = []
                for i in range(0, len(features), 3):
                    norm_x = (features[i] - face_center_x) / face_width
                    norm_y = (features[i+1] - face_center_y) / face_height
                    # norm_z = features[i+2] / face_width  # Example normalization
                    normalized_features_extended.extend([norm_x, norm_y])

                # Map gaze to screen using all models
                try:
                    predictions = map_gaze_to_screen(models_trained, normalized_features_extended)
                    draw_blobs(frame, predictions)
                except Exception as e:
                    logging.error(f"Error in mapping gaze to screen: {e}")
                    continue

                # Determine regions for agreement logic based on SVR predictions
                svr_prediction = predictions.get('SVR', (None, None))
                if svr_prediction[0] is None or svr_prediction[1] is None:
                    continue

                # Check which region the SVR prediction is in
                in_top = is_in_bounds(svr_prediction[0], svr_prediction[1], TOP_BOUNDS)
                in_left = is_in_bounds(svr_prediction[0], svr_prediction[1], LEFT_BOUNDS)
                in_bottom = is_in_bounds(svr_prediction[0], svr_prediction[1], BOTTOM_BOUNDS)
                in_right = is_in_bounds(svr_prediction[0], svr_prediction[1], RIGHT_BOUNDS)

                # ==== Agreement Logic Start ====
                # Define gaze directions based on SVR predictions
                gaze_vertical = "uncalibrated"
                gaze_horizontal = "uncalibrated"

                if svr_prediction[1] < SCREEN_HEIGHT // 4:
                    gaze_vertical = "up"
                elif svr_prediction[1] > SCREEN_HEIGHT * 3 // 4:
                    gaze_vertical = "down"
                else:
                    gaze_vertical = "center"

                if svr_prediction[0] < SCREEN_WIDTH // 4:
                    gaze_horizontal = "left"
                elif svr_prediction[0] > SCREEN_WIDTH * 3 // 4:
                    gaze_horizontal = "right"
                else:
                    gaze_horizontal = "center"

                # Combine directions for display
                text = f"Looking {gaze_horizontal} and {gaze_vertical}"
                cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Handle each region
                # Top Region Logic
                if in_top:
                    if gaze_vertical != "down":  # Not looking down
                        direction_timer["up"] += 1 / 30  # Assuming 30 FPS
                        draw_progress_bar(frame, (SCREEN_WIDTH // 2 - 100, 10), direction_timer["up"], AGREEMENT_THRESHOLD)
                        if direction_timer["up"] >= AGREEMENT_THRESHOLD:
                            if len(arrayOfResponses) > 0:
                                selected_response = arrayOfResponses[0]
                                logging.info(f"User is looking at the TOP element: {selected_response}")
                                direction_timer["up"] = 0  # Reset the timer
                                
                                # Define the prompt for top
                                prompt = (
                                    f"Given this question, {context}, here is a word that someone answered the question with: {selected_response}. "
                                    f"Please provide a sample response sentence that uses the one-word answer to respond to the question. "
                                    f"For example, if the question is 'How are you feeling?' and the one-word answer is 'happy', you should output 'I'm feeling happy.' "
                                    f"Just respond with the answer short and brief; do not ask anything else. It is supposed to be an answer only. "
                                    f"If the question is 'What do you want for dinner?' and the one-word answer is 'sushi', do not say 'Sushi sounds great, do you have a specific type of sushi in mind, like rolls or sashimi?' "
                                    f"Instead, say, 'Sushi sounds great.' Keep it short and brief and only answer the question as best you can and keep it at that."
                                )
                                logging.info(f"Prompt being sent to ChatGPT: {prompt}")  # Debug

                                # Create a chat completion
                                try:
                                    chat_response = openai.ChatCompletion.create(
                                        model=model,
                                        messages=[
                                            {"role": "system", "content": "You are a helpful assistant."},
                                            {"role": "user", "content": prompt}
                                        ]
                                    )
                                    # Extract and print the assistant's reply
                                    assistant_reply = chat_response.choices[0].message.content.strip()
                                    logging.info(f"Assistant Reply: {assistant_reply}")
                                    speak(assistant_reply)

                                    # ==== NEW FEATURE START ====
                                    if scheduled_question_active:
                                        analysis_prompt = (
                                            f"This is a question that an ALS patient who is severely paralyzed responded to, and this is their answer: {selected_response}. "
                                            f"Do you believe they need assistance from a caregiver? Examples of when an ALS patient needs assistance from a caregiver is if they need water, or if they feel bad. "
                                            f"Examples of when an ALS patient does not need assistance from a caregiver is if they say they are feeling good. "
                                            f"Please return a true or false output, true being they need assistance, false being they do not need assistance. Your answer must be one word: True or False."
                                        )
                                        logging.info(f"Analysis Prompt being sent to ChatGPT: {analysis_prompt}")  # Debug

                                        try:
                                            analysis_response = openai.ChatCompletion.create(
                                                model=model,
                                                messages=[
                                                    {"role": "system", "content": "You are a helpful assistant."},
                                                    {"role": "user", "content": analysis_prompt}
                                                ]
                                            )
                                            analysis_reply = analysis_response.choices[0].message.content.strip().lower()
                                            logging.info(f"Analysis Reply: {analysis_reply}")
                                            if analysis_reply == "true":
                                                send_email_for_help(emergency_email)
                                            elif analysis_reply == "false":
                                                pass  # Do nothing
                                            else:
                                                logging.warning("Unexpected response from ChatGPT for analysis.")
                                        except Exception as e:
                                            logging.error(f"ChatGPT API call for analysis failed: {e}")

                                        # Reset the flag after processing
                                        scheduled_question_active = False
                                    # ==== NEW FEATURE END ====

                                except Exception as e:
                                    logging.error(f"ChatGPT API call failed: {e}")
                    else:
                        direction_timer["up"] = 0  # Reset if looking opposite
                else:
                    direction_timer["up"] = 0  # Reset if not in top region

                # Bottom Region Logic
                if in_bottom:
                    if gaze_vertical != "up":  # Not looking up
                        direction_timer["down"] += 1 / 30  # Assuming 30 FPS
                        draw_progress_bar(frame, (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT - 30), direction_timer["down"], AGREEMENT_THRESHOLD)
                        if direction_timer["down"] >= AGREEMENT_THRESHOLD:
                            if len(arrayOfResponses) > 2:
                                selected_response = arrayOfResponses[2]
                                logging.info(f"User is looking at the BOTTOM element: {selected_response}")
                                direction_timer["down"] = 0
                                
                                # Define the prompt for bottom
                                prompt = (
                                    f"Given this question, {context}, here is a word that someone answered the question with: {selected_response}. "
                                    f"Please provide a sample response sentence that uses the one-word answer to respond to the question. "
                                    f"For example, if the question is 'How are you feeling?' and the one-word answer is 'happy', you should output 'I'm feeling happy.' "
                                    f"Just respond with the answer short and brief; do not ask anything else. It is supposed to be an answer only. "
                                    f"If the question is 'What do you want for dinner?' and the one-word answer is 'sushi', do not say 'Sushi sounds great, do you have a specific type of sushi in mind, like rolls or sashimi?' "
                                    f"Instead, say, 'Sushi sounds great.' Keep it short and brief and only answer the question as best you can and keep it at that."
                                )
                                logging.info(f"Prompt being sent to ChatGPT: {prompt}")  # Debug

                                # Create a chat completion
                                try:
                                    chat_response = openai.ChatCompletion.create(
                                        model=model,
                                        messages=[
                                            {"role": "system", "content": "You are a helpful assistant."},
                                            {"role": "user", "content": prompt}
                                        ]
                                    )
                                    # Extract and print the assistant's reply
                                    assistant_reply = chat_response.choices[0].message.content.strip()
                                    logging.info(f"Assistant Reply: {assistant_reply}")
                                    speak(assistant_reply)

                                    # ==== NEW FEATURE START ====
                                    if scheduled_question_active:
                                        analysis_prompt = (
                                            f"This is a question that an ALS patient who is severely paralyzed responded to, and this is their answer: {selected_response}. "
                                            f"Do you believe they need assistance from a caregiver? Examples of when an ALS patient needs assistance from a caregiver is if they need water, or if they feel bad. "
                                            f"Examples of when an ALS patient does not need assistance from a caregiver is if they say they are feeling good. "
                                            f"Please return a true or false output, true being they need assistance, false being they do not need assistance. Your answer must be one word: True or False."
                                        )
                                        logging.info(f"Analysis Prompt being sent to ChatGPT: {analysis_prompt}")  # Debug

                                        try:
                                            analysis_response = openai.ChatCompletion.create(
                                                model=model,
                                                messages=[
                                                    {"role": "system", "content": "You are a helpful assistant."},
                                                    {"role": "user", "content": analysis_prompt}
                                                ]
                                            )
                                            analysis_reply = analysis_response.choices[0].message.content.strip().lower()
                                            logging.info(f"Analysis Reply: {analysis_reply}")
                                            if analysis_reply == "true":
                                                send_email_for_help(emergency_email)
                                            elif analysis_reply == "false":
                                                pass  # Do nothing
                                            else:
                                                logging.warning("Unexpected response from ChatGPT for analysis.")
                                        except Exception as e:
                                            logging.error(f"ChatGPT API call for analysis failed: {e}")

                                        # Reset the flag after processing
                                        scheduled_question_active = False
                                    # ==== NEW FEATURE END ====

                                except Exception as e:
                                    logging.error(f"ChatGPT API call failed: {e}")
                    else:
                        direction_timer["down"] = 0  # Reset if looking opposite
                else:
                    direction_timer["down"] = 0  # Reset if not in bottom region

                # Left Region Logic
                if in_left:
                    if gaze_horizontal != "right":  # Not looking right
                        direction_timer["left"] += 1 / 30  # Assuming 30 FPS
                        draw_progress_bar(frame, (10, SCREEN_HEIGHT // 2 - 10), direction_timer["left"], AGREEMENT_THRESHOLD)
                        if direction_timer["left"] >= AGREEMENT_THRESHOLD:
                            if len(arrayOfResponses) > 1:
                                selected_response = arrayOfResponses[1]
                                logging.info(f"User is looking at the LEFT element: {selected_response}")
                                direction_timer["left"] = 0
                                
                                # Define the prompt for left
                                prompt = (
                                    f"Given this question, {context}, here is a word that someone answered the question with: {selected_response}. "
                                    f"Please provide a sample response sentence that uses the one-word answer to respond to the question. "
                                    f"For example, if the question is 'How are you feeling?' and the one-word answer is 'happy', you should output 'I'm feeling happy.' "
                                    f"Just respond with the answer short and brief; do not ask anything else. It is supposed to be an answer only. "
                                    f"If the question is 'What do you want for dinner?' and the one-word answer is 'sushi', do not say 'Sushi sounds great, do you have a specific type of sushi in mind, like rolls or sashimi?' "
                                    f"Instead, say, 'Sushi sounds great.' Keep it short and brief and only answer the question as best you can and keep it at that."
                                )
                                logging.info(f"Prompt being sent to ChatGPT: {prompt}")  # Debug

                                # Create a chat completion
                                try:
                                    chat_response = openai.ChatCompletion.create(
                                        model=model,
                                        messages=[
                                            {"role": "system", "content": "You are a helpful assistant."},
                                            {"role": "user", "content": prompt}
                                        ]
                                    )
                                    # Extract and print the assistant's reply
                                    assistant_reply = chat_response.choices[0].message.content.strip()
                                    logging.info(f"Assistant Reply: {assistant_reply}")
                                    speak(assistant_reply)

                                    # ==== NEW FEATURE START ====
                                    if scheduled_question_active:
                                        analysis_prompt = (
                                            f"This is a question that an ALS patient who is severely paralyzed responded to, and this is their answer: {selected_response}. "
                                            f"Do you believe they need assistance from a caregiver? Examples of when an ALS patient needs assistance from a caregiver is if they need water, or if they feel bad. "
                                            f"Examples of when an ALS patient does not need assistance from a caregiver is if they say they are feeling good. "
                                            f"Please return a true or false output, true being they need assistance, false being they do not need assistance. Your answer must be one word: True or False."
                                        )
                                        logging.info(f"Analysis Prompt being sent to ChatGPT: {analysis_prompt}")  # Debug

                                        try:
                                            analysis_response = openai.ChatCompletion.create(
                                                model=model,
                                                messages=[
                                                    {"role": "system", "content": "You are a helpful assistant."},
                                                    {"role": "user", "content": analysis_prompt}
                                                ]
                                            )
                                            analysis_reply = analysis_response.choices[0].message.content.strip().lower()
                                            logging.info(f"Analysis Reply: {analysis_reply}")
                                            if analysis_reply == "true":
                                                send_email_for_help(emergency_email)
                                            elif analysis_reply == "false":
                                                pass  # Do nothing
                                            else:
                                                logging.warning("Unexpected response from ChatGPT for analysis.")
                                        except Exception as e:
                                            logging.error(f"ChatGPT API call for analysis failed: {e}")

                                        # Reset the flag after processing
                                        scheduled_question_active = False
                                    # ==== NEW FEATURE END ====

                                except Exception as e:
                                    logging.error(f"ChatGPT API call failed: {e}")
                    else:
                        direction_timer["left"] = 0  # Reset if looking opposite
                else:
                    direction_timer["left"] = 0  # Reset if not in left region

                # Right Region Logic
                if in_right:
                    if gaze_horizontal != "left":  # Not looking left
                        direction_timer["right"] += 1 / 30  # Assuming 30 FPS
                        draw_progress_bar(frame, (SCREEN_WIDTH - 220, SCREEN_HEIGHT // 2 - 10), direction_timer["right"], AGREEMENT_THRESHOLD)
                        if direction_timer["right"] >= AGREEMENT_THRESHOLD:
                            if len(arrayOfResponses) > 3:
                                selected_response = arrayOfResponses[3]
                                logging.info(f"User is looking at the RIGHT element: {selected_response}")
                                direction_timer["right"] = 0
                                
                                # Define the prompt for right
                                prompt = (
                                    f"Given this question, {context}, here is a word that someone answered the question with: {selected_response}. "
                                    f"Please provide a sample response sentence that uses the one-word answer to respond to the question. "
                                    f"For example, if the question is 'How are you feeling?' and the one-word answer is 'happy', you should output 'I'm feeling happy.' "
                                    f"Just respond with the answer short and brief; do not ask anything else. It is supposed to be an answer only. "
                                    f"If the question is 'What do you want for dinner?' and the one-word answer is 'sushi', do not say 'Sushi sounds great, do you have a specific type of sushi in mind, like rolls or sashimi?' "
                                    f"Instead, say, 'Sushi sounds great.' Keep it short and brief and only answer the question as best you can and keep it at that."
                                )
                                logging.info(f"Prompt being sent to ChatGPT: {prompt}")  # Debug

                                # Create a chat completion
                                try:
                                    chat_response = openai.ChatCompletion.create(
                                        model=model,
                                        messages=[
                                            {"role": "system", "content": "You are a helpful assistant."},
                                            {"role": "user", "content": prompt}
                                        ]
                                    )
                                    # Extract and print the assistant's reply
                                    assistant_reply = chat_response.choices[0].message.content.strip()
                                    logging.info(f"Assistant Reply: {assistant_reply}")
                                    speak(assistant_reply)

                                    # ==== NEW FEATURE START ====
                                    if scheduled_question_active:
                                        analysis_prompt = (
                                            f"This is a question that an ALS patient who is severely paralyzed responded to, and this is their answer: {selected_response}. "
                                            f"Do you believe they need assistance from a caregiver? Examples of when an ALS patient needs assistance from a caregiver is if they need water, or if they feel bad. "
                                            f"Examples of when an ALS patient does not need assistance from a caregiver is if they say they are feeling good. "
                                            f"Please return a true or false output, true being they need assistance, false being they do not need assistance. Your answer must be one word: True or False."
                                        )
                                        logging.info(f"Analysis Prompt being sent to ChatGPT: {analysis_prompt}")  # Debug

                                        try:
                                            analysis_response = openai.ChatCompletion.create(
                                                model=model,
                                                messages=[
                                                    {"role": "system", "content": "You are a helpful assistant."},
                                                    {"role": "user", "content": analysis_prompt}
                                                ]
                                            )
                                            analysis_reply = analysis_response.choices[0].message.content.strip().lower()
                                            logging.info(f"Analysis Reply: {analysis_reply}")
                                            if analysis_reply == "true":
                                                send_email_for_help(emergency_email)
                                            elif analysis_reply == "false":
                                                pass  # Do nothing
                                            else:
                                                logging.warning("Unexpected response from ChatGPT for analysis.")
                                        except Exception as e:
                                            logging.error(f"ChatGPT API call for analysis failed: {e}")

                                        # Reset the flag after processing
                                        scheduled_question_active = False
                                    # ==== NEW FEATURE END ====

                                except Exception as e:
                                    logging.error(f"ChatGPT API call failed: {e}")
                    else:
                        direction_timer["right"] = 0  # Reset if looking opposite
                else:
                    direction_timer["right"] = 0  # Reset if not in right region
                # ==== Agreement Logic End ====

    # ==================== NEW KEY PRESS HANDLING ====================
    # Handle key presses after processing gaze tracking
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # Space key
        logging.info("Space key pressed. Listening for question...")
        # Start a new thread to handle the question to avoid blocking the main loop
        question_thread = Thread(target=handle_question)
        question_thread.start()
    # Exit on pressing 'q'
    if key == ord('q'):
        logging.info("Exiting program.")
        break
    # ==================== END OF NEW KEY PRESS HANDLING ====================

    # Display the frame in fullscreen
    cv2.namedWindow("Eye Gaze Tracker", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Eye Gaze Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Eye Gaze Tracker", frame)

# Release resources
cap.release()
cv2.destroyAllWindows()

# ==== SCHEDULED QUESTION FEATURE START ====
# Stop the scheduler thread before exiting
scheduler.stop()
scheduler_thread.join()
# ==== SCHEDULED QUESTION FEATURE END ==== 
