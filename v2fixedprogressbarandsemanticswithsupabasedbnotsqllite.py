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
import pyttsx3
import pyaudio
import webrtcvad
import whisper
from openai import OpenAI
from elevenlabs import ElevenLabs, play
from dotenv import load_dotenv
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import smtplib

from supabase import create_client, Client 
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

emergency_email = None

load_dotenv()

APP_PASSWORD = os.getenv("APP_PASSWORD")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    print("[ERROR] SUPABASE_URL and SUPABASE_KEY must be set in the environment variables.")
    exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def preload_whisper_model():
    print("[INFO] Downloading Whisper model...")
    try:
        whisper.load_model("tiny")  
        print("[INFO] Whisper model downloaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to download Whisper model: {e}")

preload_whisper_model()

def send_email_for_help(emergency_email):
    subject = "Help Needed"
    body = "The ALS patient has been determined to be in need of assistance. Please check on them immediately."
    sender = "kevinx8017@gmail.com"
    recipients = [sender, emergency_email]

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender, APP_PASSWORD)
            smtp.sendmail(sender, recipients, msg.as_string())
        print("Email sent successfully")
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")


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
            for sq in list(self.scheduled_questions): 
                if sq.scheduled_time <= now:
                   
                    trigger_scheduled_question(sq)
                    if sq.recurring:
                        sq.reschedule()
                    else:
                        self.scheduled_questions.remove(sq)
            time.sleep(1) 

    def stop(self):
        self.stop_event.set()

def is_semantically_unique(existing_answers, new_answer, model, threshold=0.7):
    """
    Checks if the new_answer is semantically unique compared to existing_answers.

    Args:
        existing_answers (list of str): List of already selected answers.
        new_answer (str): The new answer to check.
        model (SentenceTransformer): The sentence transformer model.
        threshold (float): The cosine similarity threshold.

    Returns:
        bool: True if unique, False otherwise.
    """
    if not existing_answers:
        return True
    new_embedding = model.encode([new_answer])[0]
    existing_embeddings = model.encode(existing_answers)
    similarities = cosine_similarity([new_embedding], existing_embeddings)[0]
    return all(sim < threshold for sim in similarities)

def get_top_responses(user_question, top_n=4, similarity_threshold=0.55, diversity_threshold=0.7):
    """
    Hybrid approach:
    1 Compute cosine similarity with existing Q&A in DB
    2) Add the best matches if similarity >= thfreshold and semantically unique
    3 Use GPT to generate additional candidate answers, pick top by similarity and uniqueness
    4) Return final top_n.d
    """
    global sample_data, sample_questions, sample_embeddings


    sample_data = fetch_all_qa()
    sample_questions = [item["question"] for item in sample_data]
    sample_embeddings = st_model.encode(sample_questions)

  
    user_embedding = st_model.encode([user_question])
   
    similarities = cosine_similarity(user_embedding, sample_embeddings)[0]

    
    sample_with_similarity = [
        {"answer": item["answer"], "similarity": sim}
        for item, sim in zip(sample_data, similarities)
    ]
   
    sample_with_similarity.sort(key=lambda x: x["similarity"], reverse=True)

    final_responses = []
    methods = []

    
    for sample_item in sample_with_similarity:
        if len(final_responses) >= top_n:
            break
        if sample_item["similarity"] >= similarity_threshold:
            if is_semantically_unique([resp["answer"] for resp in final_responses], sample_item["answer"], st_model, diversity_threshold):
                final_responses.append({
                    "answer": sample_item["answer"],
                    "similarity": sample_item["similarity"],
                    "method": "cosine_similarity"
                })
                methods.append("cosine_similarity")

   
    remaining = top_n - len(final_responses)

    if remaining > 0:
      
        gpt_responses = generate_dynamic_answers(user_question, num_responses=remaining * 2)
     
        for resp in gpt_responses:
            if not resp.startswith("Error") and is_semantically_unique(
                [r["answer"] for r in final_responses], resp, st_model, diversity_threshold
            ):
                final_responses.append({
                    "answer": resp,
                    "similarity": cosine_similarity(user_embedding, st_model.encode([resp]))[0][0],
                    "method": "GPT"
                })
                methods.append("GPT")
                if len(final_responses) >= top_n:
                    break

  
    while len(final_responses) < top_n:
        final_responses.append({
            "answer": "N/A",
            "similarity": 0,
            "method": "placeholder"
        })

    answers = [resp["answer"] for resp in final_responses[:top_n]]
    similarities = [resp["similarity"] for resp in final_responses[:top_n]]
    methods_used = [resp["method"] for resp in final_responses[:top_n]]

    return answers, similarities, methods_used

def trigger_scheduled_question(sq):
    """
    Triggers the scheduled question: alerts the user, vocalizes the question,
    fetches responses from ChatGPT (using the new semantic method below),
    and displays them.
    """
    print("[INFO] Triggering scheduled question.")
    speak("Alert, time for scheduled question.")
    speak(sq.question)

   
    arrayOfResponses.clear()

    global context
    context = sq.question
    print(f"[INFO] Scheduled Question: {context}")

    
    global scheduled_question_active
    scheduled_question_active = True


    answers, sims, methods = get_top_responses(context, top_n=4, similarity_threshold=0.55, diversity_threshold=0.7)
    if len(answers) < 4:
       
        print("[WARNING] Fallback: Not enough answers found from semantic method, filling with placeholders.")
        while len(answers) < 4:
            answers.append("N/A")

    
    arrayOfResponses.extend(answers)

    print(f"[INFO] All scheduled responses (semantic/gpt): {arrayOfResponses}")



monitor = get_monitors()[0]
SCREEN_WIDTH = monitor.width
SCREEN_HEIGHT = monitor.height


TOP_BOUNDS = (SCREEN_WIDTH // 4, SCREEN_WIDTH * 3 // 4, 0, SCREEN_HEIGHT // 4)
LEFT_BOUNDS = (0, SCREEN_WIDTH // 4, SCREEN_HEIGHT // 4, SCREEN_HEIGHT * 3 // 4)
BOTTOM_BOUNDS = (SCREEN_WIDTH // 4, SCREEN_WIDTH * 3 // 4, SCREEN_HEIGHT * 3 // 4, SCREEN_HEIGHT)
RIGHT_BOUNDS = (SCREEN_WIDTH * 3 // 4, SCREEN_WIDTH, SCREEN_HEIGHT // 4, SCREEN_HEIGHT * 3 // 4)

arrayOfResponses = []
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

direction_timer = {"up": 0, "left": 0, "down": 0, "right": 0}
AGREEMENT_THRESHOLD = 0.6  

def is_in_bounds(x, y, bounds):
    """Checks if a point (x, y) is within given bounds."""
    x_min, x_max, y_min, y_max = bounds
    return x_min <= x <= x_max and y_min <= y <= y_max


client_eleven = ElevenLabs(api_key=os.getenv("ELEVEN_API_KEY"))
cloned_voice = None
voice_status_var = None

def speak(text):
    """
    Uses ElevenLabs to generate and play speech from text.
    """
    if cloned_voice is None:
        print("[ERROR] Voice not cloned yet. Using fallback TTS engine.")
        fallback_tts_engine = pyttsx3.init()
        fallback_tts_engine.say(text)
        fallback_tts_engine.runAndWait()
    else:
        try:
            audio = client_eleven.generate(text=text, voice=cloned_voice)
            play(audio)
        except Exception as e:
            print(f"[ERROR] ElevenLabs TTS failed: {e}. Using fallback TTS engine.")
            fallback_tts_engine = pyttsx3.init()
            fallback_tts_engine.say(text)
            fallback_tts_engine.runAndWait()



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
    except Exception as e:
        print(f"[ERROR] Transcription failed: {e}")
    finally:
        recorder.terminate()
    return context

# Medhiapipe face mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)


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


gaze = GazeTracking()
calibration_phase_flag = True
directions = ["up", "center", "down"]
current_direction_index = 0


dwell_time = 3.2
start_time = None
buffer_start_time = None
post_calibration_buffer = 0.3
circle_radius = 50
circle_decrement = 1
phase = "vertical_calibration"
blob_position = None
smoothing_factor = 0.2
blob_radius = 50

def get_eye_position(landmarks, face_center_x, face_center_y, face_width, face_height):
    """Extract normalized eye position (average of both irises)."""
    left_iris = landmarks[468]  
    right_iris = landmarks[473] 
    eye_x = (left_iris.x + right_iris.x) / 2
    eye_y = (left_iris.y + right_iris.y) / 2

    normalized_x = (eye_x - face_center_x) / face_width
    normalized_y = (eye_y - face_center_y) / face_height
    return normalized_x, normalized_y

def map_gaze_to_screen(normalized_x, normalized_y, calibration_data):
    """Map normalized eye positions to screen coordinates using calibration."""
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
    """Draw a morphing blob at the given position, ensuring it stays within the window."""
    noise = random.randint(-5, 5)
    radius += noise

    x, y = position
    r = max(radius, 10)  
    
    x = max(r, min(x, frame.shape[1] - r))
    y = max(r, min(y, frame.shape[0] - r))
   
    cv2.circle(frame, (x, y), r, color, -1)


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
fine_tuned_client = OpenAI(api_key=os.getenv("OPENAI_FINE_TUNED_API_KEY"))
model = "gpt-4o-mini"  


def fetch_all_qa():
    try:
        response = supabase.table('qa').select('question, answer').execute()
        if hasattr(response, 'error') and response.error:
            print(f"[ERROR] Fetching Q&A from Supabase: {response.error}")
            return []
        elif isinstance(response, dict) and 'error' in response:
            print(f"[ERROR] Fetching Q&A from Supabase: {response['error']}")
            return []
        data = response.data if hasattr(response, 'data') else response.get('data', [])
        return [{"question": item['question'], "answer": item['answer']} for item in data]
    except Exception as e:
        print(f"[ERROR] Exception in fetch_all_qa: {e}")
        return []


def add_qa_to_db(question, answer):
    try:
        response = supabase.table('qa').insert({"question": question, "answer": answer}).execute()
        
        if hasattr(response, 'error') and response.error:
            print(f"[ERROR] Adding Q&A to Supabase: {response.error}")
        elif isinstance(response, dict) and 'error' in response:
            print(f"[ERROR] Adding Q&A to Supabase: {response['error']}")
        else:
            print(f"[INFO] Added Q&A to Supabase -> Q: {question}, A: {answer}")
    except Exception as e:
        print(f"[ERROR] Could not add Q&A to Supabase: {e}")


def display_all_qa():
    try:
        response = supabase.table('qa').select('*').execute()
        
        if hasattr(response, 'error') and response.error:
            print(f"[ERROR] Fetching Q&A from Supabase for display: {response.error}")
            return
        elif isinstance(response, dict) and 'error' in response:
            print(f"[ERROR] Fetching Q&A from Supabase for display: {response['error']}")
            return
        rows = response.data if hasattr(response, 'data') else response.get('data', [])
        print("\nCurrent Q&A in Supabase Database:")
        for row in rows:
            print(f'ID: {row["id"]}')
            print(f'Question: {row["question"]}')
            print(f'Answer: {row["answer"]}\n')
    except Exception as e:
        print(f"[ERROR] Exception in display_all_qa: {e}")


sample_data = fetch_all_qa()


if not sample_data:
    default_data = [
        {"question": "How are you feeling?", "answer": "I'm feeling fine."},
        {"question": "What do you want to eat?", "answer": "I'd like to have some rice."},
        {"question": "What's your favorite hobby?", "answer": "I enjoy reading books."},
        {"question": "Where do you live?", "answer": "I live in New York."},
    ]
    for item in default_data:
        add_qa_to_db(item["question"], item["answer"])
    sample_data = fetch_all_qa()
    print("[INFO] Added default Q&A to the Supabase database.")


display_all_qa()


st_model = SentenceTransformer('all-MiniLM-L6-v2')


sample_questions = [item["question"] for item in sample_data]
sample_embeddings = st_model.encode(sample_questions)

def generate_dynamic_answers(user_question, num_responses=5):
    
    pastResponses = []
    answers = []
    try:
        for _ in range(num_responses):
            print("using fine tuned model")
            model = "ft:gpt-4o-mini-2024-07-18:personal::AibOSZch"
            prompt = (
                f"You are a helpful assistant. Generate one unique and diverse possible answer to this question: {user_question}. "
                f"Ensure that this answer is different in meaning and phrasing from previous answers: {pastResponses}. "
                f"Provide a short, single-sentence answer. "
                f"Pretend a user has to choose 2 different answers so your answers must be diverse and different, "
                f"preferably opposite."
            )
            completion = fine_tuned_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                n=1,
                temperature=0.9
            )
            response = completion.choices[0].message.content.strip()
            if response not in pastResponses:
                pastResponses.append(response)
                answers.append(response)

        print(f"\n[DEBUG] Generated GPT Responses:")
        for idx, ans in enumerate(answers, 1):
            print(f"{idx}. {ans}")  

        return answers
    except Exception as e:
        print(f"[ERROR] Generating GPT responses: {e}")
        return [f"Error generating response: {e}"]


scheduler = Scheduler()
scheduler_thread = Thread(target=scheduler.run, daemon=True)
scheduler_thread.start()

def open_schedule_window():
    schedule_window = Toplevel(root)
    schedule_window.title("Schedule a Question")

    Label(schedule_window, text="Question:").grid(row=0, column=0, padx=10, pady=5, sticky='e')
    question_var = StringVar()
    Entry(schedule_window, textvariable=question_var, width=50).grid(row=0, column=1, padx=10, pady=5)

    Label(schedule_window, text="Schedule Type:").grid(row=1, column=0, padx=10, pady=5, sticky='e')
    schedule_type_var = StringVar(value="time")
    Radiobutton(schedule_window, text="Specific Time", variable=schedule_type_var, value="time").grid(row=1, column=1, padx=10, pady=5, sticky='w')
    Radiobutton(schedule_window, text="Countdown Timer (seconds)", variable=schedule_type_var, value="timer").grid(row=2, column=1, padx=10, pady=5, sticky='w')

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

    Label(schedule_window, text="Hour (0-23):").grid(row=3, column=0, padx=10, pady=5, sticky='e')
    time_entry = Entry(schedule_window, textvariable=time_var, width=10)
    time_entry.grid(row=3, column=1, padx=10, pady=5, sticky='w')

    timer_entry = Entry(schedule_window, textvariable=timer_var, width=10)
    timer_entry.grid(row=3, column=1, padx=10, pady=5, sticky='w')
    timer_entry.grid_remove()

    recurring_var = IntVar()
    Checkbutton(schedule_window, text="Repeat Daily", variable=recurring_var).grid(row=4, column=1, padx=10, pady=5, sticky='w')

    def add_scheduled_question():
        question = question_var.get().strip()
        if not question:
            print("[ERROR] Question cannot be empty.")
            speak("Question cannot be empty.")
            return

        schedule_type = schedule_type_var.get()
        if schedule_type == "time":
            try:
                hour = int(time_var.get())
                if not (0 <= hour <= 23):
                    raise ValueError
                minute = 0
                scheduled_time = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
                if scheduled_time < datetime.now():
                    scheduled_time += timedelta(days=1)
            except ValueError:
                print("[ERROR] Invalid time format.")
                speak("Invalid time format.")
                return
        else:
            try:
                countdown = int(timer_var.get())
                if countdown <= 0:
                    raise ValueError
                scheduled_time = datetime.now() + timedelta(seconds=countdown)
            except ValueError:
                print("[ERROR] Invalid countdown timer format.")
                speak("Invalid countdown timer format.")
                return

        recurring = bool(recurring_var.get())
        sq = ScheduledQuestion(question, scheduled_time, recurring)
        scheduler.add_question(sq)
        print(f"[INFO] Scheduled question added: '{question}' at {scheduled_time} | Recurring: {recurring}")
        speak("Scheduled question added successfully.")
        schedule_window.destroy()

    Button(schedule_window, text="Add Scheduled Question", command=add_scheduled_question).grid(row=5, column=1, padx=10, pady=10, sticky='e')

def upload_voice():
    global cloned_voice
    file_path = filedialog.askopenfilename(
        title="Select Voice Recording",
        filetypes=(("Audio Files", "*.mp3 *.wav"), ("All Files", "*.*"))
    )
    if file_path:
        print(f"[INFO] Cloning voice from {file_path}...")
        voice_status_var.set("Voice Status: Cloning...")
        try:
            cloned_voice = client_eleven.clone(
                name="UserClonedVoice",
                description="Cloned voice from user uploaded recording.",
                files=[file_path]
            )
            print("[INFO] Voice cloned successfully.")
            voice_status_var.set("Voice Status: Cloned")
            speak("Voice cloned successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to clone voice: {e}")
            voice_status_var.set("Voice Status: Clone Failed")
            speak("Failed to clone voice. Please try again.")

def schedule_gui():
    global root, voice_status_var
    root = tk.Tk()
    root.title("Eye Gaze Tracker Scheduler")

    Button(root, text="Schedule a Question", command=open_schedule_window).pack(padx=20, pady=10)

    # Emergency Email
    Label(root, text="Emergency Email:").pack(padx=10, pady=5)
    email_var = StringVar()
    email_entry = Entry(root, textvariable=email_var, width=50)
    email_entry.pack(padx=10, pady=5)
    Button(root, text="Set Emergency Email", command=lambda: set_emergency_email(email_var, email_entry)).pack(padx=10, pady=5)

    # Upload Voice
    Label(root, text="Upload Voice Recording:").pack(padx=10, pady=5)
    Button(root, text="Upload Voice", command=upload_voice).pack(padx=10, pady=5)
    voice_status_var = StringVar(value="Voice Status: Not Cloned")
    Label(root, textvariable=voice_status_var).pack(padx=10, pady=5)

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
        root.after(1000, refresh_scheduled_list)

    def remove_selected_question():
        selected = scheduled_listbox.curselection()
        if selected:
            index = selected[0]
            scheduler.remove_question(index)
            print(f"[INFO] Removed scheduled question at index {index+1}.")
            speak("Scheduled question removed successfully.")

    Button(root, text="Remove Selected Question", command=remove_selected_question).pack(padx=10, pady=5)
    refresh_scheduled_list()
    return root

def set_emergency_email(email_var, email_entry):
    global emergency_email
    email = email_var.get().strip()
    if email:
        if "@" in email and "." in email:
            emergency_email = email
            print(f"[INFO] Emergency email set to: {emergency_email}")
            voice_status_var.set("Emergency Email Set")
            speak("Emergency email set successfully.")
            email_entry.delete(0, END)
        else:
            print("[ERROR] Invalid email format.")
            speak("Please enter a valid email address.")
    else:
        print("[ERROR] Email field is empty.")
        speak("Please enter an email address.")

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

schedule_thread = Thread(target=lambda: schedule_gui().mainloop(), daemon=True)
schedule_thread.start()

scheduled_question_active = False
global context
context = None

multi_models = None 

def train_all_models(calibration_data):
    """
    Trains multiple regression models (OLD linear regression via np.linalg.lstsq, plus RF, GBR, KNN) for X and Y
    using the 2D input (normalized_x, normalized_y).
    """
    print("[INFO] Training multiple ML models (Linear Regression, RF, GBR, KNN) with 2D input...")
    
    X = np.array([data[0] for data in calibration_data]) 
    y_x = np.array([data[1][0] for data in calibration_data])  # screen_x
    y_y = np.array([data[1][1] for data in calibration_data])  # screen_y

    models = {}

    
    class LstsqModel:
        def __init__(self):
            self.coeffs = None
            self.scaler = StandardScaler()  
        def fit(self, X_train, y_train):
            
            X_scaled = self.scaler.fit_transform(X_train)
            
            A = np.column_stack((X_scaled, np.ones(len(X_scaled))))
            self.coeffs, _, _, _ = np.linalg.lstsq(A, y_train, rcond=None)
        def predict(self, X_in):
            X_scaled = self.scaler.transform(X_in)
            A = np.column_stack((X_scaled, np.ones(len(X_scaled))))
            return A.dot(self.coeffs)

    # 1.np.linalg.lstsq
    lr_x = LstsqModel()
    lr_y = LstsqModel()
    lr_x.fit(X, y_x)
    lr_y.fit(X, y_y)
    models['LR_X'] = lr_x
    models['LR_Y'] = lr_y

    # 2. Random Forest
    rf_x = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42))
    rf_y = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42))
    rf_x.fit(X, y_x)
    rf_y.fit(X, y_y)
    models['RF_X'] = rf_x
    models['RF_Y'] = rf_y

    # 3. Gradient Boosting
    gbr_x = make_pipeline(StandardScaler(), GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    gbr_y = make_pipeline(StandardScaler(), GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    gbr_x.fit(X, y_x)
    gbr_y.fit(X, y_y)
    models['GBR_X'] = gbr_x
    models['GBR_Y'] = gbr_y

    # 4. KNN
    knn_x = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5))
    knn_y = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5))
    knn_x.fit(X, y_x)
    knn_y.fit(X, y_y)
    models['KNN_X'] = knn_x
    models['KNN_Y'] = knn_y

    print("[INFO] All models trained successfully.")
    return models

model_positions = {
    'LR': None,
    'RF': None,
    'GBR': None,
    'KNN': None
}
model_colors = {
    'LR': (255, 0, 0),    # Blue 
    'RF': (0, 255, 0),    # Green
    'GBR': (0, 0, 255),   # Red
    'KNN': (0, 255, 255)  # Yellow
}

def handle_question():
    
    global arrayOfResponses, context
    arrayOfResponses = []
    context = transcription_phase()
    if context:
        print(f"[INFO] Transcribed Question: {context}")

       
        answers, sims, methods = get_top_responses(context, top_n=4, similarity_threshold=0.55, diversity_threshold=0.7)

        if len(answers) < 4:
            print("[WARNING] Fallback: Not enough answers found. Filling with placeholders.")
            while len(answers) < 4:
                answers.append("N/A")

        arrayOfResponses.extend(answers)
        print(f"[INFO] Final set of responses: {arrayOfResponses}")
    else:
        print("[ERROR] No transcription available.")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("[ERROR] Failed to capture frame from webcam.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    
    cv2.rectangle(frame, (TOP_BOUNDS[0], TOP_BOUNDS[2]), (TOP_BOUNDS[1], TOP_BOUNDS[3]), (0, 255, 0), 2)
    cv2.rectangle(frame, (LEFT_BOUNDS[0], LEFT_BOUNDS[2]), (LEFT_BOUNDS[1], LEFT_BOUNDS[3]), (255, 0, 0), 2)
    cv2.rectangle(frame, (BOTTOM_BOUNDS[0], BOTTOM_BOUNDS[2]), (BOTTOM_BOUNDS[1], BOTTOM_BOUNDS[3]), (0, 0, 255), 2)
    cv2.rectangle(frame, (RIGHT_BOUNDS[0], RIGHT_BOUNDS[2]), (RIGHT_BOUNDS[1], RIGHT_BOUNDS[3]), (255, 255, 0), 2)

   
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

            
            if len(calibration_data) >= 3:
                multi_models = train_all_models(calibration_data)
            
            continue

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

                try:
                    screen_x, screen_y = map_gaze_to_screen(normalized_x, normalized_y, calibration_data)

                    if multi_models is not None:
                        
                        X_in = np.array([[normalized_x, normalized_y]]) 

                        
                        lr_x = multi_models['LR_X'].predict(X_in)[0]
                        lr_y = multi_models['LR_Y'].predict(X_in)[0]
                        if model_positions['LR'] is None:
                            model_positions['LR'] = (int(lr_x), int(lr_y))
                        else:
                            old_pos = model_positions['LR']
                            new_x = int(old_pos[0] * (1 - smoothing_factor) + lr_x * smoothing_factor)
                            new_y = int(old_pos[1] * (1 - smoothing_factor) + lr_y * smoothing_factor)
                            model_positions['LR'] = (new_x, new_y)

                       
                        rf_x = multi_models['RF_X'].predict(X_in)[0]
                        rf_y = multi_models['RF_Y'].predict(X_in)[0]
                        if model_positions['RF'] is None:
                            model_positions['RF'] = (int(rf_x), int(rf_y))
                        else:
                            old_pos = model_positions['RF']
                            new_x = int(old_pos[0] * (1 - smoothing_factor) + rf_x * smoothing_factor)
                            new_y = int(old_pos[1] * (1 - smoothing_factor) + rf_y * smoothing_factor)
                            model_positions['RF'] = (new_x, new_y)

                        
                        gbr_x = multi_models['GBR_X'].predict(X_in)[0]
                        gbr_y = multi_models['GBR_Y'].predict(X_in)[0]
                        if model_positions['GBR'] is None:
                            model_positions['GBR'] = (int(gbr_x), int(gbr_y))
                        else:
                            old_pos = model_positions['GBR']
                            new_x = int(old_pos[0] * (1 - smoothing_factor) + gbr_x * smoothing_factor)
                            new_y = int(old_pos[1] * (1 - smoothing_factor) + gbr_y * smoothing_factor)
                            model_positions['GBR'] = (new_x, new_y)

                       
                        knn_x = multi_models['KNN_X'].predict(X_in)[0]
                        knn_y = multi_models['KNN_Y'].predict(X_in)[0]
                        if model_positions['KNN'] is None:
                            model_positions['KNN'] = (int(knn_x), int(knn_y))
                        else:
                            old_pos = model_positions['KNN']
                            new_x = int(old_pos[0] * (1 - smoothing_factor) + knn_x * smoothing_factor)
                            new_y = int(old_pos[1] * (1 - smoothing_factor) + knn_y * smoothing_factor)
                            model_positions['KNN'] = (new_x, new_y)

                       
                        for m_name, m_pos in model_positions.items():
                            if m_pos is not None:
                                draw_blob(frame, m_pos, blob_radius, model_colors[m_name])

                    
                    if multi_models is not None and model_positions['LR'] is not None:
                        primary_pos = model_positions['LR']
                        in_top = is_in_bounds(primary_pos[0], primary_pos[1], TOP_BOUNDS)
                        in_left = is_in_bounds(primary_pos[0], primary_pos[1], LEFT_BOUNDS)
                        in_bottom = is_in_bounds(primary_pos[0], primary_pos[1], BOTTOM_BOUNDS)
                        in_right = is_in_bounds(primary_pos[0], primary_pos[1], RIGHT_BOUNDS)
                    else:
                        
                        in_top = is_in_bounds(screen_x, screen_y, TOP_BOUNDS)
                        in_left = is_in_bounds(screen_x, screen_y, LEFT_BOUNDS)
                        in_bottom = is_in_bounds(screen_x, screen_y, BOTTOM_BOUNDS)
                        in_right = is_in_bounds(screen_x, screen_y, RIGHT_BOUNDS)

                    

                    
                    if in_top:
                        if calibrated_vertical_direction != opposite_directions["up"]:
                            direction_timer["up"] += 1 / 30
                            print(f"[DEBUG] direction_timer['up']: {direction_timer['up']}")
                            draw_progress_bar(frame, (SCREEN_WIDTH // 2 - 100, 10),
                                              direction_timer["up"], AGREEMENT_THRESHOLD)
                            if direction_timer["up"] >= AGREEMENT_THRESHOLD:
                                if len(arrayOfResponses) > 0:
                                    selected_response = arrayOfResponses[0]
                                    print(f"User is looking at the TOP element: {selected_response}")
                                    direction_timer["up"] = 0

                                   
                                    add_qa_to_db(context, selected_response)

                                    
                                    prompt = (
                                        f"Given this question, {context}, here is a one-word or short phrase answer: {selected_response}. "
                                        f"Please provide a short answer that uses that word/phrase in the context of answering the question. Keep it brief."
                                    )
                                    print(f"Prompt being sent to ChatGPT: {prompt}")
                                    try:
                                        chat_response = client.chat.completions.create(
                                            model=model,
                                            messages=[
                                                {"role": "system", "content": "You are a helpful assistant."},
                                                {"role": "user", "content": prompt}
                                            ]
                                        )
                                        assistant_reply = chat_response.choices[0].message.content.strip()
                                        print(f"Assistant Reply: {assistant_reply}")
                                        speak(assistant_reply)
                                       
                                        if scheduled_question_active:
                                            analysis_prompt = (
                                                f"This is a question that an ALS patient responded to: {context}. "
                                                f"Their chosen answer is: {selected_response}. "
                                                f"Should a caregiver be alerted? Return 'True' or 'False' in one word."
                                            )
                                            try:
                                                analysis_response = client.chat.completions.create(
                                                    model=model,
                                                    messages=[
                                                        {"role": "system", "content": "You are a helpful assistant."},
                                                        {"role": "user", "content": analysis_prompt}
                                                    ]
                                                )
                                                analysis_reply = analysis_response.choices[0].message.content.strip().lower()
                                                print(f"Analysis Reply: {analysis_reply}")
                                                if analysis_reply == "true":
                                                    send_email_for_help(emergency_email)
                                            except Exception as e:
                                                print(f"[ERROR] ChatGPT API call for analysis failed: {e}")
                                        scheduled_question_active = False
                                    except Exception as e:
                                        print(f"[ERROR] ChatGPT API call failed: {e}")
                        else:
                            direction_timer["up"] = 0
                    else:
                        direction_timer["up"] = 0

                    
                    if in_bottom:
                        if calibrated_vertical_direction != opposite_directions["down"]:
                            direction_timer["down"] += 1 / 30
                            print(f"[DEBUG] direction_timer['down']: {direction_timer['down']}")
                            draw_progress_bar(frame, (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT - 30),
                                              direction_timer["down"], AGREEMENT_THRESHOLD)
                            if direction_timer["down"] >= AGREEMENT_THRESHOLD:
                                if len(arrayOfResponses) > 2:
                                    selected_response = arrayOfResponses[2]
                                    print(f"User is looking at the BOTTOM element: {selected_response}")
                                    direction_timer["down"] = 0

                                    add_qa_to_db(context, selected_response)

                                    prompt = (
                                        f"Given this question, {context}, here is a one-word or short phrase answer: {selected_response}. "
                                        f"Please provide a short single-sentence answer that uses that word. Keep it brief."
                                    )
                                    print(f"Prompt being sent to ChatGPT: {prompt}")
                                    try:
                                        chat_response = client.chat.completions.create(
                                            model=model,
                                            messages=[
                                                {"role": "system", "content": "You are a helpful assistant."},
                                                {"role": "user", "content": prompt}
                                            ]
                                        )
                                        assistant_reply = chat_response.choices[0].message.content.strip()
                                        print(f"Assistant Reply: {assistant_reply}")
                                        speak(assistant_reply)

                                        if scheduled_question_active:
                                            analysis_prompt = (
                                                f"This is a question that an ALS patient responded to: {context}. "
                                                f"Their chosen answer is: {selected_response}. "
                                                f"Should a caregiver be alerted? Return 'True' or 'False'."
                                            )
                                            try:
                                                analysis_response = client.chat.completions.create(
                                                    model=model,
                                                    messages=[
                                                        {"role": "system", "content": "You are a helpful assistant."},
                                                        {"role": "user", "content": analysis_prompt}
                                                    ]
                                                )
                                                analysis_reply = analysis_response.choices[0].message.content.strip().lower()
                                                print(f"Analysis Reply: {analysis_reply}")
                                                if analysis_reply == "true":
                                                    send_email_for_help(emergency_email)
                                            except Exception as e:
                                                print(f"[ERROR] ChatGPT API call for analysis failed: {e}")
                                        scheduled_question_active = False
                                    except Exception as e:
                                        print(f"[ERROR] ChatGPT API call failed: {e}")
                        else:
                            direction_timer["down"] = 0
                    else:
                        direction_timer["down"] = 0

                    
                    if in_left:
                        if horizontal_direction != opposite_directions["left"]:
                            direction_timer["left"] += 1 / 30
                            print(f"[DEBUG] direction_timer['left']: {direction_timer['left']}")
                            draw_progress_bar(frame, (10, SCREEN_HEIGHT // 2 - 10),
                                              direction_timer["left"], AGREEMENT_THRESHOLD)
                            if direction_timer["left"] >= AGREEMENT_THRESHOLD:
                                if len(arrayOfResponses) > 1:
                                    selected_response = arrayOfResponses[1]
                                    print(f"User is looking at the LEFT element: {selected_response}")
                                    direction_timer["left"] = 0

                                    add_qa_to_db(context, selected_response)

                                    prompt = (
                                        f"Given this question, {context}, here is a one-word or short phrase answer: {selected_response}. "
                                        f"Please provide a short single-sentence answer that uses that word. Keep it brief."
                                    )
                                    print(f"Prompt being sent to ChatGPT: {prompt}")
                                    try:
                                        chat_response = client.chat.completions.create(
                                            model=model,
                                            messages=[
                                                {"role": "system", "content": "You are a helpful assistant."},
                                                {"role": "user", "content": prompt}
                                            ]
                                        )
                                        assistant_reply = chat_response.choices[0].message.content.strip()
                                        print(f"Assistant Reply: {assistant_reply}")
                                        speak(assistant_reply)

                                        if scheduled_question_active:
                                            analysis_prompt = (
                                                f"This is a question that an ALS patient responded to: {context}. "
                                                f"Their chosen answer is: {selected_response}. "
                                                f"Should a caregiver be alerted? Return 'True' or 'False'."
                                            )
                                            try:
                                                analysis_response = client.chat.completions.create(
                                                    model=model,
                                                    messages=[
                                                        {"role": "system", "content": "You are a helpful assistant."},
                                                        {"role": "user", "content": analysis_prompt}
                                                    ]
                                                )
                                                analysis_reply = analysis_response.choices[0].message.content.strip().lower()
                                                print(f"Analysis Reply: {analysis_reply}")
                                                if analysis_reply == "true":
                                                    send_email_for_help(emergency_email)
                                            except Exception as e:
                                                print(f"[ERROR] ChatGPT API call for analysis failed: {e}")
                                        scheduled_question_active = False
                                    except Exception as e:
                                        print(f"[ERROR] ChatGPT API call failed: {e}")
                        else:
                            direction_timer["left"] = 0
                    else:
                        direction_timer["left"] = 0

                    
                    if in_right:
                        if horizontal_direction != opposite_directions["right"]:
                            direction_timer["right"] += 1 / 30
                            print(f"[DEBUG] direction_timer['right']: {direction_timer['right']}")
                            draw_progress_bar(frame, (SCREEN_WIDTH - 220, SCREEN_HEIGHT // 2 - 10),
                                              direction_timer["right"], AGREEMENT_THRESHOLD)
                            if direction_timer["right"] >= AGREEMENT_THRESHOLD:
                                if len(arrayOfResponses) > 3:
                                    selected_response = arrayOfResponses[3]
                                    print(f"User is looking at the RIGHT element: {selected_response}")
                                    direction_timer["right"] = 0

                                    add_qa_to_db(context, selected_response)

                                    prompt = (
                                        f"Given this question, {context}, here is a one-word or short phrase answer: {selected_response}. "
                                        f"Please provide a short single-sentence answer that uses that word. Keep it brief."
                                    )
                                    print(f"Prompt being sent to ChatGPT: {prompt}")
                                    try:
                                        chat_response = client.chat.completions.create(
                                            model=model,
                                            messages=[
                                                {"role": "system", "content": "You are a helpful assistant."},
                                                {"role": "user", "content": prompt}
                                            ]
                                        )
                                        assistant_reply = chat_response.choices[0].message.content.strip()
                                        print(f"Assistant Reply: {assistant_reply}")
                                        speak(assistant_reply)

                                        if scheduled_question_active:
                                            analysis_prompt = (
                                                f"This is a question that an ALS patient responded to: {context}. "
                                                f"Their chosen answer is: {selected_response}. "
                                                f"Should a caregiver be alerted? Return 'True' or 'False'."
                                            )
                                            try:
                                                analysis_response = client.chat.completions.create(
                                                    model=model,
                                                    messages=[
                                                        {"role": "system", "content": "You are a helpful assistant."},
                                                        {"role": "user", "content": analysis_prompt}
                                                    ]
                                                )
                                                analysis_reply = analysis_response.choices[0].message.content.strip().lower()
                                                print(f"Analysis Reply: {analysis_reply}")
                                                if analysis_reply == "true":
                                                    send_email_for_help(emergency_email)
                                            except Exception as e:
                                                print(f"[ERROR] ChatGPT API call for analysis failed: {e}")
                                        scheduled_question_active = False
                                    except Exception as e:
                                        print(f"[ERROR] ChatGPT API call failed: {e}")
                        else:
                            direction_timer["right"] = 0
                    else:
                        direction_timer["right"] = 0

                except ValueError:
                    pass

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        print("[INFO] Space key pressed. Listening for question...")
        question_thread = Thread(target=handle_question)
        question_thread.start()
    if key == ord('q'):
        print("[INFO] Exiting program.")
        break

    cv2.namedWindow("Eye Gaze Tracker", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Eye Gaze Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Eye Gaze Tracker", frame)

cap.release()
cv2.destroyAllWindows()


scheduler.stop()
scheduler_thread.join()


