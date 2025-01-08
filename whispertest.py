import collections
import contextlib
import wave
import pyaudio
import webrtcvad
import time
import whisper

# Parameters
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1              # Mono audio
RATE = 16000              # Sampling rate
CHUNK_DURATION_MS = 30    # Duration of a chunk in milliseconds
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)  # Chunk size
SILENCE_THRESHOLD = 3     # Silence threshold in seconds
RECORDING_TIMEOUT = 60    # Maximum recording duration in seconds

def countdown(seconds):
    """Prints a countdown from the specified number of seconds."""
    for i in range(seconds, 0, -1):
        print(f"Starting in {i} seconds...")
        time.sleep(1)
    print("Recording started. Please speak now.")

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

if __name__ == "__main__":
    countdown(5)  # 5-second countdown
    recorder = AudioRecorder()
    try:
        recorder.record()
        transcription = transcribe_audio("output.wav")
        print("Transcription:")
        print(transcription)
    except KeyboardInterrupt:
        print("Recording interrupted.")
    finally:
        recorder.terminate()
