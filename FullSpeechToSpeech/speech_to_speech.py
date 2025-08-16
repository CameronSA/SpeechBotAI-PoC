import sounddevice
from faster_whisper import WhisperModel
from fastapi import FastAPI, File, UploadFile
import time
import threading
import numpy as np

model = WhisperModel(
    "small", device="cpu", compute_type="int8"
)  # could be "medium", "large-v3"

buffer_lock = threading.Lock()
audio_buffer = bytearray()
sample_rate = 16000  # 16kHz
min_bytes = sample_rate * 2 * 0.5  # 0.5 seconds of audio in bytes


def audio_callback(indata, frames, time, status):
    audio_bytes = indata.tobytes()
    with buffer_lock:
        audio_buffer.extend(audio_bytes)


def transcribe(transcribe_buffer: bytearray):
    audio_np = (
        np.frombuffer(bytes(transcribe_buffer), dtype=np.int16).astype(np.float32)
        / 32768.0
    )

    segments, info = model.transcribe(
        audio_np, beam_size=5, language="en", vad_filter=True
    )
    text = " ".join([seg.text for seg in segments])
    if text:
        print(text)


def speech_detected():
    if len(audio_buffer) < min_bytes:
        return

    # Convert to float and normalize
    audio_np = (
        np.frombuffer(bytes(audio_buffer), dtype=np.int16).astype(np.float32) / 32768.0
    )

    return np.max(np.abs(audio_np)) >= 0.3


def stream_microphone_input():
    # 16kHz, mono
    with sounddevice.InputStream(
        samplerate=16000, channels=1, dtype="int16", callback=audio_callback
    ):
        print("Listening for audio input...")
        transcribe_buffer = bytearray()
        receiving_input = False
        last_input_time = time.time()

        while True:
            if receiving_input:
                # Check if silence lasted for 1 seconds
                if time.time() - last_input_time > 1:
                    print("Silence detected. Transcribing...")
                    transcribe(transcribe_buffer)
                    transcribe_buffer.clear()
                    receiving_input = False
                    continue

            # If audio is loud enough, treat as speech
            if speech_detected():
                transcribe_buffer.extend(audio_buffer)
                receiving_input = True
                last_input_time = time.time()
                print("Speech detected")
                with buffer_lock:
                    audio_buffer.clear()


def main():
    stream_microphone_input()


if __name__ == "__main__":
    main()
