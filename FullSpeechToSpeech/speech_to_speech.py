from collections.abc import Iterator
from faster_whisper import WhisperModel
from ollama import ChatResponse
import sounddevice
import numpy as np
from pynput import keyboard
from ollama import chat
import time

sample_rate = 16000  # 16kHz

speech_to_text_model = WhisperModel(
    "small", device="cpu", compute_type="int8"
)  # could be "medium", "large-v3"

stop_recording_flag = False

audio_buffer = bytearray()


def audio_callback(indata, frames, time, status):
    global audio_buffer
    audio_bytes = indata.tobytes()
    audio_buffer.extend(audio_bytes)


def on_press(key):
    global stop_recording_flag
    stop_recording_flag = True
    return False


def get_transcribed_mic_input() -> str:
    input("🎤 Press Enter to start recording...")
    print("🎤 Recording. Press Enter to stop...")
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    with sounddevice.InputStream(
        samplerate=sample_rate, channels=1, dtype="int16", callback=audio_callback
    ):
        while not stop_recording_flag:
            pass

    audio_np = (
        np.frombuffer(bytes(audio_buffer), dtype=np.int16).astype(np.float32) / 32768.0
    )

    segments, info = speech_to_text_model.transcribe(
        audio_np, beam_size=5, language="en", vad_filter=True
    )

    text = " ".join([seg.text for seg in segments])

    audio_buffer.clear()

    return text


def stream_ai_response(prompt: str) -> Iterator[ChatResponse]:
    stream = chat(
        model="phi3:mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    return stream


def play_ai_response_audio(stream: Iterator[ChatResponse]):
    print("AI response: ", end="")
    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)
    print()


def main():
    while True:
        mic_input = get_transcribed_mic_input()
        if mic_input:
            print(f"Received user input: {mic_input}")
        else:
            print("No user input received")
            continue

        ai_response_stream = stream_ai_response(mic_input)
        play_ai_response_audio(ai_response_stream)
        mic_input = ""


if __name__ == "__main__":
    main()
