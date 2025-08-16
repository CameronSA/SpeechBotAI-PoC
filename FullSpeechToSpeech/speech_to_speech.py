from collections.abc import Iterator
from faster_whisper import WhisperModel
from ollama import ChatResponse
from piper import PiperVoice
import sounddevice
import numpy as np
from pynput import keyboard
from ollama import chat
import time

sample_rate = 16000  # 16kHz

speech_to_text_model = WhisperModel(
    "small", device="cpu", compute_type="int8"
)  # could be "medium", "large-v3"

audio_buffer = bytearray()


def audio_callback(indata, frames, time, status):
    global audio_buffer
    audio_bytes = indata.tobytes()
    audio_buffer.extend(audio_bytes)


def get_transcribed_mic_input() -> str:
    input("🎤 Press Enter to start recording...")

    with sounddevice.InputStream(
        samplerate=sample_rate, channels=1, dtype="int16", callback=audio_callback
    ):
        input("🎤 Recording. Press Enter to stop...")

    audio_np = (
        np.frombuffer(bytes(audio_buffer), dtype=np.int16).astype(np.float32) / 32768.0
    )

    segments, info = speech_to_text_model.transcribe(
        audio_np, beam_size=5, language="en", vad_filter=True
    )

    text = " ".join([seg.text for seg in segments])

    return text


def stream_ai_response(prompt: str) -> Iterator[ChatResponse]:
    message = {"role": "user", "content": prompt}
    stream = chat(
        model="phi3:mini",
        messages=[message],
        stream=True,
    )

    return stream


def play_ai_response_audio(stream: Iterator[ChatResponse], voice: PiperVoice):
    print("AI response: ", end="")
    sentence = ""
    break_points = [".", "!", "?"]
    for chunk in stream:
        text = chunk["message"]["content"]
        sentence += text
        print(text, end="", flush=True)
        break_point = any(bp in text for bp in break_points)
        if break_point:
            audio_chunks = voice.synthesize(sentence)
            for chunk in audio_chunks:
                sounddevice.play(chunk.audio_int16_array, samplerate=22050)
                sounddevice.wait()
            sentence = ""

    print()


def main():
    global audio_buffer
    voice = PiperVoice.load("en_GB-northern_english_male-medium.onnx", use_cuda=False)
    while True:
        mic_input = get_transcribed_mic_input()
        if mic_input:
            print(f"Received user input: {mic_input}")
        else:
            print("No user input received")
            continue

        ai_response_stream = stream_ai_response(mic_input)
        play_ai_response_audio(ai_response_stream, voice)
        mic_input = ""
        audio_buffer.clear()


if __name__ == "__main__":
    main()
