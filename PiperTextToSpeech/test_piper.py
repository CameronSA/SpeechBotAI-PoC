import wave
from piper import PiperVoice
import sounddevice
import numpy as np

if __name__ == "__main__":
    stringArr = [
        "Hello!",
        "This",
        "is",
        "your",
        "computer",
        "speaking.",
        "How",
        "can",
        "I",
        "assist",
        "you",
        "today?",
    ]

    # Save to Wav
    voice = PiperVoice.load("en_GB-northern_english_male-medium.onnx", use_cuda=False)
    with wave.open("test.wav", "wb") as wav_file:
        voice.synthesize_wav("Welcome to the world of speech synthesis!", wav_file)

    # Play audio
    audio_chunks = voice.synthesize(
        "Hello! This is your computer speaking. How can I assist you today?"
    )
    for chunk in audio_chunks:
        sounddevice.play(chunk.audio_int16_array, samplerate=22050)
        sounddevice.wait()

    # Test streaming synthesis - chunk by sentences
    sentence = ""
    break_points = [".", "!", "?"]
    for text in stringArr:
        sentence += text + " "
        break_point = any(bp in text for bp in break_points)
        if break_point:
            audio_chunks = voice.synthesize(sentence)
            for chunk in audio_chunks:
                sounddevice.play(chunk.audio_int16_array, samplerate=22050)
                sounddevice.wait()
            sentence = ""

    # Test streaming synthesis - chunk by words
    for text in stringArr:
        audio_chunks = voice.synthesize(text)
        for chunk in audio_chunks:
            sounddevice.play(chunk.audio_int16_array, samplerate=22050)
            sounddevice.wait()
