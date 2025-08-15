from faster_whisper import WhisperModel
from fastapi import FastAPI, File, UploadFile
import uvicorn

app = FastAPI()
model = WhisperModel(
    "small", device="cpu", compute_type="int8"
)  # could be "medium", "large-v3"


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    file_name = "mic_input.wav"
    with open(file_name, "wb") as f:
        content = await file.read()
        f.write(content)

    segments, info = model.transcribe(
        file_name, beam_size=5, language="en", vad_filter=True
    )
    text = " ".join([seg.text for seg in segments])
    print(text)
    return {"text": text}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
