# import whisper

# whisper_model = whisper.load_model("base")
# result = whisper_model.transcribe("D:/Virtual Assistent/Runpod/file1.mp3")
# print(result["text"])


# import whisper
# import psutil
# import time

# def log_usage(step):
#     cpu_usage = psutil.cpu_percent(interval=1)
#     ram_usage = psutil.virtual_memory().used / (1024 ** 2)
#     print(f"[{step}] CPU Usage: {cpu_usage}% | RAM Usage: {ram_usage} MB")

# # Load model
# log_usage("Before model loading")
# whisper_model = whisper.load_model("base")
# log_usage("After model loading")

# # Transcribe audio
# log_usage("Before transcription")
# result = whisper_model.transcribe("D:/Virtual Assistent/Runpod/file1.mp3")
# log_usage("After transcription")

# # Print the result
# print(result["text"])

### Whisper Transcription API

import os

# Directly assign the API key in the script
os.environ['OPENAI_API_KEY'] = 'XXX'

# Now you can use the API key
api_key = os.getenv("OPENAI_API_KEY")
print("API Key Loaded:", api_key)



from openai import OpenAI
client = OpenAI()

audio_file= open("D:/Virtual Assistent/Runpod/file1.mp3", "rb")
transcription = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file
)
print(transcription.text)


## TTS
from pathlib import Path
from openai import OpenAI
client = OpenAI()

speech_file_path = Path(__file__).parent / "speech.mp3"
response = client.audio.speech.create(
  model="tts-1",
  voice="alloy",
  input="Today is a wonderful day to build something people love!"
)

response.stream_to_file(speech_file_path)


## Combine Pipeline


from pathlib import Path
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI()

# Step 1: Transcribe the audio file
audio_file_path = "D:/Virtual Assistent/Runpod/file1.mp3"
with open(audio_file_path, "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )

# Print the transcription text
print("Transcription:", transcription.text)

# Step 2: Convert the transcription text back into speech
speech_file_path = Path(__file__).parent / "transcribed_speech.mp3"
response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input=transcription.text
)

# Save the generated speech to a file
response.stream_to_file(speech_file_path)

print(f"Generated speech saved to {speech_file_path}")
