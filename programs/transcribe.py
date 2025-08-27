import whisper
import json
from pathlib import Path
import os

ROOT_FOLDER = Path(__file__).parent.parent
INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"
audio_input_filename = "conversation.wav"
transcription_output_filename = "transcription.json"

# --- Part 1: Setup ---
# Use the long conversation file we downloaded
audio_file_path = os.path.join(ROOT_FOLDER, INPUT_FOLDER, audio_input_filename)
transcription_output_path = os.path.join(
    ROOT_FOLDER, OUTPUT_FOLDER, transcription_output_filename
)
# Force CPU to prevent the CUDA error you saw before
device = "cpu"

# check if output folder exists, if not create it
if not os.path.exists(os.path.join(ROOT_FOLDER, OUTPUT_FOLDER)):
    os.makedirs(os.path.join(ROOT_FOLDER, OUTPUT_FOLDER))

# --- Part 2: Load the Model ---
print("Loading Whisper model (base)...")
# Using "base" model. Other options: "tiny", "small", "medium", "large"
model = whisper.load_model("base", device=device)
print("Model loaded.")

# --- Part 3: Transcribe the Audio ---
print(f"\nTranscribing {audio_file_path}...")
# The transcribe function does all the work
result = model.transcribe(audio_file_path)
print("Transcription complete.")

# --- Part 4: Print the Results ---
print("\n--- Full Transcription Text ---")
# The 'text' key contains the entire transcription as one block
print(result["text"])

# Optional: Print with timestamps for each segment Whisper found
print("\n--- Transcription with Timestamps (No Speaker Labels) ---")
for segment in result["segments"]:
    start = segment["start"]
    end = segment["end"]
    text = segment["text"]
    print(f"[{start:.2f}s --> {end:.2f}s] {text.strip()}")

# Export result to JSON file


with open(transcription_output_path, "w") as f:
    json.dump(result, f, indent=4)
