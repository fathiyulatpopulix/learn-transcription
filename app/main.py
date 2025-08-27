import torch
from pyannote.audio import Pipeline
import whisper
import librosa
import os
from dotenv import load_dotenv
from pathlib import Path

ROOT_FOLDER = Path(__file__).parent.parent
INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"
audio_input_filename = "conversation.wav"
output_filename = "diarized-transcription.txt"

load_dotenv(ROOT_FOLDER / ".env")

# check if output folder exists, if not create it
if not os.path.exists(os.path.join(ROOT_FOLDER, OUTPUT_FOLDER)):
    os.makedirs(os.path.join(ROOT_FOLDER, OUTPUT_FOLDER))

device = "cpu"
torch.set_num_threads(4)

# --- Part 0: Setup ---
# Replace with your actual Hugging Face token
hf_token = os.getenv("HF_TOKEN")
file_path = os.path.join(ROOT_FOLDER, INPUT_FOLDER, audio_input_filename)

# --- Part 1: Speaker Diarization ---
print("Step 1: Performing Speaker Diarization...")
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
)
diarization = diarization_pipeline(file_path)
print("Diarization complete.")

# --- Part 2: Audio Transcription ---
print("\nStep 2: Loading Transcription Model...")
# Load the Whisper model. "base" is a good starting point.
# Other options: "tiny", "small", "medium", "large"
transcription_model = whisper.load_model("base", device=device)
print("Transcription model loaded.")

# --- Part 3: Process and Transcribe Each Segment ---
print("\n--- Full Transcription ---")
# Load the audio file with librosa to easily slice it
y, sr = librosa.load(file_path, sr=16000)  # Whisper works best with 16kHz audio

output_file_path = os.path.join(ROOT_FOLDER, OUTPUT_FOLDER, output_filename)

with open(output_file_path, "w") as output_file:
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # Define the start and end times of the segment
        start_time = turn.start
        end_time = turn.end

        # Extract the audio segment
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment = y[start_sample:end_sample]

        # Transcribe the segment
        # We need to convert our numpy array back to a tensor for Whisper
        result = transcription_model.transcribe(
            segment, fp16=False
        )  # Set fp16=False if not using a GPU

        # Format the output string
        output_string = f"[{start_time:.2f}s --> {end_time:.2f}s] {speaker}: {result['text'].strip()}\n"

        # Print the result to the console
        print(output_string.strip())

        # Write the result to the output file
        output_file.write(output_string)

print(f"\nTranscription saved to {output_file_path}")
