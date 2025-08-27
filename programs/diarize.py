from pyannote.audio import Pipeline
import os
from dotenv import load_dotenv
from pathlib import Path

ROOT_FOLDER = Path(__file__).parent.parent
INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"
audio_input_filename = "conversation.wav"
diarization_output_filename = "diarization.rttm"

load_dotenv(ROOT_FOLDER / ".env") 

audio_file_path = os.path.join(ROOT_FOLDER, INPUT_FOLDER, audio_input_filename) 
diarization_output_path = os.path.join(ROOT_FOLDER, OUTPUT_FOLDER, diarization_output_filename)

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=os.getenv("HF_TOKEN")
)

# import torch
# pipeline.to(torch.device("cuda"))

# run the pipeline on an audio file
diarization = pipeline(audio_file_path)

print("\n--- Speaker Diarization ---")
for turn, _, speaker in diarization.itertracks(yield_label=True):
    start_time = turn.start
    end_time = turn.end
    print(f"[{start_time:.2f}s --> {end_time:.2f}s] Speaker_{speaker}")

# or, dump the diarization output to disk using RTTM format
with open(diarization_output_path, "w") as rttm:
    diarization.write_rttm(rttm)