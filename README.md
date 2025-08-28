# Speaker Diarization and Transcription Pipeline

This project demonstrates a complete pipeline for automatic speech recognition (ASR) and speaker diarization. Given a single audio file containing a conversation, this script will:

1.  Identify how many speakers are present.
2.  Determine "who spoke when" (diarization).
3.  Transcribe what each speaker said (transcription).

This is achieved by combining two powerful, pre-trained models:

- **`pyannote.audio`**: For state-of-the-art speaker diarization.
- **`openai-whisper`**: For accurate speech-to-text transcription.

---

## Setup and Installation

Follow these steps carefully to set up the environment and dependencies.

### 1. Install Dependencies

This project uses `uv` for Python package management. First, ensure you have Python and `uv` installed.

Initialize a new project and add the necessary dependencies:

```bash
# Initialize a new uv project (if not already done)
uv init

# Sync the environment
uv sync
```

### 2. Get Hugging Face Access Token

The pre-trained models are hosted on Hugging Face and require authentication.

1.  **Create a free Hugging Face account:** [huggingface.co](https://huggingface.co)
2.  **Generate an Access Token:** Go to your **Settings -> Access Tokens** page and create a new token with "read" permissions. Copy this token.
    - Direct Link: <https://huggingface.co/settings/tokens>
3.  **Log in from your terminal:** Run the following command and paste your token when prompted.
    ```bash
    huggingface-cli login
    ```

### 3. Accept Model User Agreements

The specific models used in this pipeline are "gated." You must visit their pages and accept the terms and conditions before your account is authorized to download them.

- **Visit and accept the terms for both models below:**
  1.  Diarization Pipeline: <https://huggingface.co/pyannote/speaker-diarization-3.1>
  2.  Segmentation Model (Dependency): <https://huggingface.co/pyannote/segmentation-3.0>

> **Note:** The script will fail with a `401 Unauthorized` error if this step is missed.

### 4. Download Sample Audio File

A sample one-minute conversation file is provided for testing. Download it using the following command:

```bash
mkdir -p input
wget -O input/conversation.wav "https://github.com/pyannote/pyannote-audio/raw/develop/tutorials/assets/sample.wav"
```

---

### 5. Create `.env` File

Create a `.env` file in the root directory of your project. This file will store your Hugging Face token.

1.  **Copy the example:**
    ```bash
    cp .env.example .env
    ```
2.  **Edit `.env`:** Open the `.env` file and add your Hugging Face token:
    ```
    HF_TOKEN="YOUR_HUGGING_FACE_TOKEN"
    ```
    Replace `YOUR_HUGGING_FACE_TOKEN` with the actual token you obtained in Step 2.

## Running the Program

Run the program from your terminal:

```bash
uv run app/main.py
```

The first time you run the script, it will download the necessary models (a few hundred MB). Subsequent runs will be faster. The process can take some time, especially on a CPU.

### 4. Understand the Output

The script will print a timestamped and speaker-labeled transcript of the entire conversation, like this:

```
--- Full Transcription ---
[0.21s --> 1.21s] SPEAKER_00: Hello.
[1.34s --> 2.22s] SPEAKER_01: Oh, hello.
[2.35s --> 3.45s] SPEAKER_01: I didn't know you were there.
...
```
