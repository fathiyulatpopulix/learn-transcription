import streamlit as st
import torch
from pyannote.audio import Pipeline
import whisper
import librosa
import os
from dotenv import load_dotenv
from pathlib import Path
import tempfile
import numpy as np
import io
import soundfile as sf

# Setup
ROOT_FOLDER = Path(__file__).parent.parent
load_dotenv(ROOT_FOLDER / ".env")


@st.cache_resource
def load_models():
    """Load diarization and transcription models"""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        st.error(
            "HF_TOKEN not found in environment variables. Please add it to your .env file."
        )
        st.stop()

    device = "cpu"
    torch.set_num_threads(4)

    # Load models
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
    )
    transcription_model = whisper.load_model("base", device=device)

    return diarization_pipeline, transcription_model


@st.cache_data
def process_audio(_diarization_pipeline, _transcription_model, audio_path):
    """Process audio file for diarization and transcription - cached to prevent reprocessing"""

    # Perform diarization
    with st.spinner("Performing speaker diarization..."):
        diarization = _diarization_pipeline(audio_path)

    # Load audio for transcription
    with st.spinner("Loading audio for transcription..."):
        y, sr = librosa.load(audio_path, sr=16000)

    # Process each segment
    segments = []
    total_segments = len(list(diarization.itertracks()))
    progress_bar = st.progress(0)

    with st.spinner("Transcribing segments..."):
        for i, (turn, _, speaker) in enumerate(
            diarization.itertracks(yield_label=True)
        ):
            start_time = turn.start
            end_time = turn.end

            # Extract audio segment
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment = y[start_sample:end_sample]

            # Transcribe segment
            result = _transcription_model.transcribe(segment, fp16=False)

            # Only add segments with meaningful text content
            transcribed_text = result["text"].strip()
            if transcribed_text:  # Only append if text is not empty
                segments.append(
                    {
                        "start": start_time,
                        "end": end_time,
                        "speaker": speaker,
                        "text": transcribed_text,
                    }
                )

            # Update progress
            progress_bar.progress((i + 1) / total_segments)

    progress_bar.empty()
    return segments, y.tolist(), sr  # Convert numpy array to list for caching


def create_audio_bytes(audio_data, sample_rate):
    """Create properly formatted audio bytes for streamlit"""
    # Convert list back to numpy array if needed
    if isinstance(audio_data, list):
        audio_data = np.array(audio_data)

    # Create a BytesIO buffer
    buffer = io.BytesIO()

    # Write audio data to buffer as WAV
    sf.write(buffer, audio_data, sample_rate, format="WAV")

    # Get bytes from buffer
    buffer.seek(0)
    return buffer.getvalue()


def display_transcription_with_audio(segments, audio_data, sample_rate):
    """Display transcription with audio player and simple text view"""

    # Create audio player
    st.subheader("🎵 Audio Player")
    audio_bytes = create_audio_bytes(audio_data, sample_rate)
    st.audio(audio_bytes, format="audio/wav")

    # Display simple transcription
    st.subheader("📝 Transcription")

    # Speaker colors
    speaker_colors = {
        "SPEAKER_00": "#FF6B6B",  # Red
        "SPEAKER_01": "#4ECDC4",  # Teal
        "SPEAKER_02": "#45B7D1",  # Blue
        "SPEAKER_03": "#96CEB4",  # Green
        "SPEAKER_04": "#FFEAA7",  # Yellow
        "SPEAKER_05": "#DDA0DD",  # Plum
        "SPEAKER_06": "#FFB347",  # Peach
        "SPEAKER_07": "#98FB98",  # Pale Green
    }

    # Display each segment with speaker colors
    for segment in segments:
        speaker_color = speaker_colors.get(segment["speaker"], "#DDD")

        st.markdown(
            f"""
        <div style="
            background-color: {speaker_color}22; 
            padding: 15px; 
            margin: 10px 0; 
            border-radius: 8px; 
            border-left: 5px solid {speaker_color};
        ">
            <div style="color: {speaker_color}; font-weight: bold; margin-bottom: 8px;">
                {segment['speaker']} ({segment['start']:.1f}s - {segment['end']:.1f}s)
            </div>
            <div style="color: #333; font-size: 1.1em; line-height: 1.4;">
                {segment['text']}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )


def main():
    st.set_page_config(
        page_title="Speaker Diarization & Transcription", page_icon="🎙️", layout="wide"
    )

    st.title("🎙️ Speaker Diarization & Transcription")
    st.markdown(
        "Upload a WAV audio file to get speaker-diarized transcription with interactive timestamp highlighting!"
    )

    # Initialize session state
    if "processed_segments" not in st.session_state:
        st.session_state.processed_segments = None
    if "processed_audio_data" not in st.session_state:
        st.session_state.processed_audio_data = None
    if "processed_sample_rate" not in st.session_state:
        st.session_state.processed_sample_rate = None
    if "processed_file_name" not in st.session_state:
        st.session_state.processed_file_name = None

    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        st.info("Make sure you have set your HF_TOKEN in the .env file")

        # Model loading status
        try:
            diarization_pipeline, transcription_model = load_models()
            st.success("✅ Models loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            st.stop()

    # File upload
    uploaded_file = st.file_uploader(
        "Choose a WAV audio file",
        type=["wav"],
        help="Upload a WAV file for speaker diarization and transcription",
    )

    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")

        # Show process button
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            process_button = st.button(
                "🚀 Process Audio",
                type="primary",
                help="Click to start diarization and transcription",
            )

        with col2:
            if st.session_state.processed_segments is not None:
                if st.button("🗑️ Clear Results"):
                    st.session_state.processed_segments = None
                    st.session_state.processed_audio_data = None
                    st.session_state.processed_sample_rate = None
                    st.session_state.processed_file_name = None
                    st.rerun()

        # Process audio when button is clicked
        if process_button:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                # Process the audio
                segments, audio_data, sample_rate = process_audio(
                    diarization_pipeline, transcription_model, tmp_path
                )

                # Store in session state
                st.session_state.processed_segments = segments
                st.session_state.processed_audio_data = audio_data
                st.session_state.processed_sample_rate = sample_rate
                st.session_state.processed_file_name = uploaded_file.name

            except Exception as e:
                st.error(f"❌ Error processing audio: {str(e)}")

            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        # Display results if they exist in session state
        if st.session_state.processed_segments is not None:
            segments = st.session_state.processed_segments
            audio_data = st.session_state.processed_audio_data
            sample_rate = st.session_state.processed_sample_rate

            if segments:
                st.success(
                    f"🎉 Processing complete! Found {len(segments)} segments from {len(set(seg['speaker'] for seg in segments))} speakers."
                )

                # Display results with interactive features
                display_transcription_with_audio(segments, audio_data, sample_rate)

                # Download button for transcription
                transcription_text = "\n".join(
                    [
                        f"[{seg['start']:.2f}s --> {seg['end']:.2f}s] {seg['speaker']}: {seg['text']}"
                        for seg in segments
                    ]
                )

                st.download_button(
                    label="📥 Download Transcription",
                    data=transcription_text,
                    file_name=f"transcription_{st.session_state.processed_file_name}.txt",
                    mime="text/plain",
                )
            else:
                st.warning("No segments found in the audio file.")

        elif uploaded_file is not None:
            st.info(
                "👆 Click the **Process Audio** button above to start diarization and transcription!"
            )

    else:
        st.info("👆 Please upload a WAV audio file to get started!")

        # Show example
        st.markdown("""
        ### 🔍 What this app does:
        1. **Upload**: Choose a WAV audio file
        2. **Process**: Click the process button to start analysis
        3. **Diarize**: Identify different speakers in the conversation
        4. **Transcribe**: Convert speech to text for each speaker
        5. **Interact**: Navigate through timestamps with highlighting
        6. **Analyze**: View speaker statistics and grouped transcriptions
        
        ### 🎮 Navigation Features:
        - **Slider**: Drag to navigate through the timeline
        - **Quick buttons**: Jump to start/end or skip ±10 seconds
        - **Click segments**: Click any transcript segment to jump to that time
        """)


if __name__ == "__main__":
    main()
