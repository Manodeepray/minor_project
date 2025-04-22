from .core import llm_integration


AUDIO_DIR = "./data/raw/recorded_audio"
TRANSCRIPT_DIR = "./data/raw/transcription"

output_transcript_path , transcription = llm_integration.transcribe_audio(TRANSCRIPT_DIR = TRANSCRIPT_DIR ,
                                          AUDIO_FILENAME=audio_path)
