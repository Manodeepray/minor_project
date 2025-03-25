import llm_integration


AUDIO_DIR = "./recorded_audio"
TRANSCRIPT_DIR = "./transcription"

output_transcript_path , transcription = llm_integration.transcribe_audio(TRANSCRIPT_DIR = TRANSCRIPT_DIR ,
                                          AUDIO_FILENAME=audio_path)
