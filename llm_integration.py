import speech_recognition as sr
import wave
import os
from dotenv import load_dotenv
import pyaudio
import wave

import subprocess


load_dotenv()

def record_audio(AUDIO_DIR, record_seconds=5, sample_rate=44100, channels=1):
    # Initialize PyAudio
    
    AUDIO_FILENAME = os.path.join(AUDIO_DIR, "recorded_audio.wav")

    # Ensure directories exist
    os.makedirs(AUDIO_DIR, exist_ok=True)

    audio = pyaudio.PyAudio()

    # Set up the stream
    stream = audio.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=1024)

    print("Recording...")

    frames = []

    # Record audio in chunks
    for _ in range(0, int(sample_rate / 1024 * record_seconds)):
        data = stream.read(1024)
        frames.append(data)

    print("Recording finished. \n")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded data as a .wav file
    with wave.open(AUDIO_FILENAME, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

    print(f"Audio saved as {AUDIO_FILENAME} \n ")
    
    return AUDIO_FILENAME


import os
import wave
import json
from vosk import Model, KaldiRecognizer





def convert_to_PCM(AUDIO_FILENAME):
    try:
        # Create a temporary file name
        temp_filename = AUDIO_FILENAME + "_temp.wav"
        
        command = [
            "ffmpeg", "-y", "-i", AUDIO_FILENAME, "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", temp_filename
        ]
        
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Overwrite the original file
        os.replace(temp_filename, AUDIO_FILENAME)
        
        print(f"✅ Converted {AUDIO_FILENAME} to PCM format.")
        return AUDIO_FILENAME
    
    except Exception as e:
        print(f"❌ Error converting audio to PCM: {e}")
        return None














def transcribe_audio(TRANSCRIPT_DIR, AUDIO_FILENAME, VOSK_MODEL_PATH="model"):
    """
    Transcribes an audio file using Vosk (offline).
    
    Parameters:
        - TRANSCRIPT_DIR (str): Directory to save the transcript.
        - AUDIO_FILENAME (str): Path to the audio file.
        - VOSK_MODEL_PATH (str): Path to the Vosk model directory.

    Returns:
        - (str, str): Path to transcription file and transcribed text.
    """


    AUDIO_FILENAME = convert_to_PCM(AUDIO_FILENAME)

    if not AUDIO_FILENAME:
        print("❌ Failed to convert audio.")
        return "None", "None"




    TRANSCRIPT_FILENAME = os.path.join(TRANSCRIPT_DIR, "transcription.txt")
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

    # Check if the Vosk model exists
    if not os.path.exists(VOSK_MODEL_PATH):
        print("❌ Vosk model not found! Download a model from https://alphacephei.com/vosk/models")
        return "None", "None"

    # Load Vosk model
    model = Model(VOSK_MODEL_PATH)
    recognizer = KaldiRecognizer(model, 16000)  # Vosk model expects 16kHz audio

    # Open the audio file
    with wave.open(AUDIO_FILENAME, "rb") as wf:
        # Ensure the audio format is compatible with Vosk
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
            print("⚠️ Audio format not supported! Convert to 16kHz, 16-bit PCM, mono channel.")
            return "None", "None"

        # Read and process audio in chunks
        print("🔍 Transcribing audio...")
        text = ""
        while True:
            data = wf.readframes(4000)  # Read 4000 frames at a time
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text += result.get("text", "") + " "

        # Final transcription
        final_result = json.loads(recognizer.FinalResult())
        text += final_result.get("text", "")

        if text.strip():
            # Save transcription
            with open(TRANSCRIPT_FILENAME, "w") as f:
                f.write(text)

            print(f"✅ Transcription saved at: {TRANSCRIPT_FILENAME}")
            return TRANSCRIPT_FILENAME, text
        else:
            print("❌ No speech detected in the audio.")
            return "None", "None"









import os
from groq import Groq
def load_llm():
    # Set your Groq API key
    os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY')
    # Initialize the Groq client
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))


    return client

def generate_notes(transcription , client, NOTES_DIR , class_name):
    prompt = f"""
    Please summarize the main points and create structured class notes, including key topics, subpoints, and any important details in an .md format
    
    1.i want to store them in a .md file
    2.if no context write no transcription found
    context : 
    {transcription}
    
    """
    
    response = client.chat.completions.create(
        model="llama3-8b-8192",  # Use the appropriate Groq model
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.7
    )
    
    notes = response.choices[0].message.content
    
    notes_path = os.path.join(NOTES_DIR , f"{class_name}.md")
    print(f"writing notes at {notes_path}")
    
    with open(notes_path, "w" , encoding='utf-8') as file:
        file.write(notes)
    
    return notes , notes_path


# Example transcription input






# Run the full process

if __name__ == "__main__":

    # Directories
    AUDIO_DIR = "./recorded_audio"
    TRANSCRIPT_DIR = "./transcription"
    RECORD_TIME = 10
    NOTES_DIR = "database_notes"
    class_name =  'test'
    # output_audio_path = record_audio(AUDIO_DIR=AUDIO_DIR, record_seconds=RECORD_TIME)

    output_transcript_path , transcription = transcribe_audio(TRANSCRIPT_DIR = TRANSCRIPT_DIR ,
                                            AUDIO_FILENAME="database_audios/recording_20250310-194001.wav" ,
                                            VOSK_MODEL_PATH="models/vosk-model-small-en-us-0.15")



    client = load_llm()

    # Get class notes
    notes = generate_notes(transcription , client ,NOTES_DIR , class_name )
    print("\nGenerated Notes:\n")
    print(notes)

