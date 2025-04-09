# audio.py
import os
from openai import OpenAI
from pydub import AudioSegment
import io

# Assume OPENAI_API_KEY is set in environment
# client = OpenAI() # Initialize client here or pass it in

def speech_to_text(client: OpenAI, audio_segment: AudioSegment):
    """Uses OpenAI's Whisper to convert audio file to text.
       @param client  OpenAI client
       @audio_segment : pydub.AudioSegment  Audio data to convert.
    """
    print("Audio segment duration:", len(audio_segment), "ms")

    # Whisper API expects a file-like object.
    # Export AudioSegment to an in-memory buffer. Use 'mp3' or other supported format.
    buffer = io.BytesIO()
    try:
        # Exporting requires ffmpeg/libav to be installed if not using wav/raw
        audio_segment.export(buffer, format="mp3")
        buffer.seek(0)
        buffer.name = "audio.mp3" # API needs a filename hint

        print("Sending audio buffer to Whisper...")
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=buffer
        )
        print("Whisper transcription received.")
        return transcript.text
    except Exception as e:
        print(f"Error during audio export or transcription: {e}")
        # Consider specific exception handling for pydub export errors (e.g., ffmpeg not found)
        raise # Re-raise the exception or handle it appropriately
    finally:
        buffer.close()


def text_to_speech(client: OpenAI, ele, text):
    """Uses OpenAI to convert text to speech.
       @param client  OpenAI client
       @param ele  HTML element (Not directly usable in backend - this param seems misplaced for a backend function)
       @param text  Text to convert
    """
    # This function's design seems more suited for client-side JS or a different backend interaction.
    # A backend function would typically return the audio data or a URL to it.
    # Let's implement it to return audio data (e.g., bytes) for now.
    print(f"Generating speech for text: {text[:30]}...")
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        print("Speech generated.")
        # Return audio content as bytes
        return response.content # The raw audio data
    except Exception as e:
        print(f"Error during text-to-speech generation: {e}")
        raise

# Example usage (if run directly, requires OPENAI_API_KEY)
if __name__ == '__main__':
    # This part is for testing audio.py independently.
    # You would need a sample audio file and set your API key.
    # from dotenv import load_dotenv
    # load_dotenv()
    # client = OpenAI()
    #
    # # --- Test STT ---
    # try:
    #     # Create a dummy silent audio segment for testing structure
    #     # Replace with actual audio loading: sample_audio = AudioSegment.from_file("your_audio.wav")
    #     sample_audio = AudioSegment.silent(duration=1000) # 1 second silence
    #     print("Testing speech_to_text...")
    #     text_result = speech_to_text(client, sample_audio)
    #     print("STT Result:", text_result)
    # except Exception as e:
    #     print(f"STT Test Error: {e}")

    # # --- Test TTS ---
    # try:
    #     print("\nTesting text_to_speech...")
    #     test_text = "Hello, this is a test of the text to speech system."
    #     audio_data = text_to_speech(client, None, test_text) # ele=None
    #     if audio_data:
    #         print(f"TTS Result: Received {len(audio_data)} bytes of audio data.")
    #         # Optionally save the audio to a file to verify
    #         # with open("tts_output.mp3", "wb") as f:
    #         #     f.write(audio_data)
    #         # print("Saved TTS output to tts_output.mp3")
    # except Exception as e:
    #     print(f"TTS Test Error: {e}")
    pass # Avoid running example code when imported
