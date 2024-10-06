# audio_processor.py

import sounddevice as sd
import scipy.io.wavfile as wav
import time
from openai import OpenAI
import json
import os
from datetime import datetime
import numpy as np
from pydub import AudioSegment
from langchain_mistralai import ChatMistralAI

from pydantic import BaseModel, Field
from typing import List
import threading


class ProcessedTranscriptionSegment(BaseModel):
    start_time: float = Field(..., description="The start time of the audio segment.")
    end_time: float = Field(..., description="The end time of the audio segment.")
    text: str = Field(..., description="The transcribed text for the segment.")
    keywords: list[str] = Field([], description="Keywords extracted from the text.")
    sentiment: str = Field("neutral", description="The sentiment of the text (positive, neutral, negative).")


class ModelOutput(BaseModel):
    processed_text: List[ProcessedTranscriptionSegment] = Field(..., description="The processed transcription segments.")


class AudioProcessor:
    def __init__(self):
        # Configure OpenAI API key
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Configure MistralAI API key
        self.llm = ChatMistralAI(
            model="mistral-small-latest",
            temperature=0,
            max_retries=2,
            api_key=os.environ.get("MISTRALAI_API_KEY")
        ).with_structured_output(ModelOutput)
        
        # Configuration Parameters
        self.SAMPLE_RATE = 16000  # Sample rate for recording
        self.CHANNELS = 1  # Mono recording
        self.AUDIO_DIR = "./audio_files"
        self.LOG_FILE = "speech_log.json"
        
        # Ensure audio directory exists
        os.makedirs(self.AUDIO_DIR, exist_ok=True)
        
        # Initialize recording attributes
        self.recording = False
        self.audio_data = []
        self.stream = None
        self.record_thread = None
        self.start_time = None
        self.stop_event = threading.Event()

    def start(self, duration=None):
        """
        Starts the audio recording.
        If duration is specified, records for that duration.
        Otherwise, records until stop() is called.
        """
        if self.recording:
            print("Already recording.")
            return
        
        self.recording = True
        self.audio_data = []
        self.start_time = time.time()
        self.stop_event.clear()
        
        def callback(indata, frames, time_info, status):
            if not self.recording:
                raise sd.CallbackStop()
            self.audio_data.append(indata.copy())
        
        try:
            if duration:
                print(f"Recording audio for {duration} seconds...")
                with sd.InputStream(samplerate=self.SAMPLE_RATE,
                                    channels=self.CHANNELS,
                                    dtype='int16',
                                    callback=callback):
                    sd.sleep(int(duration * 1000))
                self.recording = False
                self.process_recording()
            else:
                print("Recording started. Press Ctrl+C to stop.")
                self.stream = sd.InputStream(samplerate=self.SAMPLE_RATE,
                                             channels=self.CHANNELS,
                                             dtype='int16',
                                             callback=callback)
                self.stream.start()
                # Run in a separate thread to allow main thread to catch KeyboardInterrupt
                self.record_thread = threading.Thread(target=self._keep_recording, daemon=True)
                self.record_thread.start()
        except Exception as e:
            print(f"Error during recording: {e}")
            self.recording = False

    def _keep_recording(self):
        """
        Keeps the recording active until the stop_event is set.
        """
        while not self.stop_event.is_set():
            time.sleep(0.1)

    def stop(self):
        """
        Stops the audio recording and processes the audio.
        """

        print("Stopping recording...")
        if not self.recording:
            print("Not currently recording.")
            return
        
        self.recording = False
        self.stop_event.set()
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if self.record_thread and self.record_thread.is_alive():
            self.record_thread.join()
        
        print("Recording stopped.")
        self.process_recording()

    def process_recording(self):
        """
        Saves the audio, transcribes it, post-processes the transcription, and saves the log.
        """
        print("Processing audio data...")

        if not self.audio_data:
            print("No audio data to process.")
            return

        # Convert recorded data to numpy array
        audio_data_np = np.concatenate(self.audio_data, axis=0)
        
        # Save the recorded audio
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_filename = os.path.join(self.AUDIO_DIR, f"audio_{timestamp}.wav")
        self.save_audio(audio_data_np, wav_filename)
        
        # Convert WAV to MP3
        mp3_filename = wav_filename.replace('.wav', '.mp3')
        self.convert_wav_to_mp3(wav_filename, mp3_filename)
        
        # Transcribe the audio
        transcription = self.transcribe_audio(mp3_filename)
        if transcription is None:
            print("Transcription failed. Exiting...")
            return
        
        # Format the transcript
        formatted_transcript = []
        for segment in transcription.segments:
            formatted_transcript.append({
                "start_time": segment.start,
                "end_time": segment.end,
                "text": segment.text
            })
        
        # Compress the transcription
        formatted_transcript = self.compress_transcription(formatted_transcript)
        
        # Save formatted transcript to a file (optional)
        # formatted_transcript_path = os.path.join(self.AUDIO_DIR, f"transcription_{timestamp}.json")
        # with open(formatted_transcript_path, "w") as f:
        #     json.dump(formatted_transcript, f, indent=4)
        
        # Post-process the transcription using MistralAI
        processed_data = self.post_process_transcription(json.dumps(formatted_transcript, indent=4))
        
        # Print the processed data as JSON
        print("Processed Data:")
        print(json.dumps(processed_data, indent=4))
        
        # Save the log entry
        self.save_log(processed_data.get("processed_text"))

    def save_audio(self, indata, filename):
        """
        Saves the recorded WAV audio data to a WAV file.
        """
        try:
            print(f"Saving audio to {filename}...")
            wav.write(filename, self.SAMPLE_RATE, indata)
            print(f"Audio saved to {filename}")
        except Exception as e:
            print(f"Error saving audio: {e}")

    def convert_wav_to_mp3(self, wav_filename, mp3_filename):
        """
        Converts a WAV file to MP3 format.
        """
        try:
            print(f"Converting {wav_filename} to {mp3_filename}...")
            audio = AudioSegment.from_wav(wav_filename)
            audio.export(mp3_filename, format="mp3")
            os.remove(wav_filename)  # Remove the WAV file after conversion
            print(f"Audio converted to {mp3_filename}")
        except Exception as e:
            print(f"Error converting WAV to MP3: {e}")

    def transcribe_audio(self, file_path):
        """
        Transcribes the audio file using OpenAI's Whisper API.
        """
        try:
            print(f"Transcribing audio from {file_path}...")
            with open(file_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-1",
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )
            print("Transcription complete.")
            return transcript
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None

    def post_process_transcription(self, transcription_text):
        """
        Processes the transcription text using MistralAI to extract keywords and sentiment.
        """
        system_prompt = """
Process a transcription JSON to include additional fields for keywords and sentiment for each text entry.

- For each transcription entry, analyze the text to extract relevant keywords.
- Determine the sentiment of the text (e.g., positive, neutral, negative).
- Add the extracted keywords and sentiment as additional fields within each JSON object maintaining the original format.

# Steps

1. **Extract Keywords**: Identify key terms or phrases from the `text` field.
2. **Determine Sentiment**: Analyze the `text` field to assign a sentiment label.
3. **Augment JSON Object**: Add `keywords` and `sentiment` fields to each transcription entry.

# Output Format

JSON with the following structure:

```json
[
    {
        "start_time": [float],
        "end_time": [float],
        "text": "[string]",
        "keywords": ["[string1]", "[string2]", ...],
        "sentiment": "[positive|neutral|negative]"
    },
    ...
]
```

# Examples

**Input**:

```json
[
    {
        "start_time": 0.0,
        "end_time": 10.0,
        "text": "Welcome everyone to our annual AI conference. It's great to have so many innovative minds gathered here."
    },
    {
        "start_time": 10.0,
        "end_time": 20.0,
        "text": "Today, we'll explore the latest advancements in machine learning and their impact on the industry."
    }
]
```

**Output**:

```json
[
    {
        "start_time": 0.0,
        "end_time": 10.0,
        "text": "Welcome everyone to our annual AI conference. It's great to have so many innovative minds gathered here.",
        "keywords": ["AI conference", "innovative minds"],
        "sentiment": "positive"
    },
    {
        "start_time": 10.0,
        "end_time": 20.0,
        "text": "Today, we'll explore the latest advancements in machine learning and their impact on the industry.",
        "keywords": ["machine learning", "advancements", "industry impact"],
        "sentiment": "neutral"
    }
]
```

# Notes

- Ensure that each transcription element retains its original structure, with the new fields appended.
- Consider edge cases where the text is very short or does not clearly convey sentiment.
- Keywords should be relevant and provide insight into the main topics or subjects of the text.
        """
        try:
            messages = [
                ("system", system_prompt),
                ("human", transcription_text)
            ]
            result = self.llm.invoke(messages)
            return result.dict()
        except Exception as e:
            print(f"Error during post-processing: {e}")
            return {
                "processed_text": transcription_text,
                "keywords": [],
                "sentiment": "neutral"
            }

    def save_log(self, log_data):
        try:
            print("Saving log data...")
            
            with open(self.LOG_FILE, "w") as f:
                json.dump(log_data, f, indent=4)
            print(f"Log data saved to {self.LOG_FILE}")
        except Exception as e:
            print(f"Error saving log data: {e}")


    def compress_transcription(self, transcription):
        """
        Compresses the transcription segments by merging adjacent segments in groups of 5.
        """
        compressed_transcription = []
        for i in range(0, len(transcription), 5):
            segment_group = transcription[i:i + 5]
            start_time = segment_group[0]["start_time"]
            end_time = segment_group[-1]["end_time"]
            text = " ".join([segment["text"] for segment in segment_group])
            compressed_transcription.append({
                "start_time": start_time,
                "end_time": end_time,
                "text": text
            })
        return compressed_transcription


if __name__ == "__main__":
    processor = AudioProcessor()
    
    try:
        duration_input = input("Enter recording duration in seconds (leave blank for manual stop): ")
        if duration_input.strip():
            duration = float(duration_input)
            processor.start(duration=duration)
        else:
            processor.start()
            print("Press Ctrl+C to stop recording.")
            while processor.recording:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected.")
        processor.stop()
    except Exception as e:
        print(f"An error occurred: {e}")
