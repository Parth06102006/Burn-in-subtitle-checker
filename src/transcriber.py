import subprocess
import logging
import whisper
import json
import pathlib

logger = logging.getLogger(__name__)

## Function to Extract Audio from Video
def extract_audio(input_file,output_file):
    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-i", input_file,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_file
    ]

    try:
        subprocess.run(ffmpeg_command,check=True)
        logger.info(f"Audio extracted successfully to {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error extracting audio: {e}")
        return Exception("Audio Extraction failed")

def transcribe_audio(audio_file, output_dir=None):
    try:
        logger.info("Loading Whisper model")
        model = whisper.load_model("turbo")

        result = model.transcribe(audio_file)

        logger.info("Transcription Completed")
        if result:
            folder_path = pathlib.Path(output_dir) if output_dir else pathlib.Path("cache/transcripts")
            folder_path.mkdir(parents=True, exist_ok=True)

            file_path = folder_path / f"{pathlib.Path(audio_file).name}.json"
            
            data_ocr = {"language_detected":result["language"]}
            data = []

            ## Writing the transcript data
            with open(file_path,"w",encoding="utf-8") as json_file:
                for segment in result["segments"]:
                    data.append({
                        "start":round(segment["start"],2),
                        "end":round(segment["end"],2),
                        "text":segment["text"]
                    })

                data_ocr["transcription"] = data
                json.dump(data_ocr, json_file, ensure_ascii=False, indent=4)
                json_file.write("\n")   

            logger.info("Transcription saved successfully")
            return str(file_path)
        return None
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return Exception("Transcription failed")
