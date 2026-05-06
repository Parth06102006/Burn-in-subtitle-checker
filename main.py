import logging
from src.transcriber import extract_audio, transcribe_audio
from src.subtitle_extractor import extract_subtitles

logging.basicConfig(
    filename='debug.log', 
    encoding='utf-8', 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    input_file = "test.mp4"
    audio_output = "test_audio.wav"
    
    print(f"Extracting audio from {input_file} to {audio_output}...")
    success = extract_audio(input_file, audio_output)
    
    if success is True:
        print(f"Audio extraction successful. Starting transcription on {audio_output}...")
        transcribe_audio(audio_output)
        print("Transcription successful")
        
        transcript_file = f"cache/transcripts/{audio_output}.json"
        print(f"Starting subtitle extraction using {transcript_file}...")
        ocr_file = extract_subtitles(input_file, transcript_file)
        print(f"Subtitle extraction successful. OCR results at {ocr_file}")
        
        print("Starting mismatch detection...")
        from src.mismatch_detector import compare
        compare(transcript_file, ocr_file)
        print("Mismatch detection completed successfully. Check the mismatch_report.json file.")
    else:
        print("Failed to extract audio. Cannot proceed with transcription.")

if __name__ == "__main__":
    main()
