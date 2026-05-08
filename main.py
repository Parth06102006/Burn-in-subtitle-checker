import argparse
import logging
import pathlib

from src.mismatch_detector import compare
from src.report_generator import generate_html_report
from src.subtitle_extractor import extract_subtitles
from src.transcriber import extract_audio, transcribe_audio


logging.basicConfig(
    filename="debug.log",
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def parse_args():
    parser = argparse.ArgumentParser(description="Check burned-in subtitles against audio transcripts.")
    parser.add_argument(
        "--input-dir",
        default="input",
        help="Folder containing video files to process."
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Folder where audio files and per-video reports will be saved."
    )
    return parser.parse_args()


def get_video_files(input_dir):
    return sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def process_video(video_file, output_dir):
    video_output_dir = output_dir / video_file.stem
    audio_output_dir = output_dir / "audio"
    video_output_dir.mkdir(parents=True, exist_ok=True)
    audio_output_dir.mkdir(parents=True, exist_ok=True)

    audio_output = audio_output_dir / f"{video_file.stem}.wav"

    print(f"Extracting audio from {video_file} to {audio_output}...")
    success = extract_audio(str(video_file), str(audio_output))

    if success is not True:
        print(f"Failed to extract audio for {video_file.name}. Skipping.")
        return None

    print(f"Audio extraction successful. Starting transcription on {audio_output}...")
    transcript_file = transcribe_audio(str(audio_output), output_dir=video_output_dir)
    if not transcript_file or isinstance(transcript_file, Exception):
        print(f"Failed to transcribe {audio_output}. Skipping.")
        return None

    print(f"Transcription saved at {transcript_file}")
    print(f"Starting subtitle extraction using {transcript_file}...")
    ocr_file = extract_subtitles(str(video_file), transcript_file, output_dir=video_output_dir)
    print(f"Subtitle extraction successful. OCR results at {ocr_file}")

    print("Starting mismatch detection...")
    mismatch_report = compare(transcript_file, ocr_file, output_file=video_output_dir / "mismatch_report.json")
    print(f"Mismatch report saved at {mismatch_report}")

    html_report = generate_html_report(mismatch_report)
    print(f"HTML report saved at {html_report}")

    return {
        "video": str(video_file),
        "audio": str(audio_output),
        "transcript": transcript_file,
        "ocr": ocr_file,
        "mismatch_report": mismatch_report,
        "html_report": html_report,
    }


def main():
    args = parse_args()
    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    videos = get_video_files(input_dir)

    if not videos:
        print(f"No video files found in {input_dir}")
        return

    results = []
    for video_file in videos:
        print(f"\nProcessing {video_file.name}...")
        try:
            result = process_video(video_file, output_dir)
            if result:
                results.append(result)
        except Exception as e:
            logging.exception(f"Failed to process {video_file}")
            print(f"Failed to process {video_file.name}: {e}")

    print(f"\nCompleted {len(results)} of {len(videos)} video file(s).")


if __name__ == "__main__":
    main()
