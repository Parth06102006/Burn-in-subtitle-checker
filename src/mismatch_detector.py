from rapidfuzz import fuzz
import unicodedata
import re
import json
import pathlib
import logging

logger = logging.getLogger(__name__)

def normalise(text):
    logger.info("Normalizing Texts")
    text = unicodedata.normalize("NFC", text)
    text = text.strip().lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def classify(score, threshold_ok=0.8, threshold_review=0.6):
    if score >= threshold_ok:
        return "OK"
    elif score >= threshold_review:
        return "REVIEW"
    else:
        return "MISMATCH"
    

def compare(transcription_file,ocr_result_file):
    try:
        logger.info("Started Comparing Results")
        with open(transcription_file,encoding="utf-8") as json_file:
            transcription = json.load(json_file)

        with open(ocr_result_file,encoding="utf-8") as json_file:
            ocr_results = json.load(json_file)

        results = []

        for i, (seg, ocr) in enumerate(zip(transcription, ocr_results)):
            audio_text = seg["text"].strip()
            ocr_text   = ocr["ocr_text"].strip()

            if len(ocr_text) < 3:
                results.append({
                    "index":      i,
                    "start":      seg["start"],
                    "end":        seg["end"],
                    "timestamp":  ocr["timestamp"],
                    "audio_text": audio_text,
                    "ocr_text":   ocr_text,
                    "score":      None,
                    "status":     "OCR_FAILED"
                })
                continue

            score = fuzz.token_sort_ratio(
                normalise(audio_text),
                normalise(ocr_text)
            ) / 100.0

            results.append({
                "index":      i,
                "start":      seg["start"],
                "end":        seg["end"],
                "timestamp":  ocr["timestamp"],
                "audio_text": audio_text,
                "ocr_text":   ocr_text,
                "score":      round(score, 3),
                "status":     classify(score)
            })

        file_path = f"{pathlib.Path(transcription_file).parent}/mismatch_report.json"
        with open(file_path,"w",encoding="utf-8") as json_file:
            json.dump(results,json_file,ensure_ascii=False,indent=4)

        logger.info(f"Completed the mismatch report . Saved at {file_path}")
        return True
    except Exception as e:
        logger.error("Unable to analyze the texts")
        raise