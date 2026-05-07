import cv2
import logging
import numpy as np
import pathlib
import pytesseract
from dotenv import load_dotenv
import os
import json
import unicodedata
from rapidfuzz import fuzz
import easyocr

env_path = pathlib.Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger(__name__)

WHISPER_TO_TESSERACT = {
    "hi": "hin",
    "en": "eng",
    "kn": "kan",
}

# ── Padding removal ────────────────────────────────────────────

def find_real_bottom(frame, std_threshold=8):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h = gray.shape[0]
    for y in range(h - 1, -1, -1):
        if gray[y, :].std() > std_threshold:
            return y
    return h - 1


def get_subtitle_strip(frame, std_threshold=8):
    h, w = frame.shape[:2]
    subtitle_fraction = 0.20 if h > w else 0.15
    real_bottom    = find_real_bottom(frame, std_threshold)
    content_height = real_bottom + 1
    crop_top       = int(real_bottom - (content_height * subtitle_fraction))
    crop_top       = max(0, crop_top)
    return frame[crop_top:real_bottom + 1, :]


def remove_uniform_columns(gray, std_threshold=8):
    w = gray.shape[1]
    left = 0
    for x in range(w):
        if gray[:, x].std() > std_threshold:
            left = x
            break
    right = w - 1
    for x in range(w - 1, -1, -1):
        if gray[:, x].std() > std_threshold:
            right = x
            break
    return left, right


# Forced Conversion to Black Text and White Background

def get_border_ratio(mask):
    border_pixels = np.concatenate([
        mask[0, :],
        mask[-1, :],
        mask[:, 0],
        mask[:, -1],
    ])
    return (border_pixels == 255).sum() / border_pixels.size


def make_black_text_image(gray):
    bright_mask = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=-4
    )

    dark_mask = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=4
    )

    masks = [bright_mask, dark_mask]
    scores = []
    for mask in masks:
        foreground_ratio = (mask == 255).sum() / mask.size
        border_ratio = get_border_ratio(mask)
        score = abs(foreground_ratio - 0.12) + border_ratio
        if foreground_ratio < 0.005 or foreground_ratio > 0.60:
            score += 1
        scores.append(score)

    text_mask = masks[int(np.argmin(scores))]

    binary = np.full(gray.shape, 255, dtype=np.uint8)
    binary[text_mask == 255] = 0

    return binary


# ── Preprocessing ──────────────────────────────────────────────

def preprocess_for_ocr(img):
    """
    Correct order: resize grayscale FIRST, threshold AFTER.
    This preserves edge quality for Devanagari strokes.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Remove horizontal padding columns
    left, right = remove_uniform_columns(gray, std_threshold=8)
    margin = 5
    left  = max(0, left - margin)
    right = min(w - 1, right + margin)
    gray  = gray[:, left:right + 1]

    # Mask noisy corner overlays such as timestamps or watermarks.
    ch, cw = gray.shape[:2]
    corner_h = max(1, int(ch * 0.35))
    corner_w = max(1, int(cw * 0.12))
    gray[:corner_h, :corner_w] = 200
    gray[:corner_h, -corner_w:] = 200

    # Find rows with actual text content
    row_stds  = np.array([gray[y, :].std() for y in range(gray.shape[0])])
    text_rows = np.where(row_stds > 12)[0]
    if len(text_rows) > 0:
        text_top    = max(0, text_rows[0] - 3)
        text_bottom = min(gray.shape[0] - 1, text_rows[-1] + 3)
        gray = gray[text_top:text_bottom + 1, :]

    # ── CORRECT ORDER: resize grayscale first ──
    current_h   = gray.shape[0]
    target_h    = 80
    if current_h < target_h:
        scale = target_h / current_h
        gray  = cv2.resize(
            gray, None,
            fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC   # smooth on grayscale
        )

    mean_brightness = gray.mean()

    if mean_brightness > 180:
        # Light subtitle bars with white outlined text need inversion first,
        # otherwise OCR reads the outline instead of the filled letter shape.
        inverted = cv2.bitwise_not(gray)
        binary = cv2.adaptiveThreshold(
            inverted, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=8
        )
    else:
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=10
        )

    # Reconnect broken Devanagari shirorekha and other horizontal strokes.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Tesseract performs best here with black text on a white background.
    white_ratio = (binary == 255).sum() / binary.size
    if white_ratio < 0.5:
        binary = cv2.bitwise_not(binary)

    return binary

# ── OCR ────────────────────────────────────────────────────────

def run_tesseract(image, lang):
    h, _ = image.shape[:2]
    psm = 7 if h < 60 else 6
    config  = f'--oem 1 --psm {psm}'
    raw     = pytesseract.image_to_string(image, lang=lang, config=config)
    cleaned = unicodedata.normalize("NFC", raw).strip()
    return cleaned


def is_valid_indic_text(text, lang_code):
    """
    Reject OCR that is mostly punctuation, symbols, or the wrong script.
    This catches noisy frames while keeping real Hindi/Kannada/English text.
    """
    if len(text) < 3:
        return False

    ranges = {
        "hin": (0x0900, 0x097F),
        "kan": (0x0C80, 0x0CFF),
        "eng": (0x0041, 0x007A),
    }

    start_r, end_r = ranges.get(lang_code, (0x0900, 0x097F))
    expected_chars = sum(1 for c in text if start_r <= ord(c) <= end_r)
    total_alpha = sum(1 for c in text if c.isalpha())

    if total_alpha == 0:
        return False

    return (expected_chars / total_alpha) >= 0.40


def run_easyocr(image, lang_code):
    """EasyOCR fallback — only called when Tesseract returns nothing."""
    try:
        # EasyOCR uses different language codes
        easy_lang_map = {"hin": "hi", "eng": "en", "kan": "kn"}
        easy_lang     = easy_lang_map.get(lang_code, "en")
        reader        = easyocr.Reader([easy_lang], gpu=False)
        results       = reader.readtext(image, detail=0)
        text          = " ".join(results).strip()
        return unicodedata.normalize("NFC", text)
    except Exception as e:
        logger.warning(f"EasyOCR fallback failed: {e}")
        return ""


def run_ocr(image, lang):
    """
    Try Tesseract first.
    Fall back to EasyOCR if Tesseract returns garbage or nothing.
    """
    text = run_tesseract(image, lang)
    if len(text) >= 3 and is_valid_indic_text(text, lang):
        return text, "tesseract"

    logger.info("Tesseract output invalid or empty - trying EasyOCR fallback")
    text = run_easyocr(image, lang)
    if len(text) >= 3 and is_valid_indic_text(text, lang):
        return text, "easyocr"

    return "", "failed"

# ── Single frame processor ─────────────────────────────────────

def process_single_frame(vid, timestamp_s, lang,
                          folder_path, label):
    """
    Seek to timestamp_s, extract frame, preprocess, run OCR.
    Returns OCR text string. label is used for debug filenames.
    """
    vid.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000)
    success, frame = vid.read()

    if not success:
        logger.warning(f"Frame capture failed at {timestamp_s}s")
        return ""

    # Save original for debugging
    cv2.imwrite(
        str(folder_path / f"original_{label}.jpg"), frame)

    strip = get_subtitle_strip(frame)
    cv2.imwrite(
        str(folder_path / f"cropped_{label}.jpg"), strip)

    processed = preprocess_for_ocr(strip)
    cv2.imwrite(
        str(folder_path / f"debug_{label}.jpg"), processed)

    text, engine = run_ocr(processed, lang)
    logger.info(f"  [{label}] engine={engine} text='{text[:40]}'")
    return text

# ── Consolidation ──────────────────────────────────────────────

def consolidate_ocr_results(results, similarity_threshold=70):
    """
    Given OCR text from start/mid/end frames,
    return the best text plus a consolidation status.

    Strategy:
    - If subtitles are stable across the segment, keep the longest result
    - If subtitles changed, join the unique start/mid/end subtitle sequence
    """
    start_text, mid_text, end_text = results
    valid = [r for r in results if len(r) >= 3]

    if not valid:
        return "", "ocr_failed"

    if len(valid) == 1:
        return valid[0], "single_frame"

    has_start = len(start_text) >= 3
    has_mid   = len(mid_text)   >= 3
    has_end   = len(end_text)   >= 3

    if has_start and has_end:
        start_end_sim = fuzz.token_sort_ratio(start_text, end_text)
        subtitle_changed = start_end_sim < similarity_threshold
    else:
        subtitle_changed = False

    if not subtitle_changed:
        best = max(valid, key=len)
        logger.info(f"  Consolidated stable: '{best[:40]}' "
                    f"(from {len(valid)}/3 valid frames)")
        return best, "stable"

    unique_parts = []

    def add_if_new(text):
        if len(text) < 3:
            return
        if not unique_parts:
            unique_parts.append(text)
            return

        sim = fuzz.token_sort_ratio(text, unique_parts[-1])
        if sim < similarity_threshold:
            unique_parts.append(text)

    add_if_new(start_text)
    add_if_new(mid_text)
    add_if_new(end_text)

    combined = " ".join(unique_parts)
    logger.info(f"  Consolidated changed subtitle: '{combined[:40]}' "
                f"(from {len(unique_parts)} unique parts)")
    return combined, "subtitle_changed"

# ── Main extractor ─────────────────────────────────────────────

def extract_subtitles(video_file, transcript_file, output_dir=None):
    with open(transcript_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    tess_path = os.environ.get("TESSERACT_FILE_PATH")
    if tess_path and pathlib.Path(tess_path).exists():
        pytesseract.pytesseract.tesseract_cmd = tess_path

    # Language mapping from Whisper code to Tesseract code
    whisper_lang = data.get("language_detected", "hi")
    tess_lang    = WHISPER_TO_TESSERACT.get(whisper_lang, "hin")
    logger.info(f"Using OCR language: {tess_lang} (from Whisper: {whisper_lang})")

    data_ocr  = []
    video_name = pathlib.Path(video_file).stem
    folder_path = (pathlib.Path(output_dir)
                   if output_dir
                   else pathlib.Path(f"cache/tests/{video_name}"))
    folder_path.mkdir(parents=True, exist_ok=True)

    vid = cv2.VideoCapture(video_file)
    if not vid.isOpened():
        raise Exception(f"Cannot open video file: {video_file}")

    try:
        for i, segment in enumerate(data["transcription"]):
            start     = segment["start"]
            end       = segment["end"]
            duration  = end - start

            # Three sample points within the segment
            # Avoid the very first and last frame — subtitle
            # may still be fading in/out at exact boundaries
            t_start = start + duration * 0.15
            t_mid   = start + duration * 0.50
            t_end   = start + duration * 0.85

            logger.info(f"Segment {i}: {start:.2f}s–{end:.2f}s")

            text_start = process_single_frame(
                vid, t_start, tess_lang, folder_path,
                label=f"seg{i}_start")
            text_mid   = process_single_frame(
                vid, t_mid,   tess_lang, folder_path,
                label=f"seg{i}_mid")
            text_end   = process_single_frame(
                vid, t_end,   tess_lang, folder_path,
                label=f"seg{i}_end")

            final_text, consolidation_status = consolidate_ocr_results(
                [text_start, text_mid, text_end]
            )

            data_ocr.append({
                "timestamp": round(t_mid, 2),   # use mid as reference
                "ocr_text":  final_text,
                "consolidation_status": consolidation_status,
                "frames_used": {
                    "start": round(t_start, 2),
                    "mid":   round(t_mid,   2),
                    "end":   round(t_end,   2),
                }
            })

        file_path = folder_path / "ocr_output.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data_ocr, f, ensure_ascii=False, indent=4)

        logger.info(f"OCR results saved → {file_path}")
        return str(file_path)

    except Exception as e:
        logger.error(f"Failed to extract subtitles: {e}")
        raise

    finally:
        vid.release()
        cv2.destroyAllWindows()
