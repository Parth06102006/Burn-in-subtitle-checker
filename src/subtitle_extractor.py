import cv2
import logging
import numpy as np
import pathlib
import pytesseract
from dotenv import load_dotenv
import os
import json
import unicodedata
import re
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


## Threshold here is set to 8 to provide clear distinction between the background and the text to avoid the solid borders below the subtitle
def find_real_bottom(frame, std_threshold=8):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h = gray.shape[0]
    for y in range(h - 1, -1, -1):
        if gray[y, :].std() > std_threshold:
            return y ## Returning the bottom where the subtitle starts
    return h - 1 ## Returning the h-1 if unable to detect the subtitle bottom 


## Crops the subtitle from top to bottom
def get_subtitle_strip(frame, std_threshold=8):
    h, w = frame.shape[:2]
    subtitle_fraction = 0.20 if h > w else 0.15 ## if the video orientation is in portrait then we use 0.20 of the bottom region otherwise 0.15 of the landscape 
    real_bottom    = find_real_bottom(frame, std_threshold) ## Fetch the bottom of the subtitle
    content_height = real_bottom + 1 ## Fetched the Row 0 value of the subtitle
    crop_top       = int(real_bottom - (content_height * subtitle_fraction)) ## Fetches the rwo 0 value of the subtitle region
    crop_top       = max(0, crop_top)
    return frame[crop_top:real_bottom + 1, :] ## returns selected row and columns


def remove_uniform_columns(gray, std_threshold=8):

    ## We check if the video orientaton is different and if there is the padding in left region as well as right region so we remove it
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


def fix_outlined_text(gray):
    """
    Convert white text with a black outline on a light background into
    solid black text on a white background for OCR.
    """
    _, outline = cv2.threshold(
        gray, 80, 255, cv2.THRESH_BINARY_INV
    )

    # The subtitle outline is typically 2-4 px wide after resizing; this
    # closes the outline enough to recover the filled letter body.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5, 5)
    )
    filled = cv2.morphologyEx(
        outline, cv2.MORPH_CLOSE, kernel
    )

    return cv2.bitwise_not(filled)


def detect_text_style(gray):
    """
    Classify subtitle contrast so preprocessing can choose the safest path.
    """

    ## Used to check if the detected text is having black text on white background for the processing using OCR to be more precise
    mean = gray.mean()

    if mean < 80:
        return "light_on_dark"

    if mean > 170:
        mid_gray_pixels = np.sum((gray > 60) & (gray < 180))
        mid_gray_ratio = mid_gray_pixels / gray.size

        if mid_gray_ratio < 0.08:
            return "outlined_white"

    return "dark_on_light"


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
    ## We have taken the margin to be sure we have no accidently remove some text
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

    ## Detection of the style of the OCR to check for the black text on white background after grayscaling
    style = detect_text_style(gray)
    logger.debug(f"Text style detected: {style}")

    if style == "outlined_white":
        ## Since the text is with black border we convert it to have the white background and the black text
        binary = fix_outlined_text(gray)

    elif style == "light_on_dark":
        # Invert so light subtitle text becomes dark before thresholding.
        inverted = cv2.bitwise_not(gray)
        binary = cv2.adaptiveThreshold(
            inverted, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=10
        )

    else:
        # Standard dark text on a light background.
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=10
        )

    # Reconnect broken Devanagari shirorekha and other horizontal strokes.
    # The outlined path already performs a stronger close while reconstructing.
    if style != "outlined_white":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Tesseract performs best here with black text on a white background. Hence we check one last time that there is white background major portion or not if not we revert it
    white_ratio = (binary == 255).sum() / binary.size
    if white_ratio < 0.5:
        binary = cv2.bitwise_not(binary)

    return binary

# ── OCR ────────────────────────────────────────────────────────

def remove_repetitions(text):
    """
    Remove common OCR hallucinations from noisy subtitle regions.
    """
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = re.sub(r"(.{2})\1{2,}", r"\1", text)

    clean_lines = []
    for line in text.split("\n"):
        line = line.strip()
        if len(line) < 3:
            continue

        most_common_count = max(line.count(c) for c in set(line))
        if most_common_count / len(line) < 0.5:
            clean_lines.append(line)

    return "\n".join(clean_lines).strip()


def get_expected_script_chars(text, lang_code):
    ranges = {
        "hin": (0x0900, 0x097F),
        "kan": (0x0C80, 0x0CFF),
        "eng": (0x0041, 0x007A),
    }
    start_r, end_r = ranges.get(lang_code, (0x0900, 0x097F))
    return sum(1 for c in text if start_r <= ord(c) <= end_r)


def get_ocr_quality_score(text, lang_code="hin"):
    """
    Prefer clean subtitle-like text over long OCR noise.
    """
    compact = "".join(c for c in text if not c.isspace())
    if not compact:
        return 0

    script_chars = get_expected_script_chars(text, lang_code)
    alpha_chars = sum(1 for c in text if c.isalpha())
    useful_chars = script_chars if lang_code != "eng" else alpha_chars

    allowed_punctuation = set("।,.!?\"'“”‘’()-॰")
    suspicious_chars = set("_[]{}<>/\\|*=~^`@#$%&+:;«»")
    suspicious_count = sum(1 for c in compact if c in suspicious_chars)
    symbol_count = sum(
        1
        for c in compact
        if not c.isalnum()
        and c not in allowed_punctuation
        and get_expected_script_chars(c, lang_code) == 0
    )

    tokens = [token for token in re.split(r"\s+", text.strip()) if token]
    repeated_token_penalty = 0
    if tokens:
        token_counts = {token: tokens.count(token) for token in set(tokens)}
        repeated_token_penalty = max(
            (
                count
                for token, count in token_counts.items()
                if len(token) <= 3
            ),
            default=0
        )

    repeated_fragment_penalty = 0
    for size in (2, 3):
        fragments = [
            compact[i:i + size]
            for i in range(0, max(0, len(compact) - size + 1), size)
        ]
        if fragments:
            repeated_fragment_penalty = max(
                repeated_fragment_penalty,
                max(fragments.count(fragment) for fragment in set(fragments))
            )

    return (
        useful_chars
        - suspicious_count * 3
        - symbol_count * 2
        - repeated_token_penalty * 4
        - repeated_fragment_penalty
    )



## Checks for the low quality OCR and if the OCR quality is low then returns True
def is_noisy_ocr_text(text, lang_code):
    compact = "".join(c for c in text if not c.isspace())
    if len(compact) < 3:
        return True

    suspicious_chars = set("_[]{}<>/\\|*=~^`@#$%&+:;«»")
    suspicious_count = sum(1 for c in compact if c in suspicious_chars)
    if suspicious_count / len(compact) > 0.18:
        return True

    tokens = [token for token in re.split(r"\s+", text.strip()) if token]
    if len(tokens) >= 4:
        short_token_counts = [
            tokens.count(token)
            for token in set(tokens)
            if len(token) <= 3
        ]
        if short_token_counts and max(short_token_counts) >= 3:
            return True

    return get_ocr_quality_score(text, lang_code) <= 0


## To check if there is stack iof images being processed together which might cause additional characters to appear
def warn_if_suspicious_ocr_shape(image, engine):
    h, w = image.shape[:2]
    aspect = h / w if w else 0
    if aspect > 0.4:
        logger.warning(
            f"Suspicious {engine} OCR image shape {h}x{w} "
            f"(aspect={aspect:.2f}) - may be stacked strips"
        )
    return h, w



## Normalizes the text here so that all the characters are consistent here
def clean_ocr_text(text):
    text = unicodedata.normalize("NFC", text).strip()
    return remove_repetitions(text)


def run_tesseract(image, lang):
    h, _ = warn_if_suspicious_ocr_shape(image, "Tesseract")

    psm = 7 if h < 60 else 6
    config  = f'--oem 1 --psm {psm}'
    raw     = pytesseract.image_to_string(image, lang=lang, config=config)
    return clean_ocr_text(raw)


def is_valid_indic_text(text, lang_code):
    """
    Reject OCR that is mostly punctuation, symbols, or the wrong script.
    This catches noisy frames while keeping real Hindi/Kannada/English text.
    """
    if len(text) < 3:
        return False

    # If the text in the language is correct or not 
    expected_chars = get_expected_script_chars(text, lang_code)

    ## Total chracters found
    total_alpha = sum(1 for c in text if c.isalpha())

    if total_alpha == 0:
        return False

    ## Checks for the probablity of the characters are valid or not
    return (expected_chars / total_alpha) >= 0.40


def is_acceptable_ocr_text(text, lang_code):
    if len(text) < 3:
        return False
    if not is_valid_indic_text(text, lang_code):
        return False
    if is_noisy_ocr_text(text, lang_code):
        logger.info(f"Rejected noisy OCR text: '{text[:60]}'")
        return False
    return True


def run_easyocr(image, lang_code):
    """EasyOCR fallback — only called when Tesseract returns nothing."""
    try:
        warn_if_suspicious_ocr_shape(image, "EasyOCR")

        # EasyOCR uses different language codes
        easy_lang_map = {"hin": "hi", "eng": "en", "kan": "kn"}
        easy_lang     = easy_lang_map.get(lang_code, "en")
        reader        = easyocr.Reader([easy_lang], gpu=False)
        results       = reader.readtext(image, detail=0)
        text          = " ".join(results).strip()
        return clean_ocr_text(text)
    except Exception as e:
        logger.warning(f"EasyOCR fallback failed: {e}")
        return ""


def run_ocr(image, lang):
    """
    Try Tesseract first.
    Fall back to EasyOCR if Tesseract returns garbage or nothing.
    """
    text = run_tesseract(image, lang)
    if is_acceptable_ocr_text(text, lang):
        return text, "tesseract"

    logger.info("Tesseract output invalid or empty - trying EasyOCR fallback")
    text = run_easyocr(image, lang)
    if is_acceptable_ocr_text(text, lang):
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


    ## Calling to get the subtitle rectangular region
    strip = get_subtitle_strip(frame)

    ## Save the cropped image for further processing
    cv2.imwrite(
        str(folder_path / f"cropped_{label}.jpg"), strip)

    ## Calling the function to get the processed strip for the OCR to run and produce text more accurately
    processed = preprocess_for_ocr(strip)

    ## Saved the processed image for the ocr 
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
    - If subtitles are stable across the segment, keep the cleanest result
    - If subtitles changed, join the unique start/mid/end subtitle sequence
    """
    start_text, mid_text, end_text = results
    valid = [
        r for r in results
        if len(r) >= 3 and get_ocr_quality_score(r) > 0
    ]

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
        best = max(valid, key=get_ocr_quality_score)
        logger.info(f"  Consolidated stable: '{best[:40]}' "
                    f"(from {len(valid)}/3 valid frames)")
        return best, "stable"

    unique_parts = []

    def add_if_new(text):
        if len(text) < 3:
            return
        if get_ocr_quality_score(text) <= 0:
            logger.info(f"Skipping noisy consolidation candidate: '{text[:60]}'")
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

    if not unique_parts:
        return "", "ocr_failed"

    combined = " ".join(unique_parts)
    logger.info(f"  Consolidated changed subtitle: '{combined[:40]}' "
                f"(from {len(unique_parts)} unique parts)")
    return combined, "subtitle_changed"

# ── Main extractor ─────────────────────────────────────────────

def extract_subtitles(video_file, transcript_file, output_dir=None):

    ## Transcription File Required to Check for the timestamps
    with open(transcript_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    ## Path Provided for the Tesseract Execution Application and provides the path to the pytesseract
    tess_path = os.environ.get("TESSERACT_FILE_PATH")

    if tess_path and pathlib.Path(tess_path).exists():
        pytesseract.pytesseract.tesseract_cmd = tess_path

    # Language mapping from Whisper code to Tesseract code
    whisper_lang = data.get("language_detected", "hin") ## !! To add multi if the languag is not detected by whisper
    tess_lang    = WHISPER_TO_TESSERACT.get(whisper_lang, "hin")
    logger.info(f"Using OCR language: {tess_lang} (from Whisper: {whisper_lang})")

    data_ocr  = []
    video_name = pathlib.Path(video_file).stem
    folder_path = (pathlib.Path(output_dir)
                   if output_dir ## cahche is here the temporary folder created for the test purpose
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
