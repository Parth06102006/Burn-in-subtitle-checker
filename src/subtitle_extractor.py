import cv2
import logging
import numpy as np
import pathlib
import pytesseract
from dotenv import load_dotenv
import os
import json
import unicodedata

env_path = pathlib.Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger(__name__)


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


def get_border_ratio(mask):
    h, w = mask.shape[:2]
    border_size = max(1, min(h, w) // 20)
    border_pixels = np.concatenate([
        mask[:border_size, :].ravel(),
        mask[-border_size:, :].ravel(),
        mask[:, :border_size].ravel(),
        mask[:, -border_size:].ravel()
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


def preprocess_for_ocr(img, debug_path=None):
    w = img.shape[1]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    left, right = remove_uniform_columns(gray, std_threshold=8)
    margin = 5
    left  = max(0, left - margin)
    right = min(w - 1, right + margin)

    gray = gray[:, left:right + 1]

    row_stds = np.array([gray[y, :].std() for y in range(gray.shape[0])])
    text_rows = np.where(row_stds > 15)[0]

    if len(text_rows) > 0:
        text_top    = max(0, text_rows[0] - 3)
        text_bottom = min(gray.shape[0] - 1, text_rows[-1] + 3)
        gray = gray[text_top:text_bottom + 1, :]

    binary = make_black_text_image(gray)

    current_h = binary.shape[0]
    target_height = 80
    if current_h < target_height:
        scale = target_height / current_h
        binary = cv2.resize(
            binary, None,
            fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
    white_pixel_ratio = (binary == 255).sum() / binary.size
    if white_pixel_ratio < 0.5:
        binary = cv2.bitwise_not(binary)

    if debug_path:
        cv2.imwrite(debug_path, binary)

    return binary


def extract_subtitles(video_file, transcript_file, output_dir=None):
    with open(transcript_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    tess_path = os.environ.get("TESSERACT_FILE_PATH")
    if tess_path and pathlib.Path(tess_path).exists():
        pytesseract.pytesseract.tesseract_cmd = tess_path

    data_ocr = []
    file_path = None

    video_name = pathlib.Path(video_file).stem
    folder_path = (pathlib.Path(output_dir)
                   if output_dir
                   else pathlib.Path(f"cache/tests/{video_name}"))
    folder_path.mkdir(parents=True, exist_ok=True)

    vid = cv2.VideoCapture(video_file)
    if not vid.isOpened():
        raise Exception(f"Cannot open video file: {video_file}")

    try:
        for i, segment in enumerate(data):
            start     = segment["start"]
            end       = segment["end"]
            mid_point = round(start + (end - start) / 2, 2)

            vid.set(cv2.CAP_PROP_POS_MSEC, mid_point * 1000)
            success, frame = vid.read()

            if not success:
                logger.warning(f"Frame capture failed at {mid_point}s. Skipping.")
                data_ocr.append({
                    "timestamp": mid_point,
                    "ocr_text": ""
                })
                continue

            cv2.imwrite(
                str(folder_path / f"original_{mid_point}.jpg"), frame)

            strip = get_subtitle_strip(frame)
            cv2.imwrite(
                str(folder_path / f"cropped_{mid_point}.jpg"), strip)

            processed = preprocess_for_ocr(
                strip,
                debug_path=str(folder_path / f"debug_{mid_point}.jpg")
            )

            logger.info(f"Preprocessed image saved for {mid_point}s")

            custom_config = '--oem 1 --psm 6'
            raw_text = pytesseract.image_to_string(
                processed,
                lang='hin',
                config=custom_config
            )

            clean_text = unicodedata.normalize("NFC", raw_text).strip()
            ocr_text   = clean_text if len(clean_text) >= 3 else ""

            data_ocr.append({
                "timestamp": mid_point,
                "ocr_text":  ocr_text
            })

            logger.info(f"OCR done for segment {i} at {mid_point}s: '{ocr_text[:30]}'")

        file_path = folder_path / "ocr_output.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data_ocr, f, ensure_ascii=False, indent=4)

        logger.info(f"OCR results saved to {file_path}")
        return str(file_path)

    except Exception as e:
        logger.error(f"Failed to extract subtitles: {e}")
        raise

    finally:
        vid.release()
        cv2.destroyAllWindows()
