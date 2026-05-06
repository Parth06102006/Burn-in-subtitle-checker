import cv2
import logging
import pathlib
import pytesseract
from PIL import Image
from dotenv import load_dotenv
import os
import json

# Load .env from the project root (one level up from this script)
env_path = pathlib.Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger(__name__)


def find_real_bottom(frame, std_threshold=8):
    """
    Detects padding by row uniformity — works for
    dark padding, light padding, or no padding at all.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h = gray.shape[0]

    for y in range(h - 1, -1, -1):
        row_std = gray[y, :].std()
        if row_std > std_threshold:
            return y

    return h - 1


def get_subtitle_strip(frame, subtitle_fraction=0.15,
                       std_threshold=8):
    real_bottom    = find_real_bottom(frame, std_threshold)
    content_height = real_bottom + 1
    crop_top       = int(real_bottom - (content_height * subtitle_fraction))
    crop_top       = max(0, crop_top)

    return frame[crop_top:real_bottom + 1, :]

def extract_subtitles(video_file, transcript_file, output_dir=None):
    data = None
    data_ocr = []
    with open(transcript_file, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    tess_path = os.environ.get("TESSERACT_FILE_PATH")
    if tess_path and pathlib.Path(tess_path).exists():
        pytesseract.pytesseract.tesseract_cmd = tess_path

    vid = None
    try:
        for i in data:
            start = i["start"]
            end = i["end"]
            mid_point = round(start+(end-start)/2,2)

            vid = cv2.VideoCapture(video_file)
            vid.set(cv2.CAP_PROP_POS_MSEC, mid_point * 1000)

            success, frame = vid.read()

            if not success:
                logger.warning(f"Frame capture failed for midpoint {mid_point}. Skipping.")
                continue

            cropped_frame = get_subtitle_strip(frame)

            video_name = pathlib.Path(video_file).name
            folder_path = pathlib.Path(output_dir) if output_dir else pathlib.Path(f"cache/tests/{video_name}")
            folder_path.mkdir(parents=True, exist_ok=True)

            if not cv2.imwrite(str(folder_path / f"original_{mid_point}.jpg"), frame):
                raise Exception("Error saving original frame")

            if not cv2.imwrite(str(folder_path / f"cropped_{mid_point}.jpg"), cropped_frame):
                raise Exception("Error saving cropped frame")

            logger.info("Cropped subtitle region saved")

            img = cv2.imread(str(folder_path / f"cropped_{mid_point}.jpg"))
            if img is None:
                raise Exception("Error reading cropped image")

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(str(folder_path / f"grayscale_{mid_point}.jpg"), gray_img)

            logger.info("Grayscale conversion done")


            ### HERE IS THE MAIN LOGIC FOR GETTIG THE IMAGE FOR TESSERACT LIKE CONVERTING IT TO THE BLACK TEXT ON WHITE BACKGROUND


            # --- Improved Preprocessing ---
            # 1. Convert to grayscale (already done above)
            # 2. Apply Gaussian Blur to reduce noise
            blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
            
            # 3. Apply Otsu's Thresholding (works better for subtitles)
            # This will automatically find the best threshold and invert the colors
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # 4. Resize for better OCR
            big = cv2.resize(binary, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

            ## TILL HHERE IS THE LOGIC WHICH ALSO INCLUDES RESIZING OF THE IMAGE

            cv2.imwrite(str(folder_path / f"debug_{mid_point}.jpg"), big)
            logger.info(f"Debug image saved to {folder_path / f'debug_{mid_point}.jpg'}")
            
            # Using PSM 6 (Assume a single uniform block of text)
            custom_config = r'--oem 3 --psm 6'
            ocr_text = pytesseract.image_to_string(Image.open(str(folder_path / f"debug_{mid_point}.jpg")), lang='eng+hin+kan', config=custom_config)

            file_path = folder_path/"ocr_output.json"
            data_ocr.append({"timestamp":mid_point,"ocr_text":ocr_text})

            logger.info(f"OCR output for {mid_point} extracted")
            
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data_ocr,f,ensure_ascii=False,indent=4)
            
        logger.info(f"OCR full output saved to {file_path}")
        return str(file_path)

    except Exception as e:
        logger.error(f"Failed to extract subtitles: {e}")
        raise Exception("Failed to extract subtitles")

    finally:
        if vid is not None:
            vid.release()
        cv2.destroyAllWindows()
