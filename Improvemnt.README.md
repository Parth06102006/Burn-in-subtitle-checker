# Improvement README

This document records the current limitations found during testing and the next improvement areas for the burn-in subtitle checker. The main goal is to improve OCR accuracy, language handling, and timestamp-level comparison between audio transcription and embedded subtitles.

## Current Pipeline Limitation

At present, the pipeline transcribes the full audio first, saves the transcript segments, and then runs OCR separately for each segment using sampled frames. This works, but the audio and video checks are still treated as two separate stages.

The preferred improvement is to make the OCR check happen segment-by-segment while working with the transcription segment timestamps. For every Whisper segment, the system should immediately inspect the matching video frame region and compare OCR text with the transcription context. This would make the comparison more precise because each OCR decision can use the actual segment text, language, and timing while processing that segment, instead of OCRing multiple frames first and comparing later.

## Observed Issues

### 1. OCR Results From Multiple Frames Can Add Noise

The current extractor samples three frames for every segment:

- `15%` from the segment start
- `50%` from the segment start
- `85%` from the segment start

The three OCR outputs are then consolidated. This helps when subtitles are stable, but it can also create issues when the subtitle changes inside the segment or when one of the frames contains partial/fading text. In some cases, the combined result can include repeated characters or repeated fragments.

Improvement:

- Start with the middle frame first because it is usually the most stable point inside a segment.
- Compare the middle-frame OCR result directly with the transcript segment.
- Only inspect the start or end frames when the middle frame is empty, low quality, or does not match the transcript.
- Avoid joining start, middle, and end OCR text unless there is clear evidence that the subtitle genuinely changed during the segment.

Relevant area: `src/subtitle_extractor.py`, especially `process_single_frame()` and `consolidate_ocr_results()`.

### 2. Repeated Characters Still Appear

Some OCR outputs still contain repeated characters or repeated fragments. A check has already been added to reduce this, but the problem can still appear when Tesseract receives a noisy crop or when multiple subtitle lines/strips are detected together.

The suspected cause is that the sampled crop can sometimes include more visual information than the actual subtitle text. Even if the current suspicious-shape check has improved this, OCR can still treat noisy borders, overlays, or multiple text rows as part of the subtitle.

Improvement:

- Strengthen text-line detection before OCR.
- Add connected-component filtering to keep only subtitle-sized text components.
- Reject OCR crops with too many text rows unless the transcript segment likely contains multi-line subtitle text.
- Store per-frame OCR confidence and reject low-confidence results before consolidation.

Relevant area: `src/subtitle_extractor.py`, especially `preprocess_for_ocr()`, `warn_if_suspicious_ocr_shape()`, `remove_repetitions()`, and `get_ocr_quality_score()`.

### 3. Whisper Language Detection Can Be Wrong

Testing was done on 15 video files. In 2 files, the detected audio language did not match the actual audio language or the produced transcription language. Because OCR language selection depends on Whisper's detected language, this can make Tesseract run with the wrong language.

This issue was not frequent, but it is important because one wrong language decision can affect every segment in that video.

Improvement:

- Do not rely only on one video-level Whisper language value.
- Add segment-level language confidence where possible.
- Retry OCR with a small ranked set of languages when the transcript text and OCR script do not match.
- Add a report warning when Whisper language detection and OCR script detection disagree repeatedly.

Relevant area: `src/transcriber.py` and `src/subtitle_extractor.py`.

### 4. Fixed Bottom Crop Ratio Does Not Fit All Video Orientations

The current subtitle crop uses the lower part of the frame:

- `15%` bottom region for landscape videos
- `20%` bottom region for portrait videos

This helped with some portrait edge cases, such as reels or shorts, but it is still not suitable for all portrait videos. Subtitle placement can vary a lot depending on platform, aspect ratio, editing style, burned-in captions, and UI overlays.

Improvement:

- Detect likely subtitle rows dynamically using contrast, edges, and text-like connected components.
- Search multiple candidate subtitle zones instead of assuming only the bottom strip.
- Use different crop profiles for landscape, portrait, square, and letterboxed videos.
- Save the selected crop coordinates in `ocr_output.json` for debugging.

Relevant area: `src/subtitle_extractor.py`, especially `get_subtitle_strip()` and `find_real_bottom()`.

### 5. Language Support Is Currently Limited

The current language map supports only:

- English: `eng`
- Hindi: `hin`
- Kannada: `kan`

This map is defined in `src/subtitle_extractor.py` as `WHISPER_TO_TESSERACT`. Future improvement should make language support more flexible, especially for multilingual videos.

Improvement:

- Move the language map to a config file.
- Support more Whisper-to-Tesseract language mappings.
- Validate that the required Tesseract language data is installed before processing starts.
- Allow the user to pass allowed OCR languages from the CLI.

Relevant area: `src/subtitle_extractor.py` and `main.py`.

### 6. Multi-Language OCR Can Prefer English Noise

When multiple OCR languages are used together, such as `eng+hin+kan`, the OCR result can incorrectly prefer English text because video frames may contain timestamps, navigation labels, platform UI, or other English overlays. This causes OCR to return visible UI text instead of the actual Hindi or Kannada subtitle.

Improvement:

- Crop more tightly around subtitle text before running multi-language OCR.
- Use script detection after OCR to verify that the result matches the expected transcript script.
- Mask known UI areas more aggressively for YouTube, reels, shorts, and mobile recordings.
- Prefer the language/script that best matches the transcript text, not the language that produced the longest OCR output.

Relevant area: `src/subtitle_extractor.py`, especially `run_ocr()`, `is_valid_indic_text()`, and `get_expected_script_chars()`.

### 7. YouTube Auto-Generated Subtitles Perform Worse

Testing included YouTube videos with auto-generated subtitles and videos with subtitles already embedded into the video. The embedded subtitles produced better OCR results than YouTube auto-generated subtitles.

This is expected because YouTube auto-generated subtitles can have variable styling, placement, animation, and background overlays. They may also appear near controls or other UI elements depending on the captured video.

Improvement:

- Treat platform-rendered subtitles as a separate mode from burned-in subtitles.
- Add YouTube-specific crop/masking presets.
- Detect subtitle background boxes and caption overlays before OCR.
- Record whether a test video contains true burned-in subtitles or player-rendered captions.

Relevant area: `src/subtitle_extractor.py` and the test dataset organization.

### 8. Text Style Preprocessing Needs More Work

Current preprocessing handles styles such as:

- `outlined_white`
- `light_on_dark`
- `dark_on_light`

The target OCR image should ideally become black text on a white background. However, testing showed that `adaptiveThreshold` with `ADAPTIVE_THRESH_GAUSSIAN_C` can turn some outlined or bordered text into gray/noisy output. For black-bordered text, OCR can still fail because the outline and text body are not always reconstructed correctly.

The current majority-color check helps decide whether to invert the image, but this adds processing time and still does not solve every subtitle style.

Improvement:

- Use connected components and stroke-width filtering to separate text from outline.
- Add style-specific preprocessing paths and benchmark them separately.
- Cache preprocessing decisions per video when subtitle style remains stable.
- Add OCR confidence-based selection between multiple preprocessed image variants.

Relevant area: `src/subtitle_extractor.py`, especially `detect_text_style()`, `fix_outlined_text()`, `make_black_text_image()`, and `preprocess_for_ocr()`.

### 9. Add Per-Video Language and Script Validation

Before processing all segments, inspect a few transcript segments and OCR samples to estimate the likely script. If the transcript appears Hindi but OCR returns mostly English, the system should lower trust in English OCR output.

Why:

This directly addresses the issue where `eng+hin+kan` can produce English UI text instead of subtitle text.

### 10. Use OCR Confidence When Available

Tesseract can return word-level data through `image_to_data()`. EasyOCR also returns confidence values when detail output is enabled.

Why:

The current system mainly scores text after OCR. Confidence values would help reject bad OCR earlier and choose between multiple frame/preprocessing candidates more reliably.

### 11. Separate OCR Extraction From Comparison Decisions

The OCR extractor currently tries to decide the best OCR text before the mismatch detector compares it with the transcript. A future design could return multiple OCR candidates per segment and let the comparison step choose the best one using transcript similarity.

Why:

This keeps OCR extraction flexible and makes the final decision more transcript-aware.


## Summary

The most important improvement is to make OCR extraction segment-aware and transcript-aware. Instead of processing three frames first and then comparing the merged OCR result, the system should begin with the most reliable frame, compare it with the transcript immediately, and only expand the search when needed.

This should reduce repeated characters, avoid mixed subtitle text, improve performance, and make the final mismatch report more accurate.
