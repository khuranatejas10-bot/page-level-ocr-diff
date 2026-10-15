"""
Post-processing for raw recognizer output.
Paper 1: WordNinja word segmentation, confidence bleed-through filtering.
"""
from typing import List, Tuple, Optional
import re


try:
    import wordninja
    _HAS_WORDNINJA = True
except ImportError:
    _HAS_WORDNINJA = False

try:
    from rapidfuzz.distance import Levenshtein as _Lev
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False


def segment_words(text: str) -> List[str]:
    """
    Use WordNinja to split merged/concatenated text into dictionary words.
    Falls back to whitespace splitting when WordNinja is unavailable.
    """
    if not text:
        return []
    if _HAS_WORDNINJA:
        return wordninja.split(text)
    return text.split()


def filter_low_confidence(
    texts: List[str],
    confidences: List[float],
    threshold: float = 0.3,
    bleed_sigma: float = 0.15,
) -> List[Tuple[str, float]]:
    """
    Confidence bleed-through filtering.
    Paper 1: words with confidence below (block_avg - bleed_sigma * block_std)
    are discarded as bleed-through artefacts.

    Args:
        texts: list of recognised strings per word.
        confidences: list of CTC confidence scores in [0, 1].
        threshold: global hard-minimum confidence.
        bleed_sigma: number of std-devs below mean to gate bleed-through.

    Returns:
        Filtered list of (text, confidence) tuples.
    """
    if not texts:
        return []

    import numpy as np  # local import to keep module lightweight
    conf_arr = np.array(confidences)

    mu = conf_arr.mean()
    sigma = conf_arr.std()
    dynamic_thresh = max(threshold, mu - bleed_sigma * sigma)

    results = []
    for t, c in zip(texts, confidences):
        if c >= dynamic_thresh and t.strip():
            results.append((t, c))
    return results


def clean_ocr_text(text: str) -> str:
    """
    Basic text cleaning: strip trailing whitespace, normalise line-breaks,
    remove isolated punctuation runs.
    """
    # Normalise whitespace within lines
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            cleaned.append(line)
    return "\n".join(cleaned)


def merge_hyphenated_words(text: str) -> str:
    """Merge hyphen-broken line-end words across line boundaries."""
    return re.sub(r"-\n(\S)", r"\1", text)


def postprocess_page_text(
    line_texts: List[str],
    line_confidences: List[float],
    do_wordninja: bool = True,
    confidence_threshold: float = 0.2,
) -> Tuple[str, List[str]]:
    """
    Full postprocessing pipeline for a recognised page.

    Args:
        line_texts: list of raw OCR text per line.
        line_confidences: per-line average confidence.
        do_wordninja: whether to apply WordNinja word segmentation.
        confidence_threshold: minimum confidence to include a line.

    Returns:
        (full_page_text, filtered_lines)
    """
    filtered = []
    for text, conf in zip(line_texts, line_confidences):
        if conf < confidence_threshold:
            continue
        text = clean_ocr_text(text)
        if not text:
            continue
        if do_wordninja and not any(c.isspace() for c in text):
            words = segment_words(text)
            text = " ".join(words)
        filtered.append(text)

    full_text = "\n".join(filtered)
    full_text = merge_hyphenated_words(full_text)
    return full_text, filtered
