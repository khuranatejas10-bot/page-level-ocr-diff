"""Pipeline package."""
from .preprocessing import preprocess_page, normalize_for_detector, normalize_word_crop, extract_word_crop
from .components import extract_components, TextComponent
from .grouping import build_layout, group_components_into_blocks, TextBlock, TextLine
from .reading_order import determine_reading_order, extract_ordered_text
from .postprocess import postprocess_page_text, segment_words, filter_low_confidence
from .inference import PageOCRPipeline, PageOCRResult
