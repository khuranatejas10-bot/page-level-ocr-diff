"""Data package."""
from .synthetic_page_generator import SyntheticPageGenerator, generate_word_crops
from .datasets import DetectorDataset, RecognizerDataset, SRPairDataset, recognizer_collate
