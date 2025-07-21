"""
ViT-Lips Dataset Module

This module provides dataset classes for lip reading tasks using LRS2/LRS3 datasets.
"""

from .lip_reading_dataset import LipReadingDataset
from .utils import load_manifest, load_labels, load_phoneme_dict, load_audio
from .config import DatasetConfig, LRS2Config, LRS3Config

__all__ = [
    'LipReadingDataset',
    'load_manifest',
    'load_labels',
    'load_phoneme_dict',
    'load_audio',
    'DatasetConfig',
    'LRS2Config',
    'LRS3Config'
]
