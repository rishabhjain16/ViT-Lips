"""
Utility functions for dataset loading and processing
"""

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)


def load_audio(audio_path, target_sample_rate=16000):
    """
    Load audio file using librosa
    
    Args:
        audio_path: Path to audio file (.wav)
        target_sample_rate: Target sample rate for audio
        
    Returns:
        audio: numpy array of audio samples
        sample_rate: sample rate of the audio
    """
    try:
        import librosa
        audio, sr = librosa.load(audio_path, sr=target_sample_rate)
        logger.debug(f"Loaded audio: {audio.shape} @{sr}Hz from {audio_path}")
        return audio, sr
    except ImportError:
        logger.warning("librosa not installed - install with: pip install librosa")
        return None, target_sample_rate
    except Exception as e:
        logger.error(f"Error loading audio from {audio_path}: {e}")
        return None, target_sample_rate


def load_manifest(manifest_path):
    """
    Load video manifest file (TSV format) - NO FILTERING, keep all data
    
    LRS3 TSV format:
    /  (root directory - usually just "/")
    file_id    video_path    audio_path    num_frames    audio_samples
    
    Args:
        manifest_path: Path to the .tsv manifest file
        
    Returns:
        root: Root directory path (usually "/")
        video_info: List of (file_id, video_path, audio_path, num_frames, audio_samples) tuples
        indices: List of all valid indices
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    
    video_info = []
    indices = []
    
    with open(manifest_path, 'r') as f:
        root = f.readline().strip()
        
        for idx, line in enumerate(f):
            parts = line.strip().split('\t')
            if len(parts) < 5:  # Need at least 5 columns for LRS3 format
                logger.warning(f"Skipping line {idx+1}: insufficient columns ({len(parts)} < 5)")
                continue
                
            file_id = parts[0]
            video_path = parts[1]
            audio_path = parts[2]
            num_frames = int(parts[3])
            audio_samples = int(parts[4])
            
            # Keep ALL data - no filtering
            video_info.append((file_id, video_path, audio_path, num_frames, audio_samples))
            indices.append(idx)
    
    logger.info(f"Loaded {len(video_info)} videos from {manifest_path} (no filtering applied)")
    
    return root, video_info, indices


def load_labels(label_path, indices=None):
    """
    Load labels from file
    
    Args:
        label_path: Path to the label file
        indices: List of indices to keep (optional)
        
    Returns:
        labels: List of label strings
    """
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")
    
    with open(label_path, 'r') as f:
        labels = [line.strip() for line in f]
    
    if indices is not None:
        labels = [labels[i] for i in indices]
    
    logger.info(f"Loaded {len(labels)} labels from {label_path}")
    return labels


def load_phoneme_dict(dict_path):
    """
    Load phoneme dictionary from file
    
    Args:
        dict_path: Path to dictionary file (format: phoneme index)
        
    Returns:
        phoneme_dict: Dictionary mapping phonemes to indices
    """
    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"Dictionary file not found: {dict_path}")
    
    phoneme_dict = {}
    with open(dict_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                phoneme = parts[0]
                index = int(parts[1])
                phoneme_dict[phoneme] = index
    
    # Ensure blank token exists for CTC
    if '<blank>' not in phoneme_dict:
        # Add blank at index 0 and shift others
        shifted_dict = {'<blank>': 0}
        for phoneme, idx in phoneme_dict.items():
            shifted_dict[phoneme] = idx + 1
        phoneme_dict = shifted_dict
    
    logger.info(f"Loaded phoneme dictionary with {len(phoneme_dict)} phonemes")
    return phoneme_dict
