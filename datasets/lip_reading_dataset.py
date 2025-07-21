"""
Simple Lip Reading Dataset for ViT-Lips

This module provides a simplified dataset class for loading LRS2/LRS3 data
for lip reading tasks with ViT transformers.
"""

import os
import logging
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

from .utils import load_manifest, load_labels, load_phoneme_dict, load_audio

logger = logging.getLogger(__name__)


class LipReadingDataset(Dataset):
    """
    Simple dataset for lip reading with video frames and text/phoneme labels
    """
    
    def __init__(
        self,
        data_dir,
        split='train',
        label_type='phn',  # 'phn' for phonemes, 'wrd' for words/sentences
        image_size=224,
        transform=None,
        load_audio=False,  # Set to True if you want audio loading in future
        mask_prob=0.0      # Probability for masking frames (0.0 = no masking)
    ):
        """
        Args:
            data_dir: Directory containing dataset files
            split: Data split (train, valid, test)
            label_type: Type of labels ('phn' for phonemes, 'wrd' for words)
            image_size: Size to resize frames to
            transform: Optional transform to apply to frames
            load_audio: Whether to load audio data (placeholder for future)
            mask_prob: Probability for random frame masking during training
        """
        self.data_dir = data_dir
        self.split = split
        self.label_type = label_type
        self.image_size = image_size
        self.transform = transform
        self.load_audio = load_audio
        self.mask_prob = mask_prob
        
        # File paths
        self.manifest_path = os.path.join(data_dir, f"{split}.tsv")
        self.label_path = os.path.join(data_dir, f"{split}.{label_type}")
        
        # Load manifest and labels (no filtering - keep all data)
        self.root, self.video_info, self.indices = load_manifest(self.manifest_path)
        
        self.labels = load_labels(self.label_path, self.indices)
        
        # Load phoneme dictionary if using phonemes
        self.phoneme_dict = None
        if label_type == 'phn':
            dict_path = os.path.join(data_dir, "dict.phn.txt")
            if os.path.exists(dict_path):
                self.phoneme_dict = load_phoneme_dict(dict_path)
            else:
                logger.warning(f"Phoneme dictionary not found at {dict_path}")
        
        logger.info(f"Initialized {split} dataset with {len(self)} samples")
    
    def __len__(self):
        return len(self.video_info)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        file_id, video_path, audio_path, num_frames, audio_samples = self.video_info[idx]
        label = self.labels[idx]
        
        # Load video frames - video_path is already absolute
        frames = self.load_video(video_path)
        
        # Apply masking if specified
        if self.mask_prob > 0.0:
            frames = self.apply_frame_masking(frames, self.mask_prob)
        
        # Apply transform if provided
        if self.transform is not None:
            frames = self.transform(frames)
        
        # Load audio if requested (placeholder for now)
        audio_data = None
        if self.load_audio:
            audio_data, sample_rate = load_audio(audio_path)
        
        # Process label based on type
        if self.label_type == 'phn' and self.phoneme_dict is not None:
            # Convert phonemes to indices
            phoneme_tokens = label.split()
            label_indices = [self.phoneme_dict.get(p, 0) for p in phoneme_tokens]
            label_tensor = torch.LongTensor(label_indices)
        else:
            # Keep text as string for now
            label_tensor = label
        
        return {
            'video': frames,           # [C, T, H, W] - Video frames
            'audio': audio_data,       # Audio samples (None if not loaded)
            'label': label_tensor,     # String or LongTensor depending on label_type
            'file_id': file_id,        # Unique identifier
            'video_path': video_path,  # Absolute path to video
            'audio_path': audio_path   # Absolute path to audio
        }
    
    def load_video(self, video_path):
        """
        Load video frames using OpenCV
        
        Args:
            video_path: Path to video file
            
        Returns:
            frames: Tensor of shape [C, T, H, W]
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame
            frame = cv2.resize(frame, (self.image_size, self.image_size))
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames loaded from: {video_path}")
        
        # Convert to tensor [T, H, W, C] -> [C, T, H, W]
        frames = np.stack(frames, axis=0)
        frames = torch.from_numpy(frames).float() / 255.0  # Normalize to [0, 1]
        frames = frames.permute(3, 0, 1, 2)  # [C, T, H, W]
        
        return frames
    
    def apply_frame_masking(self, frames, mask_prob):
        """
        Apply random frame masking for self-supervised learning
        
        Args:
            frames: Video frames tensor [C, T, H, W]
            mask_prob: Probability of masking each frame
            
        Returns:
            masked_frames: Frames with some randomly masked (set to 0)
        """
        if mask_prob <= 0.0:
            return frames
            
        C, T, H, W = frames.shape
        mask = torch.rand(T) < mask_prob  # Random mask for each frame
        
        masked_frames = frames.clone()
        masked_frames[:, mask, :, :] = 0.0  # Set masked frames to black
        
        logger.debug(f"Masked {mask.sum().item()}/{T} frames (prob={mask_prob})")
        return masked_frames
    
    def get_sample_info(self, idx):
        """Get information about a specific sample"""
        file_id, video_path, audio_path, num_frames, audio_samples = self.video_info[idx]
        label = self.labels[idx]
        
        info = {
            'index': idx,
            'file_id': file_id,
            'video_path': video_path,
            'audio_path': audio_path,
            'num_frames': num_frames,
            'audio_samples': audio_samples,
            'label': label,
            'label_type': self.label_type,
            'load_audio': self.load_audio,
            'mask_prob': self.mask_prob
        }
        
        if self.label_type == 'phn':
            phonemes = label.split()
            info['num_phonemes'] = len(phonemes)
            info['phonemes'] = phonemes
        
        return info
