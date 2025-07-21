#!/usr/bin/env python3
"""
Simple inference script for ViT lip reading model
"""

import os
import sys
import logging
import argparse
import torch
import cv2
import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.vit_lipreading import create_phoneme_model, create_word_model
from datasets.utils import load_video

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(checkpoint_path, label_type='phn', device='cpu'):
    """Load trained model from checkpoint"""
    
    # Create model
    if label_type == 'phn':
        model = create_phoneme_model(pretrained=True)
    else:
        vocab_size = 1000  # Adjust based on your vocabulary
        model = create_word_model(vocab_size=vocab_size, pretrained=True)
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path}. Using untrained model.")
    
    model = model.to(device)
    model.eval()
    return model


def preprocess_video(video_path, target_fps=25, target_size=(224, 224)):
    """Preprocess video for inference"""
    
    frames = load_video(video_path)
    if frames is None or len(frames) == 0:
        raise ValueError(f"Could not load video: {video_path}")
    
    # Convert to numpy array
    frames = np.array(frames)  # [T, H, W, C]
    
    # Resize frames
    resized_frames = []
    for frame in frames:
        resized = cv2.resize(frame, target_size)
        resized_frames.append(resized)
    
    frames = np.array(resized_frames)  # [T, H, W, C]
    
    # Convert to tensor and normalize
    frames = torch.from_numpy(frames).float()  # [T, H, W, C]
    frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]
    frames = frames / 255.0  # Normalize to [0, 1]
    
    # Normalize using ImageNet stats (since we use pretrained ViT)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    frames = (frames - mean) / std
    
    # Add batch dimension: [T, C, H, W] -> [1, C, T, H, W]
    frames = frames.transpose(0, 1).unsqueeze(0)
    
    return frames


def decode_predictions(predictions, label_type='phn', phoneme_dict=None):
    """Decode model predictions to text"""
    
    if label_type == 'phn':
        # Create reverse phoneme mapping
        if phoneme_dict is not None:
            idx_to_phoneme = {v: k for k, v in phoneme_dict.items()}
        else:
            # Fallback mapping
            idx_to_phoneme = {i: f'PHN_{i}' for i in range(1, 41)}
        
        decoded_sequences = []
        for pred_seq in predictions:
            phonemes = [idx_to_phoneme.get(idx, f'UNK({idx})') for idx in pred_seq]
            decoded_sequences.append(' '.join(phonemes))
        
        return decoded_sequences
    else:
        # For word-level, you'd need a vocabulary mapping
        # This is a placeholder
        return [' '.join([f'WORD_{idx}' for idx in pred_seq]) for pred_seq in predictions]


def predict_video(model, video_path, device, label_type='phn'):
    """Predict lip reading for a single video"""
    
    try:
        # Preprocess video
        video_tensor = preprocess_video(video_path)
        video_tensor = video_tensor.to(device)
        
        logger.info(f"Video shape: {video_tensor.shape}")
        
        # Predict
        with torch.no_grad():
            predictions = model.decode_greedy(video_tensor)
        
        # Decode predictions
        decoded = decode_predictions(predictions, label_type, phoneme_dict=None)
        
        return decoded[0] if decoded else ""
        
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Inference for ViT Lip Reading Model')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--label_type', type=str, default='phn', choices=['phn', 'wrd'],
                       help='Type of labels (phn=phonemes, wrd=words)')
    
    # Input arguments
    parser.add_argument('--video', type=str,
                       help='Path to single video file')
    parser.add_argument('--video_dir', type=str,
                       help='Directory containing video files')
    parser.add_argument('--output_file', type=str,
                       help='Output file to save predictions')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, args.label_type, device)
    logger.info("Model loaded successfully")
    
    # Collect video files
    video_files = []
    if args.video:
        if os.path.exists(args.video):
            video_files = [args.video]
        else:
            logger.error(f"Video file not found: {args.video}")
            return
    
    elif args.video_dir:
        if os.path.exists(args.video_dir):
            for file in os.listdir(args.video_dir):
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_files.append(os.path.join(args.video_dir, file))
        else:
            logger.error(f"Video directory not found: {args.video_dir}")
            return
    
    else:
        logger.error("Please provide either --video or --video_dir")
        return
    
    if not video_files:
        logger.error("No video files found")
        return
    
    logger.info(f"Found {len(video_files)} video files")
    
    # Process videos
    results = []
    
    for video_path in video_files:
        logger.info(f"Processing: {os.path.basename(video_path)}")
        
        prediction = predict_video(model, video_path, device, args.label_type)
        
        if prediction is not None:
            result = {
                'video': os.path.basename(video_path),
                'prediction': prediction
            }
            results.append(result)
            
            logger.info(f"Prediction: {prediction}")
        else:
            logger.error(f"Failed to process: {video_path}")
    
    # Save results
    if args.output_file and results:
        logger.info(f"Saving results to: {args.output_file}")
        
        with open(args.output_file, 'w') as f:
            for result in results:
                f.write(f"{result['video']}\t{result['prediction']}\n")
    
    logger.info(f"Processed {len(results)}/{len(video_files)} videos successfully")


if __name__ == '__main__':
    main()
