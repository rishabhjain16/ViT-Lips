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
from datasets.utils import load_phoneme_dict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_video(video_path, image_size=224):
    """
    Load video frames using OpenCV
    
    Args:
        video_path: Path to video file
        image_size: Size to resize frames to
        
    Returns:
        frames: List of numpy arrays [H, W, C]
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
        frame = cv2.resize(frame, (image_size, image_size))
        frames.append(frame)
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"No frames loaded from: {video_path}")
    
    logger.info(f"Loaded {len(frames)} frames from {video_path}")
    return frames


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
    
    frames = load_video(video_path, image_size=target_size[0])
    if frames is None or len(frames) == 0:
        raise ValueError(f"Could not load video: {video_path}")
    
    # Convert to numpy array
    frames = np.array(frames)  # [T, H, W, C]
    
    # Convert to tensor and normalize
    frames = torch.from_numpy(frames).float()  # [T, H, W, C]
    frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]
    frames = frames / 255.0  # Normalize to [0, 1]
    
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


def predict_video(model, video_path, device, label_type='phn', phoneme_dict=None):
    """Predict lip reading for a single video"""
    
    try:
        # Preprocess video
        video_tensor = preprocess_video(video_path)
        video_tensor = video_tensor.to(device)
        
        logger.info(f"Video shape: {video_tensor.shape}")
        
        # Predict
        with torch.no_grad():
            # Get raw logits and log probabilities
            raw_logits = model.forward(video_tensor)
            log_probs = model.get_log_probs(video_tensor)
            logger.info(f"Raw logits shape: {raw_logits.shape}")
            logger.info(f"Raw logits (first frame): {raw_logits[0,0,:].cpu().numpy()}")
            logger.info(f"Log probs (first frame): {log_probs[0,0,:].cpu().numpy()}")
            predictions = model.decode_greedy(video_tensor)
            logger.info(f"Predictions: {predictions}")
        # Decode predictions
        decoded = decode_predictions(predictions, label_type, phoneme_dict=phoneme_dict)
        
        return decoded[0] if decoded else ""
        
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Inference for ViT Lip Reading Model')
    
    # Data arguments
    parser.add_argument('--data_root', type=str,
                       help='Path to dataset directory (for loading phoneme dictionary)')
    parser.add_argument('--output_dir', type=str, default='./outputs/inference',
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for inference (currently not used)')
    
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
    
    # Load phoneme dictionary if using phonemes and data_root is provided
    phoneme_dict = None
    if args.label_type == 'phn' and args.data_root:
        dict_path = os.path.join(args.data_root, "dict.phn.txt")
        if os.path.exists(dict_path):
            try:
                phoneme_dict = load_phoneme_dict(dict_path)
                logger.info(f"Loaded phoneme dictionary with {len(phoneme_dict)} phonemes")
            except Exception as e:
                logger.warning(f"Failed to load phoneme dictionary: {e}")
        else:
            logger.warning(f"Phoneme dictionary not found at {dict_path}")
    
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
    
    elif args.data_root:
        # If no specific video provided, read from TSV manifest files
        logger.info("No specific video provided, reading video paths from dataset manifests...")
        
        # Try test split first, then valid split
        for split in ['test', 'valid']:
            tsv_path = os.path.join(args.data_root, f"{split}.tsv")
            if os.path.exists(tsv_path):
                logger.info(f"Reading video paths from {tsv_path}")
                
                with open(tsv_path, 'r') as f:
                    lines = f.readlines()
                
                # Skip header (root path)
                for i, line in enumerate(lines[1:6]):  # Take first 5 videos as samples
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        video_path = parts[1]  # Second column is video path
                        if os.path.exists(video_path):
                            video_files.append(video_path)
                            logger.info(f"Found video: {os.path.basename(video_path)}")
                        else:
                            logger.warning(f"Video file not found: {video_path}")
                
                if video_files:
                    break  # Found videos, no need to check other splits
        
        if not video_files:
            logger.warning("No valid video files found in dataset manifests")
    
    else:
        logger.error("Please provide either --video, --video_dir, or --data_root to find sample videos")
        return
    
    if not video_files:
        logger.error("No video files found")
        return
    
    logger.info(f"Found {len(video_files)} video files")
    
    # Process videos
    results = []
    
    for video_path in video_files:
        logger.info(f"Processing: {os.path.basename(video_path)}")
        
        prediction = predict_video(model, video_path, device, args.label_type, phoneme_dict)
        
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
