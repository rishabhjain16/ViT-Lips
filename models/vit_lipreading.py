#!/usr/bin/env python3
"""
Vision Transformer model for lip reading with CTC decoding
Simple and clean implementation for phoneme and word prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights
import logging

logger = logging.getLogger(__name__)


class ViTLipReading(nn.Module):
    """
    Simple Vision Transformer for lip reading with CTC loss
    
    Features:
    - Uses pretrained ViT-B/16 as backbone
    - Simple temporal processing
    - CTC head for sequence prediction
    - Supports both phoneme and word-level prediction
    """
    
    def __init__(
        self,
        num_classes=41,  # 40 phonemes + 1 CTC blank (index 0)
        pretrained=True,
        hidden_dim=256,
        dropout=0.1,
        freeze_backbone=False
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Load pretrained ViT-B/16
        if pretrained:
            self.backbone = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        else:
            self.backbone = vit_b_16()
        
        # Remove the classification head
        self.backbone.heads = nn.Identity()
        
        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_backbone()
        
        # ViT-B/16 outputs 768-dim features
        backbone_dim = 768
        
        # Temporal feature processing
        self.temporal_encoder = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        
        # CTC prediction head
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Initialize classifier - reduce blank token probability
        with torch.no_grad():
            self.classifier.bias[0] = -2.0  # Make blank less likely
    
    def freeze_backbone(self):
        """Freeze ViT backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("ViT backbone frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze ViT backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("ViT backbone unfrozen")
    
    def forward(self, videos, frame_batch_size=32):
        """
        Forward pass
        
        Args:
            videos: [B, C, T, H, W] video sequences
            frame_batch_size: Number of frames to process at once to save memory
            
        Returns:
            logits: [B, T, num_classes] - logits for each timestep
        """
        B, C, T, H, W = videos.shape
        
        # Process frames in smaller batches to save memory
        frame_features_list = []
        
        for b in range(B):
            batch_features = []
            video = videos[b]  # [C, T, H, W]
            
            # Process frames in chunks
            for i in range(0, T, frame_batch_size):
                end_idx = min(i + frame_batch_size, T)
                frame_chunk = video[:, i:end_idx]  # [C, chunk_size, H, W]
                chunk_size = frame_chunk.shape[1]
                
                # Reshape for ViT: [C, chunk_size, H, W] -> [chunk_size, C, H, W]
                frames = frame_chunk.transpose(0, 1)  # [chunk_size, C, H, W]
                
                # Extract features using ViT backbone
                chunk_features = self.backbone(frames)  # [chunk_size, 768]
                batch_features.append(chunk_features)
            
            # Concatenate all chunks for this video
            video_features = torch.cat(batch_features, dim=0)  # [T, 768]
            frame_features_list.append(video_features)
        
        # Stack all videos: [B, T, 768]
        frame_features = torch.stack(frame_features_list, dim=0)
        
        # Temporal encoding
        temporal_features = self.temporal_encoder(frame_features)  # [B, T, hidden_dim]
        
        # Classification
        logits = self.classifier(temporal_features)  # [B, T, num_classes]
        
        return logits
    
    def get_log_probs(self, videos):
        """Get log probabilities for CTC loss"""
        logits = self.forward(videos)
        return F.log_softmax(logits, dim=-1)
    
    def compute_ctc_loss(self, videos, targets, target_lengths, input_lengths=None):
        """
        Compute CTC loss
        
        Args:
            videos: [B, C, T, H, W] video sequences
            targets: [sum(target_lengths)] concatenated target sequences
            target_lengths: [B] length of each target sequence
            input_lengths: [B] actual length of each input sequence (optional)
            
        Returns:
            loss: CTC loss value
        """
        log_probs = self.get_log_probs(videos)  # [B, T, num_classes]
        B, T, _ = log_probs.shape
        
        # Use provided input_lengths or assume all frames are valid
        if input_lengths is None:
            input_lengths = torch.full((B,), T, dtype=torch.long, device=videos.device)
        else:
            input_lengths = input_lengths.to(videos.device)
        
        # Check CTC constraints (input_length >= target_length)
        valid_mask = input_lengths >= target_lengths
        if not valid_mask.all():
            # Filter out invalid samples
            valid_indices = valid_mask.nonzero().squeeze(-1)
            if len(valid_indices) == 0:
                # No valid samples - return zero loss
                return torch.tensor(0.0, requires_grad=True, device=videos.device)
            
            # Use only valid samples
            log_probs = log_probs[valid_indices]
            input_lengths = input_lengths[valid_indices]
            
            # Adjust targets for valid samples
            new_targets = []
            new_target_lengths = []
            start_idx = 0
            for i, length in enumerate(target_lengths):
                if i in valid_indices:
                    new_targets.append(targets[start_idx:start_idx + length])
                    new_target_lengths.append(length)
                start_idx += length
            
            if new_targets:
                targets = torch.cat(new_targets)
                target_lengths = torch.stack(new_target_lengths)
            else:
                return torch.tensor(0.0, requires_grad=True, device=videos.device)
        
        # CTC loss expects [T, B, num_classes]
        log_probs = log_probs.transpose(0, 1)
        
        loss = F.ctc_loss(
            log_probs=log_probs,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            blank=0,  # Blank token index
            reduction='mean',
            zero_infinity=True
        )
        
        return loss
    
    def decode_greedy(self, videos):
        """
        Greedy CTC decoding
        
        Args:
            videos: [B, C, T, H, W] video sequences
            
        Returns:
            predictions: List of decoded sequences (list of token indices)
        """
        log_probs = self.get_log_probs(videos)  # [B, T, num_classes]
        
        # Get most likely token at each timestep
        predicted_tokens = torch.argmax(log_probs, dim=-1)  # [B, T]
        
        # Remove blanks and consecutive duplicates
        predictions = []
        for batch_idx in range(predicted_tokens.shape[0]):
            sequence = []
            prev_token = None
            
            for token in predicted_tokens[batch_idx]:
                token = token.item()
                # Skip blank tokens (0) and consecutive duplicates
                if token != 0 and token != prev_token:
                    sequence.append(token)
                prev_token = token
            
            predictions.append(sequence)
        
        return predictions


def create_phoneme_model(pretrained=True, freeze_backbone=False):
    """Create model for phoneme prediction"""
    # 40 phonemes + 1 blank token = 41 classes
    return ViTLipReading(
        num_classes=41,
        pretrained=pretrained,
        hidden_dim=256,
        dropout=0.1,
        freeze_backbone=freeze_backbone
    )


def create_word_model(vocab_size, pretrained=True, freeze_backbone=False):
    """Create model for word-level prediction"""
    # vocab_size + 1 blank token
    return ViTLipReading(
        num_classes=vocab_size + 1,
        pretrained=pretrained,
        hidden_dim=256,
        dropout=0.1,
        freeze_backbone=freeze_backbone
    )


if __name__ == "__main__":
    # Test the model
    logging.basicConfig(level=logging.INFO)
    
    # Create model
    model = create_phoneme_model()
    
    # Test input
    batch_size = 2
    video = torch.randn(batch_size, 3, 25, 224, 224)  # 25 frames, 224x224 resolution
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input shape: {video.shape}")
    
    # Forward pass
    with torch.no_grad():
        logits = model(video)
        print(f"Output shape: {logits.shape}")
        
        # Test decoding
        predictions = model.decode_greedy(video)
        print(f"Predictions: {predictions}")
    
    print("✅ Model test successful!")
