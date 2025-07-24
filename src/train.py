#!/usr/bin/env python3
"""
Simple training script for ViT lip reading model
Clean and readable implementation
"""

import os
import sys
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False
import numpy as np
import random

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.vit_lipreading import create_phoneme_model, create_word_model
from datasets.lip_reading_dataset import LipReadingDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_to_file(log_file_path, epoch, batch_num, loss, avg_loss, lr=None, validation=False):
    """Log training/validation losses to a text file"""
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    mode = 'a' if os.path.exists(log_file_path) else 'w'
    with open(log_file_path, mode) as f:
        if mode == 'w':
            # Write header for new file
            f.write("timestamp,epoch,batch,loss,avg_loss,lr,type\n")
        
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_type = "validation" if validation else "training"
        lr_str = f"{lr:.6f}" if lr is not None else "N/A"
        
        f.write(f"{timestamp},{epoch},{batch_num},{loss:.6f},{avg_loss:.6f},{lr_str},{log_type}\n")


def collate_fn(batch):
    """
    Custom collate function for CTC training
    Handles variable length sequences by padding
    """
    # Batch is a list of dictionaries from dataset.__getitem__
    videos = [item['video'] for item in batch]
    labels = [item['label'] for item in batch]
    video_paths = [item['video_path'] for item in batch]
    
    # Get video lengths for CTC loss
    input_lengths = torch.tensor([video.shape[1] for video in videos], dtype=torch.long)
    
    # Pad videos to the same temporal length
    max_length = max(video.shape[1] for video in videos)
    batch_size = len(videos)
    c, h, w = videos[0].shape[0], videos[0].shape[2], videos[0].shape[3]
    
    padded_videos = torch.zeros(batch_size, c, max_length, h, w)
    for i, video in enumerate(videos):
        padded_videos[i, :, :video.shape[1], :, :] = video
    
    # Prepare targets for CTC loss
    target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    targets = torch.cat(labels)  # Concatenate all targets
    
    return {
        'videos': padded_videos,
        'targets': targets,
        'input_lengths': input_lengths,
        'target_lengths': target_lengths,
        'video_paths': video_paths
    }


def train_epoch(model, dataloader, optimizer, device, epoch, args):
    """Train for one epoch with gradient accumulation and mixed precision"""
    model.train()
    total_loss = 0
    num_batches = 0
    running_loss = 0  # For running average
    samples_processed = 0
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler(enabled=args.use_amp)
    
    # Create progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", 
                unit="batch", dynamic_ncols=True)
    
    for batch_idx, batch in enumerate(pbar):
        videos = batch['videos'].to(device)
        targets = batch['targets'].to(device)
        target_lengths = batch['target_lengths'].to(device)
        input_lengths = batch['input_lengths'].to(device)

        batch_size = videos.size(0)
        samples_processed += batch_size

        # Forward pass with mixed precision
        with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', enabled=args.use_amp):
            loss = model.compute_ctc_loss(videos, targets, target_lengths, input_lengths)
            loss = loss / args.gradient_accumulation_steps

        # Backward pass with scaler
        scaler.scale(loss).backward()
        
        # Update weights every gradient_accumulation_steps
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            # Gradient clipping with scaler
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights with scaler
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # CONSERVATIVE MEMORY CLEANUP: Only when memory usage is high
            if torch.cuda.is_available():
                memory_used_gb = torch.cuda.memory_allocated() / 1024**3
                if memory_used_gb > 15.0:  # Only clean if using >15GB
                    torch.cuda.empty_cache()
        
        # Emergency cleanup every 1000 batches to prevent fragmentation buildup
        if batch_idx % 1000 == 0 and torch.cuda.is_available():
            memory_used_gb = torch.cuda.memory_allocated() / 1024**3
            if memory_used_gb > 10.0:  # Only if really needed
                torch.cuda.empty_cache()
        
        # Track losses
        current_loss = loss.item() * args.gradient_accumulation_steps
        total_loss += current_loss
        running_loss = 0.9 * running_loss + 0.1 * current_loss  # Exponential moving average
        num_batches += 1
        
        # Update progress bar
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            memory_cached = torch.cuda.memory_reserved() / 1024**3
            pbar.set_postfix({
                'Loss': f'{current_loss:.3f}',
                'Avg': f'{running_loss:.3f}',
                'GPU': f'{memory_used:.1f}GB/{memory_cached:.1f}GB',
                'Samples': f'{samples_processed:,}'
            })
        else:
            pbar.set_postfix({
                'Loss': f'{current_loss:.3f}',
                'Avg': f'{running_loss:.3f}',
                'Samples': f'{samples_processed:,}'
            })
        
        # Sample-based checkpoint saving
        if samples_processed % args.save_every_samples == 0:
            save_checkpoint(
                model, optimizer, None, epoch, current_loss,
                os.path.join(args.output_dir, f'checkpoint_samples_{samples_processed}.pth')
            )
            logger.info(f"Saved checkpoint after {samples_processed:,} samples")
        
        # Regular logging (less frequent now with tqdm)
        if batch_idx % (args.log_interval * 10) == 0:  # Log every 100 batches instead of 10
            progress_pct = (batch_idx / len(dataloader)) * 100
            logger.info(f'Epoch {epoch}, Batch {batch_idx}/{len(dataloader)} ({progress_pct:.1f}%), '
                       f'Loss: {current_loss:.4f} (avg: {running_loss:.4f}), '
                       f'Samples: {samples_processed:,}')
            
            # Log to text file
            if hasattr(args, 'output_dir'):
                log_file = os.path.join(args.output_dir, 'training_log.txt')
                current_lr = optimizer.param_groups[0]['lr'] if hasattr(optimizer, 'param_groups') else None
                log_to_file(log_file, epoch, batch_idx, current_loss, running_loss, current_lr)
    
    pbar.close()
    avg_loss = total_loss / max(1, num_batches)
    logger.info(f'Epoch {epoch} completed - Average Loss: {avg_loss:.4f}, Total Samples: {samples_processed:,}')
    return avg_loss


def validate_epoch(model, dataloader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    # Create progress bar for validation
    pbar = tqdm(dataloader, desc="Validation", unit="batch", dynamic_ncols=True)
    
    with torch.no_grad():
        for batch in pbar:
            videos = batch['videos'].to(device)
            targets = batch['targets'].to(device)
            target_lengths = batch['target_lengths'].to(device)
            input_lengths = batch['input_lengths'].to(device)
            
            loss = model.compute_ctc_loss(videos, targets, target_lengths, input_lengths)
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            avg_loss_so_far = total_loss / num_batches
            pbar.set_postfix({'Val Loss': f'{avg_loss_so_far:.4f}'})
    
    pbar.close()
    avg_loss = total_loss / max(1, num_batches)
    logger.info(f'Validation Loss: {avg_loss:.4f}')
    return avg_loss


def test_predictions(model, dataloader, device, num_samples=3):
    """Test model predictions and show examples"""
    model.eval()
    
    # Get phoneme mapping from dataset
    dataset = dataloader.dataset
    if hasattr(dataset, 'phoneme_dict') and dataset.phoneme_dict is not None:
        idx_to_phoneme = {v: k for k, v in dataset.phoneme_dict.items()}
    else:
        # Fallback for word-level or if no phoneme dict
        idx_to_phoneme = {i: f'TOKEN_{i}' for i in range(41)}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_samples:
                break
            
            videos = batch['videos'].to(device)
            targets = batch['targets']
            target_lengths = batch['target_lengths']
            video_paths = batch['video_paths']
            
            # Get predictions
            predictions = model.decode_greedy(videos)
            
            # Show results for each sample in batch
            start_idx = 0
            for i, length in enumerate(target_lengths):
                target_seq = targets[start_idx:start_idx + length]
                pred_seq = predictions[i]
                
                # Convert to phonemes
                target_phonemes = [idx_to_phoneme.get(idx.item(), f'UNK({idx.item()})') 
                                 for idx in target_seq]
                pred_phonemes = [idx_to_phoneme.get(idx, f'UNK({idx})') 
                               for idx in pred_seq]
                
                logger.info(f"\nSample {batch_idx}-{i}: {os.path.basename(video_paths[i])}")
                logger.info(f"Target:    {' '.join(target_phonemes)}")
                logger.info(f"Predicted: {' '.join(pred_phonemes)}")
                
                start_idx += length


def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_path):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'loss': loss
    }, save_path)
    
    logger.info(f"Checkpoint saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train ViT Lip Reading Model')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of LRS3 dataset')
    parser.add_argument('--label_type', type=str, default='phn', choices=['phn', 'wrd'],
                       help='Type of labels to use (phn=phonemes, wrd=words)')
    parser.add_argument('--load_audio', action='store_true',
                       help='Load audio features (experimental)')
    
    # Model arguments
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained ViT backbone')
    # parser.add_argument('--freeze_backbone', action='store_true',
    #                    help='Freeze ViT backbone (train only head)')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True,
                       help='Use gradient checkpointing to save memory')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use automatic mixed precision (AMP) to save memory')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=1,
                       help='Batch size for validation')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Number of steps to accumulate gradients before update')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='step', 
                       choices=['step', 'cosine', 'none'],
                       help='Learning rate scheduler')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Log every N batches')
    parser.add_argument('--save_interval', type=int, default=1,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--save_every_samples', type=int, default=5000,
                       help='Save checkpoint every N samples processed')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    logger.info(f"Configuration: {vars(args)}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup tensorboard
    writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard'))
    
    # Create datasets
    logger.info("Loading datasets...")

    train_dataset = LipReadingDataset(
        data_dir=args.data_root,
        split='train',
        label_type=args.label_type,
        load_audio=args.load_audio
    )

    # DEBUG: Overfit experiment - use only 2 samples for training
    train_dataset = torch.utils.data.Subset(train_dataset, [0, 1])

    val_dataset = LipReadingDataset(
        data_dir=args.data_root,
        split='valid',
        label_type=args.label_type,
        load_audio=args.load_audio
    )
    # Optionally, also restrict validation set
    val_dataset = torch.utils.data.Subset(val_dataset, [0, 1])

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    logger.info("Creating model...")
    
    if args.label_type == 'phn':
        model = create_phoneme_model(
            pretrained=args.pretrained,
            freeze_backbone=False,
            use_gradient_checkpointing=args.gradient_checkpointing
        )
    else:
        # For word-level, you'd need to determine vocabulary size
        vocab_size = 1000  # Placeholder - adjust based on your data
        model = create_word_model(
            vocab_size=vocab_size,
            pretrained=args.pretrained,
            freeze_backbone=False,
            use_gradient_checkpointing=args.gradient_checkpointing
        )
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create scheduler
    scheduler = None
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    logger.info("Starting training...")
    
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"{'='*50}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, args)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, device)
        
        # Log validation loss to text file
        log_file = os.path.join(args.output_dir, 'training_log.txt')
        current_lr = optimizer.param_groups[0]['lr'] if scheduler else None
        log_to_file(log_file, epoch, -1, val_loss, val_loss, current_lr, validation=True)
        
        # Step scheduler
        if scheduler:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Learning rate: {current_lr:.6f}")
        
        # Log to tensorboard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        if scheduler:
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                os.path.join(args.output_dir, 'best_model.pth')
            )
            logger.info(f"New best model! Val loss: {val_loss:.4f}")
        
        # Save regular checkpoint
        if epoch % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            )
        
        # Show sample predictions
        if epoch % 2 == 0:  # Every 2 epochs
            logger.info("\n--- Sample Predictions ---")
            test_predictions(model, val_loader, device, num_samples=2)
    
    # Final checkpoint
    save_checkpoint(
        model, optimizer, scheduler, args.epochs, val_loss,
        os.path.join(args.output_dir, 'final_model.pth')
    )
    
    logger.info("Training completed!")
    writer.close()


if __name__ == '__main__':
    main()
