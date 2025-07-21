#!/usr/bin/env python3
"""
Test script to verify the model and training setup
"""

import os
import sys
import logging

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_model_creation():
    """Test model creation"""
    try:
        import torch
        from models.vit_lipreading import create_phoneme_model, create_word_model
        
        logger.info("Testing model creation...")
        
        # Test phoneme model
        phoneme_model = create_phoneme_model(pretrained=False)  # No pretrained for test
        logger.info(f"Phoneme model created - Parameters: {sum(p.numel() for p in phoneme_model.parameters()):,}")
        
        # Test word model
        word_model = create_word_model(vocab_size=100, pretrained=False)
        logger.info(f"Word model created - Parameters: {sum(p.numel() for p in word_model.parameters()):,}")
        
        # Test forward pass
        batch_size = 2
        video = torch.randn(batch_size, 3, 25, 224, 224)
        
        with torch.no_grad():
            phoneme_output = phoneme_model(video)
            word_output = word_model(video)
        
        logger.info(f"Phoneme model output shape: {phoneme_output.shape}")
        logger.info(f"Word model output shape: {word_output.shape}")
        
        logger.info("✅ Model creation test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model creation test failed: {str(e)}")
        return False


def test_dataset_loading():
    """Test dataset loading"""
    try:
        from datasets.lip_reading_dataset import LipReadingDataset
        
        logger.info("Testing dataset loading...")
        
        # This will only work if you have the dataset
        data_root = "/home/rishabh/Desktop/Datasets/LRS3"
        
        if not os.path.exists(data_root):
            logger.warning(f"Dataset not found at {data_root}, skipping dataset test")
            return True
        
        # Test dataset creation
        dataset = LipReadingDataset(
            data_root=data_root,
            split='train',
            label_type='phn',
            load_audio=False
        )
        
        logger.info(f"Dataset loaded - {len(dataset)} samples")
        
        # Test loading a sample
        sample = dataset[0]
        video, label, video_path = sample
        
        logger.info(f"Sample video shape: {video.shape}")
        logger.info(f"Sample label length: {len(label)}")
        logger.info(f"Sample video path: {os.path.basename(video_path)}")
        
        logger.info("✅ Dataset loading test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset loading test failed: {str(e)}")
        return False


def test_training_components():
    """Test training components"""
    try:
        import torch
        from models.vit_lipreading import create_phoneme_model
        
        logger.info("Testing training components...")
        
        # Create model
        model = create_phoneme_model(pretrained=False)
        
        # Test CTC loss computation
        batch_size = 2
        seq_length = 25
        num_classes = 41
        
        # Create dummy data
        videos = torch.randn(batch_size, 3, seq_length, 224, 224)
        targets = torch.randint(1, num_classes, (10,))  # Random target sequence
        target_lengths = torch.tensor([5, 5])  # Two sequences of length 5 each
        
        # Test forward pass
        logits = model(videos)
        logger.info(f"Model output shape: {logits.shape}")
        
        # Test CTC loss
        loss = model.compute_ctc_loss(videos, targets, target_lengths)
        logger.info(f"CTC loss: {loss.item():.4f}")
        
        # Test decoding
        predictions = model.decode_greedy(videos)
        logger.info(f"Decoded predictions: {predictions}")
        
        logger.info("✅ Training components test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Training components test failed: {str(e)}")
        return False


def main():
    logger.info("="*50)
    logger.info("Running ViT-Lips Model Tests")
    logger.info("="*50)
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Dataset Loading", test_dataset_loading),
        ("Training Components", test_training_components),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
    
    logger.info("\n" + "="*50)
    logger.info(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Your setup is ready.")
    else:
        logger.warning("⚠️  Some tests failed. Check the errors above.")
    
    logger.info("="*50)


if __name__ == '__main__':
    main()
