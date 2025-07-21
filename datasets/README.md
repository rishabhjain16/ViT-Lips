# ViT-Lips Dataset Module

This module provides a complete PyTorch dataset implementation for LRS2/LRS3 lip reading datasets, designed to work with Vision Transformers (ViT) for lip reading tasks.

## 📁 Directory Structure

```
datasets/
├── __init__.py                 # Module initialization
├── lip_reading_dataset.py      # Main dataset class
├── utils.py                   # Utility functions
├── config.py                  # Configuration classes
├── test_print.py             # Quick dataset testing
├── data_structure_check.py   # Data validation script
└── README.md                 # This file
```

## 🎯 What This Module Does

The ViT-Lips dataset module loads video frames and audio from LRS2/LRS3 datasets for lip reading tasks:

- **Video Loading**: Extracts frames from MP4 files, resizes to 224×224 for ViT
- **Audio Loading**: Loads WAV files using librosa (16kHz sample rate)
- **Phoneme Labels**: Converts phoneme strings to indices for CTC training
- **Frame Masking**: Random masking for self-supervised learning
- **Multiple Splits**: Supports train/valid/test splits

## 🔧 Key Components

### 1. LipReadingDataset Class (`lip_reading_dataset.py`)

Main PyTorch Dataset class that loads LRS3 data:

```python
from datasets import LipReadingDataset

# Create dataset
dataset = LipReadingDataset(
    data_dir="/path/to/lrs3/433h_data_full_face",
    split="train",                # train/valid/test
    label_type="phn",            # phoneme labels
    image_size=224,              # resize frames to 224x224
    load_audio=True,             # load audio data with librosa
    mask_prob=0.15               # random frame masking (15%)
)

# Get a sample
sample = dataset[0]
print(f"Video: {sample['video'].shape}")      # [C, T, H, W]
print(f"Audio: {sample['audio'].shape}")      # [audio_samples] - Real audio data!
print(f"Label: {sample['label'].shape}")      # [phoneme_indices]
```

**Output format:**
- `video`: Tensor [3, T, 224, 224] - RGB frames normalized to [0,1]
- `audio`: Numpy array [N] - Audio samples at 16kHz (when load_audio=True)
- `label`: Tensor [L] - Phoneme indices (40 phonemes + CTC blank)
- `file_id`: String - Unique identifier
- `audio_path`: String - Path to audio file

### 2. Utility Functions (`utils.py`)

Helper functions for data loading:

- `load_manifest()`: Loads TSV files with video/audio paths
- `load_labels()`: Loads phoneme label files
- `load_phoneme_dict()`: Loads phoneme-to-index mapping
- `load_audio()`: Loads audio using librosa (16kHz, returns numpy array)

### 3. Configuration (`config.py`)

Configuration classes for dataset parameters:

- `DatasetConfig`: Main dataset configuration
- `AudioConfig`: Audio processing settings
- `VideoConfig`: Video processing settings

## 🚀 Quick Start

### Basic Usage

```python
import torch
from datasets import LipReadingDataset

# Create dataset
dataset = LipReadingDataset(
    data_dir="/home/rishabh/Desktop/Datasets/lrs3/433h_data_full_face",
    split="valid",
    load_audio=True
)

# Create DataLoader
dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=8, 
    shuffle=True,
    num_workers=4
)

# Training loop
for batch in dataloader:
    video = batch['video']     # [B, 3, T, 224, 224]
    audio = batch['audio']     # [B, audio_length]
    labels = batch['label']    # [B, label_length]
    
    # Your ViT model here
    # outputs = vit_model(video)
```

### Expected Data Format

Your LRS3 dataset should have this structure:

```
lrs3/433h_data_full_face/
├── train.tsv              # Video/audio paths for training
├── train.phn              # Phoneme labels for training
├── train.wrd              # Sentence labels for training
├── valid.tsv              # Validation data paths
├── valid.phn              # Validation labels
├── valid.wrd              # Sentence labels for validation
├── test.tsv               # Test data paths
├── test.phn               # Test labels
├── test.wrd               # Sentence labels for test
├── dict.phn.txt           # Phoneme dictionary (40 phonemes)
└── dict.wrd.txt           # Word dictionary
```

**TSV Format (5 columns):**
```
file_id	video_path	audio_path	num_frames	audio_samples
trainval/uS7VAFBGWqc/50001	/abs/path/video.mp4	/abs/path/audio.wav	22	14336
```

**Note**: Video/audio files can be located anywhere - TSV contains absolute paths.

## 🛠 Testing Scripts

### 1. `test_print.py` - Quick Dataset Test

**Purpose**: Quickly test if your dataset loads correctly

**Usage**:
```bash
cd /home/rishabh/Desktop/Experiments/ViT-Lips
python datasets/test_print.py
```

**What it shows**:
```
=== TRAIN ===
PHONEME:
  Raw phonemes: W IY R JH AH S T F R EH N D Z
  Converted: tensor([37, 19, 29, 20, 4, 30, 32, 15, 29, 12, 24, 10, 39])
  Video: torch.Size([3, 22, 224, 224])
  Audio: (14336,)  # Real audio samples!

SENTENCE:
  Raw sentence: we're just friends
  Converted: we're just friends
  Audio: (14336,)
```

**When to use**: 
- ✅ First time setup
- ✅ After changing data paths
- ✅ Quick sanity check

### 2. `data_structure_check.py` - Data Validation

**Purpose**: Comprehensive validation of your LRS3 data structure

**Usage**:
```bash
python datasets/data_structure_check.py
```

**What it checks**:
- ✅ TSV files exist and are readable
- ✅ Label files (.phn, .txt) exist
- ✅ Dictionary files (.txt) exist
- ✅ Video/audio paths in TSV are valid
- ✅ No missing or corrupted files

**When to use**:
- ✅ Setting up new dataset
- ✅ Debugging data loading issues
- ✅ Validating data integrity
- ✅ Before long training runs

## 📊 Dataset Statistics

Based on LRS3-433h dataset:

| Split | Samples | Total Duration | Avg Video Length |
|-------|---------|---------------|------------------|
| Train | 299,646 | ~433 hours    | ~5.2 seconds    |
| Valid | 1,200   | ~1.2 hours    | ~3.6 seconds    |
| Test  | 1,321   | ~1.3 hours    | ~3.5 seconds    |

## 🔧 Configuration Options

### Key Parameters

- **`data_dir`**: Path to your LRS3 dataset
- **`split`**: "train", "valid", or "test"
- **`label_type`**: "phn" for phonemes, "wrd" for sentences
- **`image_size`**: Frame size (default: 224 for ViT)
- **`load_audio`**: Whether to load audio data
- **`mask_prob`**: Probability of frame masking (0.0-1.0)

### Advanced Options

- **`max_frames`**: Limit video length
- **`audio_sample_rate`**: Audio sampling rate (default: 16000)
- **`normalize_audio`**: Audio normalization
- **`augmentation`**: Data augmentation settings

## 🐛 Troubleshooting

### Common Issues

**1. Import Error**
```
❌ ImportError: No module named 'datasets'
```
**Solution**: Make sure you're in the project root directory

**2. Data Not Found**
```
❌ Data directory not found
```
**Solution**: Check your `data_dir` path in the scripts

**3. Audio Loading Failed**
```
❌ Audio: Error loading
```
**Solution**: Install librosa: `pip install librosa`

**4. Audio Returns None**
```
Audio: None
```
**Solution**: Set `load_audio=True` in dataset creation

**4. Empty Dataset**
```
⚠️ No samples found!
```
**Solution**: Check TSV and PHN files exist and are not empty

### Debug Steps

1. **Run data structure check**: `python datasets/data_structure_check.py`
2. **Check file permissions**: Ensure video/audio files are readable
3. **Verify TSV format**: Check 5-column format with tab separation
4. **Test single sample**: Use `test_print.py` to load one sample

## 📚 For ViT-Lips Model

This dataset module is specifically designed for:

- **Vision Transformers**: 224×224 frame size
- **CTC Training**: Phoneme-level labels with blank token
- **Multi-modal**: Both video and audio data
- **Self-supervised**: Frame masking for pre-training

Ready for building your ViT-based lip reading model! 🚀

## 📝 Dependencies

- `torch` - PyTorch framework
- `cv2` - OpenCV for video processing
- `numpy` - Numerical operations
- `librosa` - Audio loading (optional)
- `pandas` - TSV file reading
