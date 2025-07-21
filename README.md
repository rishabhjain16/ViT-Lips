# ViT-Lips: Vision Transformer for Lip Reading

Simple implementation of Vision Transformer for lip reading with CTC decoding.

## Architecture

- **Model**: ViT-B/16 (86M parameters) + simple temporal processing + CTC head
- **Input**: Videos (224×224 frames) 
- **Output**: Phoneme sequences (40 phonemes + CTC blank)
- **Training**: CTC loss for alignment-free learning

## Structure

```
ViT-Lips/
├── models/
│   └── vit_lipreading.py    # Main ViT model
├── datasets/
│   └── lip_reading_dataset.py # LRS3 data loader
├── src/
│   ├── train.py             # Training script
│   ├── inference.py         # Inference script
│   └── test.py              # Test setup
└── run_training.sh          # Easy training script
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test setup
python src/test.py

# 3. Train model
./run_training.sh --data_root /path/to/LRS3

# 4. Run inference
python src/inference.py --checkpoint outputs/best_model.pth --video video.mp4
```

## Model Details

**ViT-B/16 Backbone**: Pretrained Vision Transformer with 16×16 patches
- Processes each video frame independently
- Outputs 768-dim features per frame

**Temporal Processing**: Simple linear layers for sequence modeling
- Linear(768 → 256) + ReLU + Dropout + LayerNorm

**CTC Head**: Classification layer for phoneme prediction
- Linear(256 → 41) for 40 phonemes + blank token
- Greedy decoding for inference

## Training Options

**Full Fine-tuning** (best accuracy):
```bash
python src/train.py --data_root /path/to/LRS3 --batch_size 4 --lr 1e-4
```

**Feature Extraction** (faster, less memory):
```bash
python src/train.py --data_root /path/to/LRS3 --freeze_backbone --batch_size 8 --lr 1e-3
```