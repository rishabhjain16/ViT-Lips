#!/bin/bash
# Enhanced training script with progress bars and sample-based checkpointing

set -e  # Exit on error

# Default values
DATA_ROOT="/home/rishabh/Desktop/Datasets/lrs3/433h_data_full_face"
OUTPUT_DIR="./outputs"
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4
EPOCHS=10
LR=1e-4
SAVE_EVERY_SAMPLES=5000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --save_every_samples)
            SAVE_EVERY_SAMPLES="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --data_root PATH         Path to LRS3 dataset (default: $DATA_ROOT)"
            echo "  --output_dir PATH        Output directory (default: $OUTPUT_DIR)"
            echo "  --batch_size NUM         Batch size (default: $BATCH_SIZE)"
            echo "  --epochs NUM             Number of epochs (default: $EPOCHS)"
            echo "  --lr FLOAT               Learning rate (default: $LR)"
            echo "  --save_every_samples NUM Save checkpoint every N samples (default: $SAVE_EVERY_SAMPLES)"
            echo "  --help                   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=================================="
echo "ViT-Lips Training with Progress Bars"
echo "=================================="
echo "Data root: $DATA_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LR"
echo "Save every: $SAVE_EVERY_SAMPLES samples"
echo "Memory optimizations: AMP + Gradient Checkpointing"
echo "=================================="

# Check if data exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: Data directory does not exist: $DATA_ROOT"
    echo "Please provide correct path with --data_root"
    exit 1
fi

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run training with enhanced features
python src/train.py \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --save_every_samples "$SAVE_EVERY_SAMPLES" \
    --pretrained \
    --gradient_checkpointing \
    --use_amp \
    --scheduler cosine \
    --log_interval 10 \
    --save_interval 2

echo "Training completed! Check outputs in: $OUTPUT_DIR"

echo "Training completed! Check results in: $OUTPUT_DIR"
