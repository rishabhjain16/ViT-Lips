#!/bin/bash
# Simple training script with good defaults

set -e  # Exit on error

# Default values
DATA_ROOT="/home/rishabh/Desktop/Datasets/lrs3/433h_data_full_face"
OUTPUT_DIR="./outputs"
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
EPOCHS=10
LR=1e-4

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
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --data_root PATH    Path to LRS3 dataset (default: $DATA_ROOT)"
            echo "  --output_dir PATH   Output directory (default: $OUTPUT_DIR)"
            echo "  --batch_size NUM    Batch size (default: $BATCH_SIZE)"
            echo "  --epochs NUM        Number of epochs (default: $EPOCHS)"
            echo "  --lr FLOAT          Learning rate (default: $LR)"
            echo "  --help              Show this help message"
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
echo "ViT-Lips Training"
echo "=================================="
echo "Data root: $DATA_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LR"
echo "=================================="

# Check if data exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: Data directory does not exist: $DATA_ROOT"
    echo "Please provide correct path with --data_root"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
python src/train.py \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --pretrained \
    --scheduler cosine \
    --log_interval 10 \
    --save_interval 2

echo "Training completed! Check results in: $OUTPUT_DIR"
