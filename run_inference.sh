#!/bin/bash

# ViT Lip Reading Inference Script
# Usage: ./run_inference.sh [options]

# Default parameters
DATA_ROOT="${DATA_ROOT:-./datasets/PHOENIX2014T}"
CHECKPOINT_PATH=""
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/inference}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DEVICE="${DEVICE:-cuda}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_PATH="$2"
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
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help|-h)
            echo "ViT Lip Reading Inference Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --data_root DIR         Path to dataset directory (default: ./datasets/PHOENIX2014T)"
            echo "  --checkpoint FILE       Path to model checkpoint (required)"
            echo "  --output_dir DIR        Output directory for results (default: ./outputs/inference)"
            echo "  --batch_size N          Batch size for inference (default: 4)"
            echo "  --device DEVICE         Device to use (cuda/cpu) (default: cuda)"
            echo "  --help, -h              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --checkpoint ./outputs/checkpoints/model_epoch_5.pth"
            echo "  $0 --checkpoint ./outputs/checkpoints/model_sample_50000.pth --batch_size 8"
            echo "  $0 --checkpoint ./outputs/checkpoints/best_model.pth --device cpu"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$CHECKPOINT_PATH" ]]; then
    echo "Error: --checkpoint parameter is required!"
    echo "Use --help for usage information."
    exit 1
fi

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT_PATH"
    exit 1
fi

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "Error: Data directory not found: $DATA_ROOT"
    exit 1
fi

# Display configuration
echo "=== ViT Lip Reading Inference Configuration ==="
echo "Data Root:      $DATA_ROOT"
echo "Checkpoint:     $CHECKPOINT_PATH"
echo "Output Dir:     $OUTPUT_DIR"
echo "Batch Size:     $BATCH_SIZE"
echo "Device:         $DEVICE"
echo "================================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run inference
python src/inference.py \
    --data_root "$DATA_ROOT" \
    --checkpoint "$CHECKPOINT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE"

if [[ $? -eq 0 ]]; then
    echo "Inference completed successfully! Results saved in: $OUTPUT_DIR"
else
    echo "Inference failed! Check the error messages above."
    exit 1
fi
