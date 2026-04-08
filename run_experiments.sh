#!/bin/bash
set -e

CONDA_ENV="researchclaw"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "Running all deepfake detection experiments"
echo "Project: $PROJECT_DIR"
echo "Conda env: $CONDA_ENV"
echo "============================================"

for MODE in rgb freq dual dual_cl; do
    echo ""
    echo "============================================"
    echo "Experiment: $MODE"
    echo "============================================"

    conda run -n $CONDA_ENV python "$PROJECT_DIR/train.py" \
        --mode $MODE \
        --batch_size 64 \
        --epochs 30 \
        --lr 1e-4 \
        --patience 5 \
        --seed 42 \
        --output_dir "$PROJECT_DIR/outputs"

    CKPT="$PROJECT_DIR/outputs/M4_${MODE}/best_model.pth"
    if [ -f "$CKPT" ]; then
        conda run -n $CONDA_ENV python "$PROJECT_DIR/evaluate.py" \
            --mode $MODE \
            --checkpoint "$CKPT" \
            --batch_size 64 \
            --output_dir "$PROJECT_DIR/outputs"
    else
        echo "WARNING: No checkpoint found at $CKPT"
    fi
done

echo ""
echo "============================================"
echo "All experiments complete!"
echo "============================================"
