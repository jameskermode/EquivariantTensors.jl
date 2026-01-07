#!/bin/bash
#=============================================================================
# IREE Compilation Script for ACE Models
#=============================================================================
#
# Compiles StableHLO MLIR to IREE VMFB format for CPU and GPU execution.
#
# Usage:
#   ./compile.sh <input.mlir> [output_dir]
#
# Requirements:
#   - IREE tools installed at ~/iree/bin/
#   - Valid StableHLO MLIR file
#
# Output:
#   - ace_model_cpu.vmfb   (CPU execution)
#   - ace_model_cuda.vmfb  (CUDA GPU)
#
#=============================================================================

set -e

IREE_DIR="${IREE_DIR:-$HOME/iree/bin}"
MLIR_FILE="${1:-../stablehlo_export/ace_energy.mlir}"
OUTPUT_DIR="${2:-./compiled}"

# Check IREE tools exist
if [ ! -f "$IREE_DIR/iree-compile" ]; then
    echo "ERROR: iree-compile not found at $IREE_DIR"
    echo "Set IREE_DIR environment variable to your IREE installation"
    exit 1
fi

# Check input file exists
if [ ! -f "$MLIR_FILE" ]; then
    echo "ERROR: Input MLIR file not found: $MLIR_FILE"
    echo "Run ace_export.jl first to generate the StableHLO file"
    exit 1
fi

echo "=========================================="
echo "IREE Compilation"
echo "=========================================="
echo "Input:  $MLIR_FILE"
echo "Output: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

#-----------------------------------------------------------------------------
# CPU Compilation
#-----------------------------------------------------------------------------
echo "Compiling for CPU (llvm-cpu)..."

"$IREE_DIR/iree-compile" \
    --iree-input-type=stablehlo \
    --iree-hal-target-backends=llvm-cpu \
    --iree-llvmcpu-target-cpu=host \
    "$MLIR_FILE" \
    -o "$OUTPUT_DIR/ace_model_cpu.vmfb"

if [ $? -eq 0 ]; then
    echo "[OK] CPU compilation successful"
    ls -lh "$OUTPUT_DIR/ace_model_cpu.vmfb"
else
    echo "[FAIL] CPU compilation failed"
fi

#-----------------------------------------------------------------------------
# CUDA GPU Compilation (optional)
#-----------------------------------------------------------------------------
echo ""
echo "Compiling for CUDA GPU..."

# Check if CUDA backend is available
if "$IREE_DIR/iree-compile" --help 2>&1 | grep -q "cuda"; then
    "$IREE_DIR/iree-compile" \
        --iree-input-type=stablehlo \
        --iree-hal-target-backends=cuda \
        --iree-cuda-target=sm_70 \
        "$MLIR_FILE" \
        -o "$OUTPUT_DIR/ace_model_cuda.vmfb" 2>/dev/null

    if [ $? -eq 0 ]; then
        echo "[OK] CUDA compilation successful"
        ls -lh "$OUTPUT_DIR/ace_model_cuda.vmfb"
    else
        echo "[SKIP] CUDA compilation failed (may not be available)"
    fi
else
    echo "[SKIP] CUDA backend not available"
fi

#-----------------------------------------------------------------------------
# Summary
#-----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Compilation Complete"
echo "=========================================="
echo ""
echo "Generated files:"
ls -la "$OUTPUT_DIR"/*.vmfb 2>/dev/null || echo "(no vmfb files found)"
echo ""
echo "To test CPU execution:"
echo "  $IREE_DIR/iree-run-module \\"
echo "      --device=local-task \\"
echo "      --module=$OUTPUT_DIR/ace_model_cpu.vmfb \\"
echo "      --function=main"
echo ""
echo "To benchmark:"
echo "  $IREE_DIR/iree-benchmark-module \\"
echo "      --device=local-task \\"
echo "      --module=$OUTPUT_DIR/ace_model_cpu.vmfb \\"
echo "      --function=main"
