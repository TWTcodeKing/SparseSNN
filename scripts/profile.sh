#!/bin/bash
# Unified profiling script for TRT engines using nsys and ncu.
#
# Usage:
#   bash scripts/profile.sh nsys <engine_path> [output_name]
#   bash scripts/profile.sh ncu  <engine_path> [output_name]
#
# Examples:
#   bash scripts/profile.sh nsys trt_engines/sew_resnet34_b16/dense_b16.engine
#   bash scripts/profile.sh ncu  trt_engines/sew_resnet34_b16/dense_native_4d_b16.engine native4d
#   bash scripts/profile.sh nsys trt_engines/sew_resnet152_b32/4d_sparse.engine resnet152_4d_sparse

set -e
mkdir -p profiles

TOOL="$1"
ENGINE="$2"
NAME="$3"

if [ -z "$TOOL" ] || [ -z "$ENGINE" ]; then
    echo "Usage: bash scripts/profile.sh <nsys|ncu> <engine_path> [output_name]"
    exit 1
fi

# Auto-generate output name from engine filename if not provided
if [ -z "$NAME" ]; then
    NAME=$(basename "$ENGINE" .engine)
fi

NCU=/opt/nvidia/nsight-compute/2025.1.1/ncu

case "$TOOL" in
    nsys)
        echo "=== nsys profiling: $ENGINE ==="
        nsys profile \
            -o "profiles/${NAME}" \
            --trace=cuda,nvtx \
            --stats=true \
            --force-overwrite=true \
            .venv/bin/python utils/profile_trt_engine.py "$ENGINE" \
                --warmup 10 --iters 10
        echo ""
        echo "=== Done. Report at profiles/${NAME}.nsys-rep ==="
        ;;
    ncu)
        echo "=== ncu profiling: $ENGINE ==="
        $NCU \
            --set full \
            --target-processes all \
            --launch-skip 20 --launch-count 500 \
            -o "profiles/${NAME}" \
            --force-overwrite \
            .venv/bin/python utils/profile_trt_engine.py "$ENGINE" \
                --warmup 20 --iters 2
        echo ""
        echo "=== Done. Report at profiles/${NAME}.ncu-rep ==="
        ;;
    *)
        echo "Unknown tool: $TOOL (use 'nsys' or 'ncu')"
        exit 1
        ;;
esac
