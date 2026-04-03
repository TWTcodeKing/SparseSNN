#!/bin/bash
# ExactOBS-style SBC 2:4 pruning on all CIFAR-100 models.
#
# Method: element-wise greedy OBS with per-row H⁻¹, N:M capacity constraint,
#         im2col SMP Hessian for Conv2d (cross-spatial compensation),
#         columnslast permutation for TRT-compatible 2:4 along C.
#
# Usage: bash scripts/run_sbc_global_cifar100.sh [gpu_id]

set -e

GPU=${1:-0}
DATA=/data/twt/datasets
CALIB=30
BN=64
T_RES=4        # timesteps for SEW-ResNet (IF neuron)
T_MS=6         # timesteps for MS-ResNet (MS neuron)
T_TF=4         # timesteps for transformers

RESULTS=obc_pt/sbc_global_cifar100_results.txt
mkdir -p obc_pt

echo "============================================================" | tee $RESULTS
echo "  ExactOBS SBC 2:4 Pruning — CIFAR-100 Benchmark"            | tee -a $RESULTS
echo "  GPU: $GPU | calib: $CALIB batches | BN: $BN batches"       | tee -a $RESULTS
echo "============================================================" | tee -a $RESULTS
echo "" | tee -a $RESULTS

run_model() {
    local TAG="$1"
    local MODEL="$2"     # --model arg (empty for config-based)
    local CONFIG="$3"    # --config arg (empty for model-based)
    local CKPT="$4"
    local T="$5"
    local IMGSIZE="$6"   # optional override

    local ARGS="--dataset cifar100 --data-root $DATA --T $T --gpu-ids $GPU"
    ARGS="$ARGS --batch-size 64 --nm 2 4"
    ARGS="$ARGS --calib-batches $CALIB --bn-batches $BN --evaluate"

    if [ -n "$MODEL" ]; then
        ARGS="$ARGS --model $MODEL"
    else
        ARGS="$ARGS --config $CONFIG"
    fi
    ARGS="$ARGS --dense-checkpoint $CKPT"

    if [ -n "$IMGSIZE" ]; then
        ARGS="$ARGS --img-size $IMGSIZE"
    fi

    echo "--- $TAG ---" | tee -a $RESULTS
    echo "  Command: python -m sparse.snn_sbc $ARGS" | tee -a $RESULTS

    OUTPUT=$(CUDA_VISIBLE_DEVICES=$GPU python -m sparse.snn_sbc $ARGS 2>&1)

    # Extract metrics
    ACC=$(echo "$OUTPUT" | grep "Evaluation accuracy" | grep -oP '[\d.]+(?=%)')
    LOSS=$(echo "$OUTPUT" | grep "total_loss" | grep -oP 'total_loss: [\d.]+' | grep -oP '[\d.]+')
    ERR=$(echo "$OUTPUT" | grep "mean rel_err" | grep -oP 'rel_err: [\d.]+' | grep -oP '[\d.]+')
    VALID=$(echo "$OUTPUT" | grep "valid:" | grep -oP '\d+/\d+')
    SAVED=$(echo "$OUTPUT" | grep "Saved to" | awk '{print $NF}')

    echo "  Accuracy:  ${ACC}%" | tee -a $RESULTS
    echo "  Loss:      ${LOSS}" | tee -a $RESULTS
    echo "  Rel_err:   ${ERR}" | tee -a $RESULTS
    echo "  2:4 valid: ${VALID}" | tee -a $RESULTS
    echo "  Saved:     ${SAVED}" | tee -a $RESULTS
    echo "" | tee -a $RESULTS
}

# # ---- SEW-ResNets (IFNeuron, T=4) ----
# echo "== SEW-ResNet Family ==" | tee -a $RESULTS

run_model "SEW-ResNet-32" "sew_resnet_cifar32" "" \
    "output/sew_resnet_cifar32_cifar100_bs100_lr0.1/best.pth" $T_RES ""

run_model "SEW-ResNet-56" "sew_resnet_cifar56" "" \
    "output/sew_resnet_cifar56_cifar100_bs100_lr0.1/best.pth" $T_RES ""

run_model "SEW-ResNet-110" "sew_resnet_cifar110" "" \
    "output/sew_resnet_cifar110_cifar100_bs100_lr0.1/best.pth" $T_RES ""

# ---- MS-ResNets (MSNeuron, T=6) ----
echo "== MS-ResNet Family ==" | tee -a $RESULTS

run_model "MS-ResNet-32" "ms_resnet_cifar32" "" \
    "output/ms_resnet_cifar32_cifar100_bs100_lr0.1/best.pth" $T_MS ""

run_model "MS-ResNet-56" "ms_resnet_cifar56" "" \
    "output/ms_resnet_cifar56_cifar100_bs100_lr0.1/best.pth" $T_MS ""

run_model "MS-ResNet-110" "ms_resnet_cifar110" "" \
    "output/ms_resnet_cifar110_cifar100_bs100_lr0.1/best.pth" $T_MS ""

# ---- Transformer Models (T=4) ----
echo "== Transformer Family ==" | tee -a $RESULTS

run_model "MaxFormer" "" "configs/maxformer/maxformer_cifar.yaml" \
    "output/maxformer_cifar_cifar100_bs64_lr0.0015/best.pth" $T_TF ""

run_model "MS-QKFormer" "" "configs/maxformer/ms_qkformer_cifar.yaml" \
    "output/ms_qkformer_cifar_cifar100_bs64_lr0.0015/best.pth" $T_TF ""

run_model "MetaSpikeFormer-384" "" "configs/metaformer/meta_spikformer_8_384_cifar.yaml" \
    "output/meta_spikformer_8_384_cifar_cifar100_bs32_lr0.0001/best.pth" $T_TF ""

# ---- SpikingResformer (T=4, img_size=128) ----
echo "== SpikingResformer Family ==" | tee -a $RESULTS

run_model "SpikingResformer-Ti" "" "configs/spikingresformer/spikingresformer_ti.yaml" \
    "output/transfer_spikingresformer_ti_cifar100_lr0.0001/best.pth" $T_TF 128

run_model "SpikingResformer-S" "" "configs/spikingresformer/spikingresformer_s.yaml" \
    "output/transfer_spikingresformer_s_cifar100_lr0.0001/best.pth" $T_TF 128

# ---- Summary ----
echo "============================================================" | tee -a $RESULTS
echo "  Full results saved to: $RESULTS" | tee -a $RESULTS
echo "============================================================" | tee -a $RESULTS
