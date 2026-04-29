#!/bin/bash
# SBC 2:4 pruning pipeline for all models.
#
# Architecture-aware settings:
#   - ResNets: channel permutation + 1-epoch KD fine-tuning (perm helps 3x3 Conv)
#   - Transformers (MaxFormer, QKFormer, SpikingResformer): SBC only (perm/ft hurt 1x1 Conv)
#   - MetaSpikeFormer: SBC only, exclude body.2.1/pwconv2 (RepConv after depthwise: ~50% rel_err)
#
# Usage: bash scripts/finetune_all.sh [gpu_id]

set -e

GPU=${1:-0}
DATA=/data/twt/datasets
CALIB=128
BN=64
OUTDIR=obc_pt_ft
RESULTS=$OUTDIR/results.txt

mkdir -p $OUTDIR

echo "============================================================" | tee $RESULTS
echo "  SBC 2:4 Pruning Pipeline (architecture-aware)"             | tee -a $RESULTS
echo "  GPU: $GPU | calib: $CALIB | BN: $BN | output: $OUTDIR"    | tee -a $RESULTS
echo "============================================================" | tee -a $RESULTS
echo "" | tee -a $RESULTS

run() {
    local TAG="$1"
    local MODEL="$2"     # --model (empty for config-based)
    local CONFIG="$3"    # --config (empty for model-based)
    local CKPT="$4"      # dense checkpoint
    local DATASET="$5"
    local T="$6"
    local BS="$7"
    local ARCH="$8"      # resnet | transformer | metaspikformer
    local EXTRA="$9"     # extra args (e.g. --img-size 224)

    local ARGS="--dataset $DATASET --data-root $DATA --T $T --gpu-ids $GPU"
    ARGS="$ARGS --batch-size $BS --nm 2 4"
    ARGS="$ARGS --calib-batches $CALIB --bn-batches $BN"
    ARGS="$ARGS --evaluate"

    # Architecture-specific settings
    local SUFFIX=""
    case "$ARCH" in
        resnet)
            ARGS="$ARGS --permute-channels --finetune 1"
            SUFFIX="_perm_ft1"
            ;;
        transformer)
            # No perm, no ft — baseline SBC only (perm/ft hurt 1x1-heavy models)
            SUFFIX=""
            ;;
        metaspikformer)
            # Exclude body.2.1 (1x1 after depthwise in RepConv: ~50% rel_err)
            # and pwconv2 (SepConv second pointwise: ~30% rel_err)
            ARGS="$ARGS --exclude head fc classifier body.2.1 pwconv2"
            SUFFIX="_excl"
            ;;
    esac

    if [ -n "$MODEL" ]; then
        ARGS="$ARGS --model $MODEL"
    else
        ARGS="$ARGS --config $CONFIG"
    fi
    ARGS="$ARGS --dense-checkpoint $CKPT"
    if [ -n "$EXTRA" ]; then
        ARGS="$ARGS $EXTRA"
    fi

    # Derive output filename
    local MTAG
    if [ -n "$MODEL" ]; then
        MTAG="${MODEL}_${DATASET}"
    else
        MTAG="$(basename ${CONFIG%.yaml})_${DATASET}"
    fi
    local OUTPATH="$OUTDIR/${MTAG}_sbc_2_4${SUFFIX}.pth"
    ARGS="$ARGS --output $OUTPATH"

    echo "--- $TAG [$ARCH] ---" | tee -a $RESULTS
    echo "  $ARGS" | tee -a $RESULTS

    OUTPUT=$(python -m sparse.snn_sbc $ARGS 2>&1)

    ACC=$(echo "$OUTPUT" | grep "Evaluation accuracy" | grep -oP '[\d.]+(?=%)')
    LOSS=$(echo "$OUTPUT" | grep "total_loss" | grep -oP 'total_loss: [\d.]+' | grep -oP '[\d.]+')
    ERR=$(echo "$OUTPUT" | grep "mean rel_err" | grep -oP 'rel_err: [\d.]+' | grep -oP '[\d.]+')
    FTACC=$(echo "$OUTPUT" | grep "train_acc=" | tail -1 | grep -oP 'train_acc=[\d.]+' | grep -oP '[\d.]+')

    echo "  Accuracy:   ${ACC}%" | tee -a $RESULTS
    echo "  OBS loss:   ${LOSS}" | tee -a $RESULTS
    echo "  Rel_err:    ${ERR}" | tee -a $RESULTS
    [ -n "$FTACC" ] && echo "  FT train:   ${FTACC}%" | tee -a $RESULTS
    echo "  Saved:      ${OUTPATH}" | tee -a $RESULTS
    echo "" | tee -a $RESULTS
}

# ============================================================
# CIFAR-100 models
# ============================================================
echo "== CIFAR-100 ==" | tee -a $RESULTS

# SEW-ResNets (IF neuron, T=4) — perm + ft
run "SEW-ResNet-32"  "sew_resnet_cifar32"  "" \
    "output/sew_resnet_cifar32_cifar100_bs100_lr0.1/best.pth" \
    cifar100 4 64 resnet

run "SEW-ResNet-56"  "sew_resnet_cifar56"  "" \
    "output/sew_resnet_cifar56_cifar100_bs100_lr0.1/best.pth" \
    cifar100 4 64 resnet

run "SEW-ResNet-110" "sew_resnet_cifar110" "" \
    "output/sew_resnet_cifar110_cifar100_bs100_lr0.1/best.pth" \
    cifar100 4 64 resnet

# MS-ResNets (MS neuron, T=6) — perm + ft
run "MS-ResNet-32"   "ms_resnet_cifar32"   "" \
    "output/ms_resnet_cifar32_cifar100_bs100_lr0.1/best.pth" \
    cifar100 6 64 resnet

run "MS-ResNet-56"   "ms_resnet_cifar56"   "" \
    "output/ms_resnet_cifar56_cifar100_bs100_lr0.1/best.pth" \
    cifar100 6 64 resnet

run "MS-ResNet-110"  "ms_resnet_cifar110"  "" \
    "output/ms_resnet_cifar110_cifar100_bs100_lr0.1/best.pth" \
    cifar100 6 64 resnet

# Transformers — SBC only (no perm, no ft)
run "MaxFormer-CIFAR" "" "configs/maxformer/maxformer_cifar.yaml" \
    "output/maxformer_cifar_cifar100_bs64_lr0.0015/best.pth" \
    cifar100 4 64 transformer

run "MS-QKFormer-CIFAR" "" "configs/maxformer/ms_qkformer_cifar.yaml" \
    "output/ms_qkformer_cifar_cifar100_bs64_lr0.0015/best.pth" \
    cifar100 4 64 transformer

# MetaSpikeFormer — exclude body.2.1 + pwconv2
run "MetaSpikeFormer-384-CIFAR" "" "configs/metaformer/meta_spikformer_8_384_cifar.yaml" \
    "output/meta_spikformer_8_384_cifar_cifar100_bs32_lr0.0001/best.pth" \
    cifar100 4 32 metaspikformer

# SpikingResformer — SBC only (T=4, img_size=128)
run "SpikingResformer-Ti-CIFAR" "" "configs/spikingresformer/spikingresformer_ti.yaml" \
    "output/transfer_spikingresformer_ti_cifar100_lr0.0001/best.pth" \
    cifar100 4 64 transformer "--img-size 128"

run "SpikingResformer-S-CIFAR" "" "configs/spikingresformer/spikingresformer_s.yaml" \
    "output/transfer_spikingresformer_s_cifar100_lr0.0001/best.pth" \
    cifar100 4 64 transformer "--img-size 128"

# ============================================================
# DVS datasets
# ============================================================
echo "== DVS Datasets ==" | tee -a $RESULTS

# MS-ResNet DVS (T=16) — perm + ft
run "MS-ResNet-DVS20-CIFAR10DVS" "ms_resnet_dvs20" "" \
    "output/ms_resnet_dvs20_cifar10dvs_bs16_lr0.001/best.pth" \
    cifar10dvs 16 16 resnet "--frames-number 16"

run "MS-ResNet-DVS20-Gesture" "ms_resnet_dvs20" "" \
    "output/ms_resnet_dvs20_dvs128gesture_bs16_lr0.001/best.pth" \
    dvs128gesture 16 16 resnet "--frames-number 16"

# MaxFormer DVS — SBC only
run "MaxFormer-CIFAR10DVS" "" "configs/maxformer/maxformer_dvs.yaml" \
    "output/maxformer_dvs_cifar10dvs_bs16_lr0.005/best.pth" \
    cifar10dvs 16 16 transformer "--frames-number 16"

run "MaxFormer-Gesture" "" "configs/maxformer/maxformer_dvs.yaml" \
    "output/maxformer_dvs_dvs128gesture_bs16_lr0.005/best.pth" \
    dvs128gesture 16 16 transformer "--frames-number 16"

# MS-QKFormer DVS — SBC only
run "MS-QKFormer-CIFAR10DVS" "" "configs/maxformer/ms_qkformer_dvs.yaml" \
    "output/ms_qkformer_dvs_cifar10dvs_bs16_lr0.005/best.pth" \
    cifar10dvs 16 16 transformer "--frames-number 16"

run "MS-QKFormer-Gesture" "" "configs/maxformer/ms_qkformer_dvs.yaml" \
    "output/ms_qkformer_dvs_dvs128gesture_bs16_lr0.005/best.pth" \
    dvs128gesture 16 16 transformer "--frames-number 16"

# SpikingResformer DVS — SBC only
run "SpikingResformer-Ti-CIFAR10DVS" "" "configs/spikingresformer/spikingresformer_ti.yaml" \
    "output/transfer_spikingresformer_ti_cifar10dvs_lr0.0001/best.pth" \
    cifar10dvs 10 16 transformer "--img-size 128 --frames-number 10"

run "SpikingResformer-Ti-Gesture" "" "configs/spikingresformer/spikingresformer_ti.yaml" \
    "output/transfer_spikingresformer_ti_dvs128gesture_lr0.0001/best.pth" \
    dvs128gesture 16 16 transformer "--img-size 128 --frames-number 16"

run "SpikingResformer-S-CIFAR10DVS" "" "configs/spikingresformer/spikingresformer_s.yaml" \
    "output/transfer_spikingresformer_s_cifar10dvs_lr0.0001/best.pth" \
    cifar10dvs 10 16 transformer "--img-size 128 --frames-number 10"

run "SpikingResformer-S-Gesture" "" "configs/spikingresformer/spikingresformer_s.yaml" \
    "output/transfer_spikingresformer_s_dvs128gesture_lr0.0001/best.pth" \
    dvs128gesture 16 16 transformer "--img-size 128 --frames-number 16"

# ============================================================
# ImageNet models
# ============================================================
echo "== ImageNet ==" | tee -a $RESULTS

# SEW-ResNets (T=4) — perm + ft
run "SEW-ResNet-18"  "sew_resnet18"  "" \
    "checkpoints/sewresnet/imagenet/sew18_checkpoint_319.pth" \
    imagenet 4 32 resnet "--img-size 224"

run "SEW-ResNet-34"  "sew_resnet34"  "" \
    "checkpoints/sewresnet/imagenet/sew34_checkpoint_319.pth" \
    imagenet 4 32 resnet "--img-size 224"

run "SEW-ResNet-50"  "sew_resnet50"  "" \
    "checkpoints/sewresnet/imagenet/sew50_checkpoint_319.pth" \
    imagenet 4 32 resnet "--img-size 224"

run "SEW-ResNet-101" "sew_resnet101" "" \
    "checkpoints/sewresnet/imagenet/sew101_checkpoint_319.pth" \
    imagenet 4 16 resnet "--img-size 224"

run "SEW-ResNet-152" "sew_resnet152" "" \
    "checkpoints/sewresnet/imagenet/sew152_checkpoint_319.pth" \
    imagenet 4 16 resnet "--img-size 224"

# MS-ResNets (T=6) — perm + ft
run "MS-ResNet-18"   "ms_resnet18"   "" \
    "checkpoints/msresnet/imagenet/resnet18.pth" \
    imagenet 6 32 resnet "--img-size 224"

run "MS-ResNet-34"   "ms_resnet34"   "" \
    "checkpoints/msresnet/imagenet/resnet34.pth" \
    imagenet 6 32 resnet "--img-size 224"

run "MS-ResNet-104"  "ms_resnet104"  "" \
    "checkpoints/msresnet/imagenet/resnet104.pth" \
    imagenet 6 8 resnet "--img-size 224"

# Transformers - ImageNet — SBC only
run "MaxFormer-10-384" "" "configs/maxformer/maxformer_10_384.yaml" \
    "checkpoints/maxformer/imagenet/maxformer_10_384_T4.pth" \
    imagenet 4 32 transformer "--img-size 224"

run "MaxFormer-10-512" "" "configs/maxformer/maxformer_10_512.yaml" \
    "checkpoints/maxformer/imagenet/maxformer_10_512_T4.pth" \
    imagenet 4 32 transformer "--img-size 224"

run "MaxFormer-10-768" "" "configs/maxformer/maxformer_10_768.yaml" \
    "checkpoints/maxformer/imagenet/maxformer_10_768_T4.pth" \
    imagenet 4 8 transformer "--img-size 224"

run "MS-QKFormer-10-384" "" "configs/maxformer/ms_qkformer_10_384.yaml" \
    "checkpoints/maxformer/imagenet/ms_qk_384_T4.pth" \
    imagenet 4 8 transformer "--img-size 224"

# SpikingResformer - ImageNet — SBC only
run "SpikingResformer-Ti-ImageNet" "" "configs/spikingresformer/spikingresformer_ti.yaml" \
    "checkpoints/spikingresformer/ImageNet_spikingresformer_ti.pth" \
    imagenet 4 8 transformer "--img-size 224"

run "SpikingResformer-S-ImageNet" "" "configs/spikingresformer/spikingresformer_s.yaml" \
    "checkpoints/spikingresformer/ImageNet_spikingresformer_s.pth" \
    imagenet 6 8 transformer "--img-size 224"

run "SpikingResformer-M-ImageNet" "" "configs/spikingresformer/spikingresformer_m.yaml" \
    "checkpoints/spikingresformer/ImageNet_spikingresformer_m.pth" \
    imagenet 4 8 transformer "--img-size 224"

run "SpikingResformer-L-ImageNet" "" "configs/spikingresformer/spikingresformer_l.yaml" \
    "checkpoints/spikingresformer/ImageNet_spikingresformer_l.pth" \
    imagenet 6 8 transformer "--img-size 224"

# ============================================================
echo "============================================================" | tee -a $RESULTS
echo "  All results saved to: $RESULTS"                            | tee -a $RESULTS
echo "============================================================" | tee -a $RESULTS
