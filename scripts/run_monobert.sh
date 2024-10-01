#!/bin/bash

TO_PERFORM=$1
if [ "$TO_PERFORM" != "train" ] && [ "$TO_PERFORM" != "test" ] && [ "$TO_PERFORM" != "train+test" ]; then
    echo "ERROR: First argument corresponds to the action to perform, and must be either 'train', 'test', or 'train+test'."
    exit 1
fi

DATASET=$2
if [[ ! "$DATASET" =~ ^(lleqa|mmarco)$ ]]; then
    echo "ERROR: Second argument corresponds to the dataset, and must be either 'lleqa' or 'mmarco-fr'."
    exit 1
fi

DEVICE=${3:-cuda}
if [ "$DEVICE" != "cuda" ] && [ "$DEVICE" != "cpu" ]; then
    echo "ERROR: Third argument corresponds to the device to use, and must be either 'cuda' or 'cpu'."
    exit 1
elif [ "$DEVICE" = "cuda" ]; then
    BF16="--use_bf16"
else
    BF16=""
fi
FP16=""

# Default values for training-related variables.
OPTIMIZERS=("AdamW" "Adafactor" "Shampoo")
LR=(2e-5 1e-3 1e-4)
OPTIM_IDX=0
WD=0.01
SCHEDULER="constantlr"
WARMUP_RATIO=0.0
TOTAL_STEPS=0
SEEDS=(42)
LOG_STEPS=1
SAVE_CKPTS_DURING_TRAINING=""
EVAL_DURING_TRAINING=""
OUTPUT="output/training"

# Training params.
if [ "$TO_PERFORM" != "test" ]; then
    DO_TRAIN="--do_train"
    DO_TEST=$([ "$TO_PERFORM" = "train+test" ] && echo "--do_test" || echo "")

    # When training a domain-general cross-encoder model on mMARCO-fr.
    if [ "$DATASET" = "mmarco-fr" ]; then
        MODEL="almanach/camembert-base"
        MAX_SEQ_LEN=256
        BATCH_SIZE=128
        TOTAL_STEPS=20000
        SCHEDULER="linear"
        SAVE_CKPTS_DURING_TRAINING="--save_during_training"

    # When training a domain-specific cross-encoder model on LLeQA.
    elif [ "$DATASET" = "lleqa" ]; then
        MODEL="antoinelouis/crossencoder-camembert-base-mmarcoFR"
        MAX_SEQ_LEN=512
        BATCH_SIZE=64
        TOTAL_STEPS=2000
        SEEDS=(42 43 44 45 46)
        EVAL_DURING_TRAINING="--eval_during_training"
    fi

# Testing params.
else
    DO_TRAIN=""
    DO_TEST="--do_test"
    MAX_SEQ_LEN=512
    BATCH_SIZE=256

    # Testing the domain-general cross-encoder model on mMARCO-fr.
    if [ "$DATASET" = "mmarco-fr" ]; then
        MODEL="antoinelouis/crossencoder-camembert-base-mmarcoFR"
        
    # Testing the domain-general or domain-specific cross-encoder models on LLeQA.
    elif [ "$DATASET" = "lleqa" ]; then
        MODEL="maastrichtlawtech/monobert-legal-french" #in-domain
        #MODEL="antoinelouis/crossencoder-camembert-base-mmarcoFR" #zero-shot
    fi
fi

# Run training or testing.
for SEED in "${SEEDS[@]}"; do
    python src/retrievers/crossencoder.py \
        --dataset "$DATASET" \
        --model_name "$MODEL" \
        --max_seq_length $MAX_SEQ_LEN \
        $DO_TRAIN \
        --max_steps $TOTAL_STEPS \
        --batch_size $BATCH_SIZE \
        --optimizer "${OPTIMIZERS[$OPTIM_IDX]}" \
        --lr ${LR[$OPTIM_IDX]} \
        --wd $WD \
        --scheduler "$SCHEDULER" \
        --warmup_ratio $WARMUP_RATIO \
        $FP16 \
        $BF16 \
        --seed $SEED \
        $SAVE_CKPTS_DURING_TRAINING \
        $EVAL_DURING_TRAINING \
        --log_steps $LOG_STEPS \
        $DO_TEST \
        --device $DEVICE \
        --output_dir $OUTPUT
done