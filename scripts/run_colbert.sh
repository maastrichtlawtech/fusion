#!/bin/bash

TO_PERFORM=$1
if [ "$TO_PERFORM" != "train" ] && [ "$TO_PERFORM" != "test" ]; then
    echo "ERROR: First argument corresponds to the action to perform, and must be either 'train' or 'test'."
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
    CUDA_VISIBLE_DEVICES='0'
else
    CUDA_VISIBLE_DEVICES=''
fi

# Default values for model-related variables.
DIM=128
SIM="cosine"
MASK_PUNCT_IN_DOCS="--mask_punctuation"
ATTEND_MASK_TOKENS_IN_QUERIES="--attend_to_mask_tokens"

# Default values for testing-related variables.
NBITS=2
KMEANS_ITERS=4

# Default values for training-related variables.
TOTAL_STEPS=0
WARMUP_STEPS=0
LR=5e-6
ACC_STEPS=1
NEGS_PER_QUERY=1
USE_INBATCH_NEGS=""
IGNORE_PROVIDED_SCORES_IF_ANY=""
DISTIL_ALPHA=1.0
SEEDS=(42)
SAVE_DATA_DIR="data"
OUT_DIR="output/training"

# Training params.
if [ "$TO_PERFORM" != "train" ]; then
    DO_TEST=""
    DO_TRAIN="--do_train"

    TRAINING_TYPE="v1.5"
    if [[ "$TRAINING_TYPE" == "v1" ]]; then
        IGNORE_PROVIDED_SCORES_IF_ANY="--ignore_scores" #If set, forces CE loss (ColBERTv1) instead of KL-div loss (ColBERTv2) if scores from cross-encoder are provided in the training tuples.

    elif [[ "$TRAINING_TYPE" == "v1.5" ]]; then
        USE_INBATCH_NEGS="--use_ib_negatives"
        IGNORE_PROVIDED_SCORES_IF_ANY="--ignore_scores"

    elif [[ "$TRAINING_TYPE" == "v2" ]]; then
        USE_INBATCH_NEGS="--use_ib_negatives"
        NEGS_PER_QUERY=64
    fi

    # When training a domain-general ColBERT model on mMARCO-fr.
    if [ "$DATASET" = "mmarco-fr" ]; then
        MODEL="almanach/camembert-base"
        QUERY_MAXLEN=32
        DOC_MAXLEN=128
        BATCH_SIZE=128
        TOTAL_STEPS=200000
        WARMUP_STEPS=20000 #If set, will perform linear decay of learning rate after warmup (https://github.com/stanford-futuredata/ColBERT/blob/main/colbert/training/training.py#L63)

    # When training a domain-specific ColBERT model on LLeQA.
    elif [ "$DATASET" = "lleqa" ]; then
        MODEL="antoinelouis/colbertv1-camembert-base-mmarcoFR"
        QUERY_MAXLEN=64
        DOC_MAXLEN=384
        BATCH_SIZE=64
        TOTAL_STEPS=1000
        WARMUP_STEPS=0 #Will use a constant learning rate
        SEEDS=(42 43 44 45 46)
    fi

# Testing params.
else
    DO_TEST="--do_test"
    DO_TRAIN=""
    QUERY_MAXLEN=64
    DOC_MAXLEN=512
    BATCH_SIZE=256

    # Testing the domain-general ColBERT model on mMARCO-fr.
    if [ "$DATASET" = "mmarco-fr" ]; then
        MODEL="antoinelouis/colbertv1-camembert-base-mmarcoFR"
        
    # Testing the domain-general or domain-specific cross-encoder models on LLeQA.
    elif [ "$DATASET" = "lleqa" ]; then
        MODEL="maastrichtlawtech/colbert-legal-french" #in-domain
        #MODEL="antoinelouis/colbertv1-camembert-base-mmarcoFR" #zero-shot
    fi
fi

# Run training or testing.
for SEED in "${SEEDS[@]}"; do
    python src/retrievers/multi_dense_biencoder.py \
        --dataset $DATASET \
        --data_dir $SAVE_DATA_DIR \
        $DO_TRAIN \
        --model_name $MODEL \
        --dim $DIM \
        --similarity $SIM \
        --doc_maxlen $DOC_MAXLEN \
        --query_maxlen $QUERY_MAXLEN \
        $MASK_PUNCT_IN_DOCS \
        $ATTEND_MASK_TOKENS_IN_QUERIES \
        --maxsteps $TOTAL_STEPS \
        --warmup $WARMUP_STEPS \
        --lr $LR \
        --bsize $BATCH_SIZE \
        --accumsteps $ACC_STEPS \
        --nway $(($NEGS_PER_QUERY + 1)) \
        $USE_INBATCH_NEGS \
        $IGNORE_PROVIDED_SCORES_IF_ANY \
        --distillation_alpha $DISTIL_ALPHA \
        --seed $SEED \
        $DO_TEST \
        --nbits $NBITS \
        --kmeans_niters $KMEANS_ITERS \
        --output_dir $OUT_DIR
done