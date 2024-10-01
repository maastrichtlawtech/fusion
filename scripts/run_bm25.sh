#!/bin/bash

TO_PERFORM=$1
if [ "$TO_PERFORM" != "tuning" ] && [ "$TO_PERFORM" != "testing" ]; then
    echo "ERROR: First argument corresponds to the action to perform, and must be either 'tuning' or 'testing'."
    exit 1
fi

DATASET=$2
if [[ ! "$DATASET" =~ ^(lleqa|mmarco)$ ]]; then
    echo "ERROR: Second argument corresponds to the dataset, and must be either 'lleqa' or 'mmarco'."
    exit 1
fi

if [ "$TO_PERFORM" == 'tuning' ]; then
    python src/retrievers/bm25.py \
        --dataset "$DATASET" \
        --do_preprocessing \
        --do_hyperparameter_tuning \
        --output_dir "output/tuning"
else
    if [ "$DATASET" == "lleqa" ]; then
        K1=2.5
        B=0.2
    else
        K1=0.9
        B=0.4
    fi
    python src/retrievers/bm25.py \
        --dataset "$DATASET" \
        --do_evaluation \
        --do_preprocessing \
        --k1 $K1 \
        --b $B \
        --output_dir "output/testing"
fi