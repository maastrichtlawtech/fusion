#!/bin/bash

LLEQA_SPLIT=$1
if [ "$LLEQA_SPLIT" != "test" ] && [ "$LLEQA_SPLIT" != "dev" ]; then
    echo "ERROR: First argument corresponds to the LLeQA data split, and must be either 'test' or 'dev'."
    exit 1
fi

TRAINING_DOMAIN=$2
if [ "$TRAINING_DOMAIN" != "general" ] && [ "$TRAINING_DOMAIN" != "legal" ]; then
    echo "ERROR: Second argument corresponds to the training domain of the neural retrievers, and must be either 'general' or 'legal'."
    exit 1
fi

EXPERIMENT_NAME=$3
if [ -z "$EXPERIMENT_NAME" ] || [[ "$EXPERIMENT_NAME" != "--tune_linear_fusion_weight" && "$EXPERIMENT_NAME" != "--analyze_score_distributions" ]]; then
    echo "ERROR: Second argument corresponds to the experiment name, and must be either empty or one of '--tune_linear_fusion_weight' '--analyze_score_distributions'."
    exit 1
fi

RETRIEVER_COMBINATIONS=(
    "--run_bm25 --run_splade"
    "--run_bm25 --run_dpr"
    "--run_bm25 --run_colbert"
    "--run_splade --run_dpr"
    "--run_splade --run_colbert"
    "--run_dpr --run_colbert"
    "--run_bm25 --run_splade --run_dpr"
    "--run_bm25 --run_splade --run_colbert"
    "--run_bm25 --run_dpr --run_colbert"
    "--run_splade --run_dpr --run_colbert"
    "--run_bm25 --run_splade --run_dpr --run_colbert"
)
FUSIONERS=("nsf" "bcf" "rrf")
NORMALIZERS=("min-max" "z-score" "percentile-rank")

for R in "${RETRIEVER_COMBINATIONS[@]}"; do
    for F in "${FUSIONERS[@]}"; do
        if [ "$F" != "nsf" ]; then
            NORMALIZERS=("none")
        fi
        for N in "${NORMALIZERS[@]}"; do
            python src/retrievers/hybrid.py \
                --data_split $LLEQA_SPLIT \
                --models_domain $TRAINING_DOMAIN \
                $R \
                --fusion $F \
                --normalization $N \
                $EXPERIMENT_NAME \
                --output_dir "output/testing"
        done
    done
done