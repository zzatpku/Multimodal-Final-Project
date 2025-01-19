#!/bin/bash
export HF_HOME="/ossfs/workspace/nas5/guhao/Data/cache"

conda activate handbook

python pointllm/eval/llama_eval.py \
--model_path /ossfs/workspace/nas5/guhao/Meta-Llama-3-8B-Instruct \
--dataset_path /ossfs/workspace/nas5/guhao/PointLLM/outputs/PointLLM_train_stage2_own_data/PointLLM_train_stage2_own_data/evaluation/PointLLM_brief_description_val_200_GT_Objaverse_classification_prompt0.json

# --dataset_path /ossfs/workspace/nas5/guhao/PointLLM/checkpoints/PointLLM_7B_v1.1/evaluation/PointLLM_brief_description_val_200_GT_Objaverse_classification_prompt0.json
