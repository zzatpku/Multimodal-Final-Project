#!/bin/bash
export HF_HOME="/ossfs/workspace/nas5/guhao/Data/cache"

conda activate handbook

python pointllm/data/filter_dataset.py \
--model_path /ossfs/workspace/nas5/guhao/Meta-Llama-3-8B-Instruct \
--dataset_path /ossfs/workspace/nas5/guhao/PointLLM/checkpoints/PointLLM_7B_v1.2/evaluation/combined_data/train_data.json \
--save_path /ossfs/workspace/nas5/guhao/PointLLM/checkpoints/PointLLM_7B_v1.2/evaluation/filtered_combined_data