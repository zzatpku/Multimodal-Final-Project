export CUDA_VISIBLE_DEVICES=3
conda activate pointllm
python pointllm/eval/eval_objaverse.py --model_name /ossfs/workspace/nas5/guhao/PointLLM/checkpoints/PointLLM_7B_v1.1 --task_type classification --prompt_index 0