import json
import os

with open('/ossfs/workspace/nas5/guhao/PointLLM/checkpoints/PointLLM_7B_v1.2/evaluation/filtered_combined_data/train_data.json', 'r') as f:
    dataset = json.load(f)

for i in range(10):
    dict = dataset[i]
    print(dict)
    print("*" * 120)
    print(dict.keys())
    print("*"*30 + "prompt" + "*"*30)
    print(dict['conversations'][0]['value'])
    print()
    print("*"*30 + "ground_truth" + "*"*30)
    print(dict['ground_truth'])
    print()
    print("*"*30 + "model_output" + "*"*30)
    print(dict['model_output'])