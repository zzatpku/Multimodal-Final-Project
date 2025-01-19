import json
import argparse
import os
import random

datasets = [
    "/ossfs/workspace/nas5/guhao/PointLLM/checkpoints/PointLLM_7B_v1.2/evaluation/split1.json",
    "/ossfs/workspace/nas5/guhao/PointLLM/checkpoints/PointLLM_7B_v1.2/evaluation/split2.json"
]

save_path = "/ossfs/workspace/nas5/guhao/PointLLM/checkpoints/PointLLM_7B_v1.2/evaluation/combined_data.json"

combined_data = []
for data_path in datasets:
    split_path = data_path
    with open(split_path, 'r') as f:
        data = json.load(f)
    combined_data.extend(data)

random.shuffle(combined_data)

os.makedirs(save_path, exist_ok=True)
save_path = f'{save_path}/train_data.json'
with open(save_path, 'w') as outfile:
    json.dump(combined_data, outfile, indent=4)