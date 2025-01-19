from datasets import load_dataset
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Manager
import os
import re
import argparse
import json
import random
from vllm import LLM, SamplingParams

parser = argparse.ArgumentParser(description='generate dpo dataset')

parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--dataset_path', type=str, default=None)
parser.add_argument('--save_path', type=str, default=None)
parser.add_argument('--max_length', type=int, default=2048)

args = parser.parse_args()

model_path = args.model_path
tokenizer_path = args.model_path

dataset = load_dataset('json', data_files=args.dataset_path)['train']
# dataset = dataset.select(range(100))

def main():
    llm = LLM(model=model_path, tokenizer=tokenizer_path)

    sampling_params = SamplingParams(temperature=0.1,
                                     max_tokens=args.max_length)

    # primitive_prompts is a list of the original prompts
    # prompts is a list of the tokenized prompts
    # Responses is a list of dictionary. When both chosen and rejected are contained, we will put it into the dataset.

    prompts = []
    items = []

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    backspace = '\n'
    for item in tqdm(dataset):
        input = [{
            'role': 'system',
            'content': ""
        }, {
            'role': 'user',
            'content': f"Please judge whether the following response is meaningful to the prompt or not. If it is meaningful, please answer 'Meaningful'. If it is not meaningful, please answer 'Meaningless'.{backspace}{backspace}Prompt: {item['conversations'][0]['value']}{backspace}Response: {item['conversations'][1]['value']}{backspace}"}]
        input = tokenizer.apply_chat_template(input,
                                              add_generation_prompt=True,
                                              tokenize=False)
        prompts.append(input)
        items.append(item)
    
    final_dataset = []
    raw_outputs = llm.generate(prompts, sampling_params)
    for i in range(len(raw_outputs)):
        cur_output = raw_outputs[i].outputs[0].text
        if 'Meaningful' in cur_output:
            final_dataset.append(items[i])

    print(f"Before filter: {len(raw_outputs)}")
    print(f"After filter: {len(final_dataset)}")

    save_dir = args.save_path
    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/train_data.json'

    with open(save_path, 'w') as f:
        json.dump(final_dataset, f, indent=4)


if __name__ == "__main__":
    main()
