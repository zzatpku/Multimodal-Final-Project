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

template = """Analyze two sentences and determine if they're referring to the same general object or concept, focusing on the type of object, not attributes such as color, size, or shape. Respond with 'True' if they refer to the same thing and 'False' if not. Also, provide a brief rationale (no more than 20 words) for your judgment.
Example:
Input: 1. Spiral staircase that goes from a ground floor. 2. This is a 3D model of wooden stairs in light brown
Output: True#Both refer to a staircase.

Now, analyze the following:
Input: 1. {ground_truth} 2. {model_output}
Output: """

parser = argparse.ArgumentParser(description='generate dpo dataset')

parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--dataset_path', type=str, default=None)
parser.add_argument('--max_length', type=int, default=2048)

args = parser.parse_args()

model_path = args.model_path
tokenizer_path = args.model_path

# dataset = load_dataset('json', data_files=args.dataset_path)['train']
with open(args.dataset_path, 'r') as f:
    dataset = json.load(f)
# dataset = dataset.select(range(100))
dataset = dataset['results']

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
    for item in tqdm(dataset):
        # print(item)
        input = [{
            'role': 'system',
            'content': ""
        }, {
            'role': 'user',
            'content': template.format(ground_truth=item['ground_truth'], model_output=item['model_output'])}]
        input = tokenizer.apply_chat_template(input,
                                              add_generation_prompt=True,
                                              tokenize=False)
        prompts.append(input)
        items.append(item)
    
    final_dataset = []
    raw_outputs = llm.generate(prompts, sampling_params)
    cnt = 0
    total_len = len(raw_outputs)
    for i in range(len(raw_outputs)):
        cur_output = raw_outputs[i].outputs[0].text
        if 'T' in cur_output:
            cnt += 1

    print(f"Total accuracy: {cnt / total_len}")


if __name__ == "__main__":
    main()
