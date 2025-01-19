import argparse
import torch
from torch.utils.data import DataLoader, Subset
import os
from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.utils import disable_torch_init
from pointllm.model import *
from pointllm.model.utils import KeywordsStoppingCriteria
from pointllm.data import ObjectPointCloudDataset
from tqdm import tqdm
from transformers import AutoTokenizer
from pointllm.eval.evaluator import start_evaluation

import os
import json
import random

PROMPT_LISTS = [
    "What is this?",
    "This is an object of ",
    "Caption this 3D model in detail."
]
    
brief_question_list = [
    "Summarize the 3D point cloud object briefly. ",
	"What kind of object is depicted by this point cloud? ",
	"Provide a short explanation of this 3D structure. ",
	"What does this collection of points represent?",
	"Offer a succinct summary of this 3D object. ",
	"Can you give a brief overview of this point cloud? ",
	"Characterize the object this point cloud is illustrating. ",
	"Share a brief interpretation of this 3D point cloud. ",
	"Provide an outline of this 3D shape's characteristics. ",
	"What object is this point cloud rendering? ",
	"Deliver a quick description of the object represented here. ",
	"How would you describe the 3D form shown in this point cloud? ",
	"What is the nature of the object this point cloud is representing? ",
	"Present a compact account of this 3D object's key features. ",
	"What can you infer about the object from this point cloud? ",
	"Offer a clear and concise description of this point cloud object. ",
	"How would you summarize this 3D data set? ",
	"Give a brief explanation of the object that this cloud of points forms.",
	"What kind of structure does this 3D point cloud depict? ",
	"Could you delineate the form indicated by this point cloud?",
	"Express in brief, what this point cloud is representing. ",
	"Give a quick overview of the object represented by this 3D cloud. ",
	"Convey a summary of the 3D structure represented in this point cloud.",
	"What kind of object is illustrated by this collection of points? ",
	"Describe the object that this point cloud forms. ",
	"How would you interpret this 3D point cloud? ",
	"Can you briefly outline the shape represented by these points? ",
	"Give a concise interpretation of the 3D data presented here. ",
	"Explain the object this point cloud depicts succinctly. ",
	"Offer a summary of the 3D object illustrated by this cloud."
]

complex_question_list = [
	"Can you tell me more about this? ",
	"What does this represent? ",
	"Can you describe this in more detail? ",
	"I'm interested in this, can you explain?" ,
	"What is this object made of? ",
	"Could you provide more info about this? ",
	"What exactly am I looking at here? ",
	"What is this? ",
	"Could you describe the detailed structure of this? ",
	"This looks interesting, can you expand on it? ",
	"Can you explain more about this form? ",
	"What can you tell me about the shape of this object? ",
	"Could you delve deeper into this? ",
	"I want to know more about this, can you help? ",
	"Can you walk me through the details of this object? ",
	"Can you provide a comprehensive account of this object? ",
	"Offer a detailed interpretation of this point cloud. ",
	"Please elucidate on the characteristics of this form. ",
	"Could you provide an in-depth description of this structure? ",
	"What does this cloud represent in its entirety? ",
	"Elaborate on the details of this point cloud, please. ",
	"Kindly furnish me with more information about this object. ",
	"Please expand on the intricate structure of this form. ",
	"Provide a meticulous explanation of what these points represent. ",
	"I request a detailed breakdown of this structure.",
	"Give a thorough rundown of this point cloud. ",
	"Can you offer a complete analysis of this object? ",
	"I would like a comprehensive explanation of this form. ",
	"Please detail the specific features of this point cloud. ",
	"Could you elaborate extensively on what this represents?"
]

def init_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)

    # * print the model_name (get the basename)
    print(f'[INFO] Model name: {os.path.basename(model_name)}')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PointLLMLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=False, use_cache=True, torch_dtype=torch.bfloat16).cuda()
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)

    conv_mode = "vicuna_v1_1"

    conv = conv_templates[conv_mode].copy()

    return model, tokenizer, conv

def load_dataset(data_path, anno_path, pointnum, conversation_types, use_color):
    print("Loading validation datasets.")
    dataset = ObjectPointCloudDataset(
        data_path=data_path,
        anno_path=anno_path,
        pointnum=pointnum,
        conversation_types=conversation_types,
        use_color=use_color,
        tokenizer=None # * load point cloud only
    )
    print("Done!")
    return dataset

def get_dataloader(dataset, batch_size, shuffle=False, num_workers=4):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

def generate_outputs(model, tokenizer, input_ids, point_clouds, stopping_criteria, do_sample=True, temperature=1.0, top_k=50, max_length=2048, top_p=0.95):
    model.eval() 
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            point_clouds=point_clouds,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            max_length=max_length,
            top_p=top_p,
            stopping_criteria=[stopping_criteria]) # * B, L'

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
    outputs = [output.strip() for output in outputs]

    return outputs

def start_generation(model, tokenizer, conv, dataloader, annos, prompt_index, output_dir, output_file):
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    qs = PROMPT_LISTS[prompt_index]

    results = {"prompt": qs}

    point_backbone_config = model.get_model().point_backbone_config
    point_token_len = point_backbone_config['point_token_len']
    default_point_patch_token = point_backbone_config['default_point_patch_token']
    default_point_start_token = point_backbone_config['default_point_start_token']
    default_point_end_token = point_backbone_config['default_point_end_token']
    mm_use_point_start_end = point_backbone_config['mm_use_point_start_end']

    if mm_use_point_start_end:
        qs = default_point_start_token + default_point_patch_token * point_token_len + default_point_end_token + '\n' + qs
    else:
        qs = default_point_patch_token * point_token_len + '\n' + qs
    
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)

    responses = []

    cnt = 0
    for batch in tqdm(dataloader):
        cnt += 1
        # if (cnt > 10):
        #     break
        batchsize = len(batch["object_ids"])
        # Assert batch size is 1
        # assert batchsize == 1
        
        prompt = conv.get_prompt()
        # generation_prompt = short_template.replace("{caption}", annos[batch["object_ids"][0]])
        # generation_prompt = "Please generate a question and the corresponding answer about this object based on the point cloud and the caption. The answer should directly answer the question. Here is the caption for the point cloud: " + annos[batch["object_ids"][0]]

        # brief_question_len = len(brief_question_list)
        # random_number = random.randint(0, brief_question_len-1)
        # brief_question = brief_question_list[random_number]

        prompt_list = []
        for i in range(batchsize):
            complex_question_len = len(complex_question_list)
            random_number = random.randint(0, complex_question_len-1)
            complex_question = complex_question_list[random_number]

            generation_prompt = f"{complex_question} I can provide you with a reference caption for the point cloud: " + annos[batch["object_ids"][i]]
            cur_prompt = prompt.replace("What is this?", generation_prompt)
            prompt_list.append(cur_prompt)

        inputs = tokenizer(prompt_list, return_tensors="pt", padding=True)
        input_ids = torch.as_tensor(inputs.input_ids).cuda() # * tensor of 1, L
        # input_ids = input_ids_.repeat(batchsize, 1) # * tensor of B, L
        
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
        
        point_clouds = batch["point_clouds"].cuda().to(model.dtype) # * tensor of B, N, C(3)
        object_ids = batch["object_ids"] # * list of string 

        outputs = generate_outputs(model, tokenizer, input_ids, point_clouds, stopping_criteria) # List of str, length is B

        # saving results
        for obj_id, output in zip(object_ids, outputs):
            responses.append({
                "object_id": obj_id,
                "ground_truth": annos[obj_id],
                "model_output": output,
                'conversation_type': 'detailed_description',
                "conversations": [
                  {"from": "human", "value": f"<point>\n{complex_question}"},
                  {"from": "gpt", "value": f"{output}"}
                  ]
            })
    
    results["results"] = responses

    os.makedirs(output_dir, exist_ok=True)
    # save the results to a JSON file
    with open(os.path.join(output_dir, output_file), 'w') as fp:
        json.dump(results, fp, indent=2)
    
    # save the results to a JSON file
    with open(os.path.join(output_dir, output_file.replace("_generated_dataset.json", "_generated_only_responses.json")), 'w') as fp:
        json.dump(responses, fp, indent=2)

    # * print info
    print(f"Saved results to {os.path.join(output_dir, output_file)}")

    return results

def main(args):
    # * ouptut
    args.output_dir = os.path.join(args.model_name, "evaluation")
    
    # * output file 
    anno_file = os.path.splitext(os.path.basename(args.anno_path))[0]
    # args.output_file = f"{anno_file}_Objaverse_{args.task_type}_prompt{args.prompt_index}.json"
    # args.output_file = f"test.json"
    args.output_file_path = os.path.join(args.output_dir, args.output_file)

    # * First inferencing, then evaluate
    if not os.path.exists(args.output_file_path):
        # * need inferencing
        # * load annotation files
        with open(args.anno_path, 'r') as fp:
            annos = json.load(fp)

        dataset = load_dataset(args.data_path, args.anno_path, args.pointnum, ("simple_description",), args.use_color)
        subset = Subset(dataset, range(40000 * args.gpu_num, 40000 * (args.gpu_num + 1)))
        dataloader = get_dataloader(subset, args.batch_size, args.shuffle, args.num_workers)
        
        model, tokenizer, conv = init_model(args)

        # * convert annos file from [{"object_id": }] to {"object_id": }
        annos = {anno["object_id"]: anno["conversations"][1]['value'] for anno in annos}

        print(f'[INFO] Start generating results for {args.output_file}.')
        results = start_generation(model, tokenizer, conv, dataloader, annos, args.prompt_index, args.output_dir, args.output_file)

        # * release model and tokenizer, and release cuda memory
        del model
        del tokenizer
        torch.cuda.empty_cache()
    else:
        # * directly load the results
        print(f'[INFO] {args.output_file_path} already exists, directly loading...')
        with open(args.output_file_path, 'r') as fp:
            results = json.load(fp)

    if args.start_eval:
        evaluated_output_file = args.output_file.replace(".json", "_generated_dataset.json")
        eval_type_mapping = {
            "captioning": "object-captioning",
            "classification": "open-free-form-classification"
        }
        start_evaluation(results, output_dir="/ossfs/workspace/nas5/guhao/PointLLM/data/generated_dataset", output_file=evaluated_output_file, eval_type=eval_type_mapping[args.task_type], model_type=args.gpt_type, parallel=True, num_workers=20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, \
        default="RunsenXu/PointLLM_7B_v1.2") 

    # * dataset type
    parser.add_argument("--data_path", type=str, default="data/objaverse_data", required=False)
    parser.add_argument("--anno_path", type=str, default="data/anno_data/PointLLM_brief_description_val_200_GT.json", required=False)
    parser.add_argument("--pointnum", type=int, default=8192)
    parser.add_argument("--use_color",  action="store_true", default=True)

    # * data loader, batch_size, shuffle, num_workers
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=10)

    # * evaluation setting
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--start_eval", action="store_true", default=False)
    parser.add_argument("--gpt_type", type=str, default="gpt-4-0613", choices=["gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", "gpt-4-0613", "gpt-4-1106-preview"], help="Type of the model used to evaluate.")
    parser.add_argument("--task_type", type=str, default="captioning", choices=["captioning", "classification"], help="Type of the task to evaluate.")
    parser.add_argument("--output_file", type=str, default="test.json")
    parser.add_argument("--gpu_num", type=int, default=0)

    args = parser.parse_args()

    # * check prompt index
    # * * classification: 0, 1 and captioning: 2. Raise Warning otherwise.
    if args.task_type == "classification":
        if args.prompt_index != 0 and args.prompt_index != 1:
            print("[Warning] For classification task, prompt_index should be 0 or 1.")
    elif args.task_type == "captioning":
        if args.prompt_index != 2:
            print("[Warning] For captioning task, prompt_index should be 2.")
    else:
        raise NotImplementedError

    main(args)