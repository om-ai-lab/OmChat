import torch

import os
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from omchat.model.builder import load_pretrained_model
from omchat.mm_utils import get_model_name_from_path
from argparse import ArgumentParser
from omchat.eval_scripts.mmmu.data_utils import load_yaml, construct_prompt, save_json, process_single_sample, CAT_SHORT2LONG
from omchat.eval_scripts.mmmu.model_utils import call_llava_engine_df, llava_image_processor
from omchat.eval_scripts.mmmu.eval_utils import parse_multi_choice_response, parse_open_response
from omchat.eval_scripts.make_context import make_context, make_context_yi
from omchat.conversation import conv_templates, SeparatorStyle
from omchat.model.builder import load_pretrained_model
from omchat.utils import disable_torch_init
from omchat.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from omchat.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from transformers import TextStreamer, PreTrainedTokenizer
from typing import List, Tuple

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def run_model(args, samples, model, call_model_engine_fn=None, tokenizer=None, processor=None):
    out_samples = dict()
    with torch.no_grad():
        for sample in tqdm(samples):
            response = call_model_engine_fn(args, sample, model, tokenizer, processor)

            if sample['question_type'] == 'multiple-choice':
                pred_ans = parse_multi_choice_response(response, sample['all_choices'], sample['index2ans'])
            else:  # open question
                pred_ans = response
            out_samples[sample['id']] = pred_ans
    return out_samples
def set_seed(seed_value):
    """
    Set the seed for PyTorch (both CPU and CUDA), Python, and NumPy for reproducible results.

    :param seed_value: An integer value to be used as the seed.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def main(args):
    print("start")
    set_seed(args.seed)
    # Model
    disable_torch_init()

    #model_name = get_model_name_from_path(args.model_path)
    #tokenizer, model, image_processor, context_len = load_pretrained_model(
    #    args.model_path, args.model_base, model_name
    #)

    processor = None
    call_model_engine = call_llava_engine_df
    vis_process_func = llava_image_processor

    # load config and process to one value
    args.config = load_yaml(args.config_path)
    for key, value in args.config.items():
        if key != 'eval_params' and type(value) == list:
            assert len(value) == 1, 'key {} has more than one value'.format(key)
            args.config[key] = value[0]

    # run for each subject
    sub_dataset_list = []
    for subject in CAT_SHORT2LONG.values():
        print (subject, args.data_path)
        sub_dataset = load_dataset(args.data_path, subject, split=args.split)
        sub_dataset_list.append(sub_dataset)

    # merge all dataset
    dataset = concatenate_datasets(sub_dataset_list)


    # load model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, vis_processors, _ = load_pretrained_model(args.model_path, None,
                                                                model_name)
   
    samples = []
    for sample in dataset:
        sample = process_single_sample(sample)
        sample = construct_prompt(sample, args.config)
        if sample['image']:
            sample['image'] = vis_process_func(sample['image'], vis_processors)#.to(device)
        samples.append(sample)

    # run ex
    out_samples = run_model(args, samples, model, call_model_engine, tokenizer, processor)

    save_json(args.output_path, out_samples)
    

if __name__ == "__main__":
    parser = ArgumentParser()
    #parser.add_argument("--model-path", type=str, default="checkpoints/omchat-llava-finetune_yi-6b-336_linear_en_s121")
    #parser.add_argument("--model-path", type=str, default="checkpoints/omchat-llava-finetune_yi-6b-336_linear_en_s122")
    parser.add_argument("--model-path", type=str, default="checkpoints/omchat-llava-Yi-6B-v1-1-finetune_336_linear_en_s127")
    parser.add_argument('--output_path', type=str, default='s122.json',
                        help='name of saved json')
    parser.add_argument('--config_path', type=str, default="/data9/kyusonglee/MLLM_evals/data/MMMU/configs/llava1.5.yaml")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--data_path", type=str, default="/root/omchat/data/MMMU")
    parser.add_argument('--split', type=str, default='validation')
    #parser.add_argument('--split', type=str, default='test')
    parser.add_argument("--eval-data-path", type=str, default="/data9/kyusonglee/MLLM_evals/data/MMMU")
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
