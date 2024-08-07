import torch
import os
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from omchat.model.builder import load_pretrained_model
from argparse import ArgumentParser
from omchat.eval_scripts.cmmmu.data_utils import load_yaml, construct_prompt_cmmmu, save_jsonl, process_single_sample_cmmmu
from omchat.eval_scripts.cmmmu.model_utils import call_llava_engine_df, llava_image_processor
from omchat.eval_scripts.cmmmu.eval_utils import get_multi_choice_prediction, get_TF_prediction, get_fill_blank_prediction
from omchat.eval_scripts.make_context import make_context, make_context_yi
from omchat.utils import disable_torch_init
from omchat.mm_utils import (
    get_model_name_from_path,
)

from typing import List


def run_model(args, samples, model, call_model_engine_fn=None, tokenizer=None, processor=None):
    # out_samples = dict()
    out_samples = []
    with torch.no_grad():
        for sample in tqdm(samples):
            out_sample = dict()
            response = call_model_engine_fn(args, sample, model, tokenizer, processor)

            if sample['question_type'] == '选择':
                pred_ans = get_multi_choice_prediction(response, sample['all_choices'], sample['index2ans'])
            # elif sample['question_type'] == '判断':
            #     pred_ans = get_TF_prediction(response)
            # elif sample['question_type'] == '填空':
            #     pred_ans = get_fill_blank_prediction(response, sample['answer'])
            else:  # open question
                pred_ans = response
            out_sample["id"] = int(sample['id'])
            out_sample["type"] = sample['question_type']
            out_sample["response"] = pred_ans
            out_samples.append(out_sample)
    print(len(out_samples))
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

    # load model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, vis_processors, _ = load_pretrained_model(args.model_path, None,
                                                                    model_name)
    # run for each subject
    
    category_list = ['art_and_design', 'business', 'science', 'health_and_medicine', 'humanities_and_social_sciences', 'technology_and_engineering']
    for subject in category_list:
        sub_dataset_list = []
        print (subject, args.data_path)
        sub_dataset = load_dataset(args.data_path, subject, split=args.split)
        sub_dataset_list.append(sub_dataset)
    
        # merge all dataset
        dataset = concatenate_datasets(sub_dataset_list)
    
        samples = []
        for sample in dataset:
            sample = process_single_sample_cmmmu(sample)
            sample = construct_prompt_cmmmu(sample, args.config)
            if sample['image']:
                sample['image'] = vis_process_func(sample['image'], vis_processors)#.to(device)
            samples.append(sample)

        # run ex
        out_samples = run_model(args, samples, model, call_model_engine, tokenizer, processor)

        save_jsonl(os.path.join(args.output_dir, subject + "-" + args.model_id + ".jsonl"), out_samples)
    

if __name__ == "__main__":
    parser = ArgumentParser()
    #parser.add_argument("--model-path", type=str, default="checkpoints/omchat-llava-finetune_yi-6b-336_linear_en_s121")
    parser.add_argument("--model-path", type=str, default="checkpoints/omchat-llava-finetune_Yi-6B_336_linear_en_s120")
    parser.add_argument('--output_dir', type=str, default='',
                        help='name of saved path')
    parser.add_argument('--config_path', type=str, default="/data9/ljj/omchat/data/CMMMU_o/configs/llava1.5.yaml")
    parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--data_path", type=str, default="/data9/ljj/omchat/data/CMMMU_o")
    parser.add_argument('--split', type=str, default='val')
    #parser.add_argument('--split', type=str, default='test')
    # parser.add_argument("--eval-data-path", type=str, default="/data9/kyusonglee/MLLM_evals/data/MMMU")
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_id', type=str, default="")
    args = parser.parse_args()
    main(args)
