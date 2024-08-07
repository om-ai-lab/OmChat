import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from omchat.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from omchat.conversation import conv_templates, SeparatorStyle
from omchat.model.builder import load_pretrained_model
from omchat.utils import disable_torch_init
from omchat.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
import requests
from io import BytesIO
from PIL import Image
import math
from transformers import PreTrainedTokenizer
from typing import List, Tuple
from omchat.eval_scripts.make_context import make_context, make_context_yi
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, image_size=args.image_size
    )

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for i, line in enumerate(tqdm(questions)):
        
        conv = conv_templates[args.conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ("user", "assistant")
        else:
            roles = conv.roles
        
        idx = line["id"]
        question = line["conversations"][0]
        qs = question["value"].replace("<image>", "").strip()
        answer = line["conversations"][1]["value"]

        if args.single_pred_prompt:
            if args.is_cn:
                qs = (
                qs
                + "\n"
                + "直接使用给定选项的字母进行回答。"
                )

            else:
                qs = (
                    qs
                    + "\n"
                    + "Answer with the option's letter from the given choices directly."
                )

        if args.freeform_answering:
            if args.is_cn:
                qs = qs + "\n" + "用一个单词或短语回答问题。"

            else:
                qs = qs + "\n" + "Answer the question using a single word or phrase."

        cur_prompt = qs
        if "image" in line:
            image = load_image(os.path.join(args.image_folder, line["image"]))
            image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
            if type(image_tensor) == list:
                image_tensor = image_tensor[0].half().cuda()
            else:
                image_tensor = image_tensor.half().cuda()
            
            if "qllama" in model_name.lower():
                inp, context_tokens = make_context(
                    tokenizer,
                    DEFAULT_IMAGE_TOKEN + "\n" + qs.replace("<image>", "").strip(),
                    None,
                    "You are a helpful assistant.",
                )
            if "yi" in model_name.lower():
                inp, context_tokens = make_context_yi(
                    tokenizer,
                    DEFAULT_IMAGE_TOKEN + "\n" + qs.replace("<image>", "").strip(),
                    None,
                    "You are a helpful assistant.",
                )

            else:
                inp = f"{roles[0]}: " + qs

                # first message
                if model.config.mm_use_im_start_end:
                    inp = (
                        DEFAULT_IM_START_TOKEN
                        + DEFAULT_IMAGE_TOKEN
                        + DEFAULT_IM_END_TOKEN
                        + "\n"
                        + inp
                    )
                else:
                    inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
                conv.append_message(conv.roles[0], inp)

        else:
            if "qllama" in model_name.lower():
                image_tensor = None
                inp, context_tokens = make_context(
                    tokenizer, qs, None, "You are a helpful assistant."
                )
            if "yi" in model_name.lower():
                image_tensor = None
                inp, context_tokens = make_context_yi(
                    tokenizer, qs, None, "You are a helpful assistant."
                )
            else:
                # later messages
                image_tensor = None
                inp = f"{roles[0]}: " + qs
                conv.append_message(conv.roles[0], inp)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        if "qllama" in model_name.lower() or "yi" in model_name.lower():
            input_ids = torch.tensor([context_tokens]).cuda()
            keywords = "<|im_end|>"
        else:
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = (
                tokenizer_image_token(
                    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .cuda()
            )
            keywords = stop_str
            
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (
            (input_ids != output_ids[:, :input_token_len]).sum().item()
        )
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        outputs = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        # print(outputs)
        # prompt for answer
        if args.answer_prompter:
            outputs_reasoning = outputs
            input_ids = (
                tokenizer_image_token(
                    prompt + outputs_reasoning + " ###\nANSWER:",
                    tokenizer,
                    IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                )
                .unsqueeze(0)
                .cuda()
            )

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=64,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (
                (input_ids != output_ids[:, :input_token_len]).sum().item()
            )
            if n_diff_input_output > 0:
                print(
                    f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
                )
            outputs = tokenizer.batch_decode(
                output_ids[:, input_token_len:], skip_special_tokens=True
            )[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()
            outputs = outputs_reasoning + "\n The answer is " + outputs

        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "prompt": cur_prompt,
                    "text": outputs if "qllama" not in model_name.lower() and "yi" not in model_name.lower() else outputs.split("<|im_end|>", 1)[0],
                    "answer_id": ans_id,
                    "model_id": model_name,
                    "answer": answer,
                    "metadata": {},
                }, ensure_ascii=False
            )
            + "\n"
        )
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--freeform-answering", action="store_true")
    parser.add_argument("--is-cn", action="store_true")
    args = parser.parse_args()

    eval_model(args)
