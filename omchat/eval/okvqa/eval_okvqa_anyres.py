import math
from tqdm import tqdm
import argparse
import torch
import json
import os
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
from PIL import Image
from typing import List, Tuple
import requests
from io import BytesIO
from transformers import TextStreamer, PreTrainedTokenizer
from omchat.eval_scripts.okvqa.vqa import VQA
from omchat.eval_scripts.okvqa.vqa_eval import VQAEval
from omchat.mm_utils import tokenizer_image_token, process_anyres_image
from omchat.eval_scripts.make_context import make_context, make_context_yi, get_image_context

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    if "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    if not args.question_file.endswith("jsonl"):
        questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    else:
        f = open(os.path.expanduser(args.question_file))
        lines = f.readlines()
        questions = []
        for x in lines:
            questions.append(json.loads(x))

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    image_grid_pinpoints = model.config.image_grid_pinpoints
    all_outputs = []
    for i, line in enumerate(tqdm(questions)):
        conv = conv_templates[args.conv_mode].copy()

        if "mpt" in model_name.lower():
            roles = ("user", "assistant")
        else:
            roles = conv.roles

        if "id" not in line:
            idx = line["question_id"]
            question = line["text"]
            qs = question
        else:
            idx = line["id"]
            question = line["conversations"][0]
            qs = question["value"]
            qs = qs.replace("<image>", "").strip()
        qs += "Answer the question using a single word or phrase."
        cur_prompt = qs
        if "image" in line:
            image = load_image(os.path.join(args.image_folder, line["image"]))
            inp, context_tokens, image_tensor = get_image_context(model_name, image, image_processor, image_grid_pinpoints, tokenizer, qs)
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
            keywords = ["<|im_end|>"]
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
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(
                    dtype=torch.float16, device="cuda", non_blocking=True
                ),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=1024,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
        if "qllama" not in model_name.lower() and "yi"  not in model_name.lower():
            conv.messages[-1][-1] = outputs

        all_outputs.append(
            {
                "question_id": idx,
                "prompt": cur_prompt,
                "answer": outputs.split("</s>")[0]
                if "qllama" not in model_name.lower() and "yi" not in model_name.lower()
                else outputs.split("<|im_end|>", 1)[0],
                "model_id": model_name,
            }
        )
    if args.is_cn:
        print("====================== Eval OKVQA (cn version)======================")
    else:
        print("====================== Eval OKVQA (en version)======================")
        
    print(f"Evaluating okvqa_val ...")
    json.dump(all_outputs, open(answers_file, "w"), ensure_ascii=False, indent=2)

    vqa = VQA(args.annotation_file, args.question)
    results = vqa.loadRes(resFile=answers_file, quesFile=args.question)
    vqa_scorer = VQAEval(vqa, results, n=2)
    vqa_scorer.evaluate()

    print(vqa_scorer.accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--annotation-file", type=str, default="")
    parser.add_argument("--question", type=str, default="")
    parser.add_argument("--is-cn", action="store_true")
    args = parser.parse_args()
    main(args)
