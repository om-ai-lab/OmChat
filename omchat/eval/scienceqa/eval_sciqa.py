import math
from tqdm import tqdm
import argparse
import torch
import json
import os
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
from PIL import Image
from typing import List, Tuple
import requests
from io import BytesIO
from transformers import TextStreamer
from omchat.eval.make_context import make_context,  get_image_context

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
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    print (model_name, args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path, model_name=model_name
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

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for i, line in enumerate(tqdm(questions)):
        roles = ('USER', 'ASSISTANT')
        idx = line["id"]
        question = line["conversations"][0]
        qs = question["value"]
        if args.single_pred_prompt:
            qs = (
                qs
                + "\n"
                + "Answer with the option's letter from the given choices directly."
            )
        cur_prompt = qs
        image_grid_pinpoints = model.config.image_grid_pinpoints
        if "image" in line:
            image = load_image(os.path.join(args.image_folder, line["image"]))
            inp, context_tokens, image_tensor = get_image_context(model_name, image, image_processor, image_grid_pinpoints, tokenizer, qs)

        else:
            image_tensor = None
            inp, context_tokens = make_context(
                    tokenizer, qs, None, "You are a helpful assistant."
            )

        input_ids = torch.tensor([context_tokens]).cuda()
        keywords = ["<|im_end|>"]

        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        model.generation_config.pad_token_id = tokenizer.pad_token_id


        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=args.temperature,
                max_new_tokens=1024,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()

        if args.debug:
            print("\n", {"prompt": cur_prompt, "outputs": outputs}, "\n")
        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "prompt": cur_prompt,
                    "text": outputs.split("<|im_end|>", 1)[0],
                    "answer_id": ans_id,
                    "model_id": model_name,
                    "metadata": {},
                },
                ensure_ascii=False,
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
    parser.add_argument("--single-pred-prompt", action="store_true")
    args = parser.parse_args()
    main(args)
