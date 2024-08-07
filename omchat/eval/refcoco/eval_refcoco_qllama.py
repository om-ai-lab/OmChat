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
import re
from torchvision.ops.boxes import box_area
from omchat.eval_scripts.make_context import make_context, make_context_yi
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

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

    
    questions = [
        json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")
    ]
    
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    outputs = []
    for line in tqdm(questions, total=len(questions)):
        conv = conv_templates[args.conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ("user", "assistant")
        else:
            roles = conv.roles

        idx = line["image"]
        cur_prompt = line["sent"]
        bbox = line["bbox"]
        hw = (line["height"], line["width"])
        qs = "In the conversation, objects in image are represented by <box_0>(0,100),(1000,1000)</box_0>. The number 0 in box_0 is image_id, (0,100),(1000,1000) is bbox formatted by (Xtopleft, Ytopleft), (Xbottomright, Ybottomright), which is normolized and then scaled from 0 to 1000. Spotlight the location of {}".format(cur_prompt.strip())
        
        if "image" in line:
            image = load_image(os.path.join(args.image_folder, line["image"]))
            image_tensor = image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ]

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
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(
                    dtype=torch.float16, device="cuda", non_blocking=True
                ),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=28,
                streamer=streamer,
                use_cache=True,
                no_repeat_ngram_size=6,
                stopping_criteria=[stopping_criteria],
            )

        output = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
        if "qllama" not in model_name.lower() and "yi"  not in model_name.lower():
            conv.messages[-1][-1] = output

        outputs.append(
            {
                "image": idx,
                "answer": output.split("</s>")[0] if "qllama" not in model_name.lower() and "yi" not in model_name.lower()
                else output.split("<|im_end|>", 1)[0],
                "prompt": cur_prompt,
                "gt_bbox": bbox,
                "hw": hw,
                "model_id": model_name,
            }
        )
        # print(answers)

    json.dump(outputs, open(answers_file, "w"), ensure_ascii=False, indent=2)
    PATTERN = re.compile(r"\((.*?)\),\((.*?)\)")
    correct = total_cnt = 0
    for i, output in enumerate(outputs):
        predict_bbox = re.findall(PATTERN, output["answer"])
        try:
            if "," not in predict_bbox[0][0] or "," not in predict_bbox[0][1]:
                predict_bbox = (0.0, 0.0, 0.0, 0.0)
            else:
                x1, y1 = [float(tmp) for tmp in predict_bbox[0][0].split(",")]
                x2, y2 = [float(tmp) for tmp in predict_bbox[0][1].split(",")]
                predict_bbox = (x1, y1, x2, y2)
        except:
            predict_bbox = (0.0, 0.0, 0.0, 0.0)
        target_bbox = torch.tensor(output["gt_bbox"], dtype=torch.float32).view(-1, 4)
        predict_bbox = torch.tensor(predict_bbox, dtype=torch.float32).view(-1, 4)

        iou, _ = box_iou(predict_bbox, target_bbox)
        iou = iou.item()
        total_cnt += 1
        if iou >= 0.5:
            correct += 1

    if args.is_cn:
        print("====================== Eval RefCOCO (cn version)======================")
    else:
        print("====================== Eval RefCOCO (en version)======================")
    print(f"Evaluating {args.dataset} ...")
    print(f"Precision @ 1: {correct / total_cnt} \n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--is-cn", action="store_true")
    args = parser.parse_args()
    main(args)
