import math
from tqdm import tqdm
import argparse
import torch
import json
import os
import re
import re
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
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from omchat.eval_scripts.make_context import make_context, make_context_yi
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

def draw_rectangle(image_pil, coordinates, matches):
    # img = Image.open(image_path)
    bbox_prompt_color_map = {}
    draw = ImageDraw.Draw(image_pil)
    for i, coordinate in enumerate(coordinates):
        if i == 0:
            draw.rectangle(coordinate, outline='red')
            bbox_prompt_color_map[matches[i]] = "red box"
        elif i == 1:
            draw.rectangle(coordinate, outline='yellow')
            bbox_prompt_color_map[matches[i]] = "yellow box"
        elif i == 2:
            draw.rectangle(coordinate, outline='blue')
            bbox_prompt_color_map[matches[i]] = "blue box"
        elif i == 3:
            draw.rectangle(coordinate, outline='green')
            bbox_prompt_color_map[matches[i]] = "green box"
        elif i == 4:
            draw.rectangle(coordinate, outline='pink')
            bbox_prompt_color_map[matches[i]] = "pink box"
    return image_pil, bbox_prompt_color_map

def visual_prompt(prompts, image_pil):
    width, height = image_pil.size
    bboxes = []
    all_matches = []
    for prompt in prompts:
        # todo 
        # Now only support 3 bboxes
        patterns = [r"<box_0>(.*?)</box_0>", r"<box_1>(.*?)</box_2>", r"<box_3>(.*?)</box_3>"]
        for i, pattern in enumerate(patterns):
            matches = re.findall(pattern, prompt)
            all_matches.extend(matches)
            if matches:
                for match in matches:
                    # print(i, match)
                    _pattern = r"\((\d+),(\d+)\),\((\d+),(\d+)\)"
                    _match = re.search(_pattern, match)
                    if _match:
                        num1 = int(_match.group(1))
                        num2 = int(_match.group(2))
                        num3 = int(_match.group(3))
                        num4 = int(_match.group(4))
                        bbox = [num1/1000*width, num2/1000*height, num3/1000*width, num4/1000*height]
                        bboxes.append(bbox)
                    
    image_pil, bbox_prompt_color_map = draw_rectangle(image_pil, bboxes, all_matches)
    return image_pil, bbox_prompt_color_map

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


def eval_coco(annFile, resFile):
    coco = COCO(annFile)
    cocoRes = coco.loadRes(resFile)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.evaluate()

    METEOR = cocoEval.eval["METEOR"]
    CIDEr = cocoEval.eval["CIDEr"]
    total = METEOR + CIDEr
    score = {"METEOR": METEOR, "CIDEr": CIDEr, "total": total}

    return score

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

    
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    image_ids = [_["image_id"] for _ in questions]
    results = []
    for line, image_id in tqdm(zip(questions, image_ids), total=len(questions)):
        conv = conv_templates[args.conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ("user", "assistant")
        else:
            roles = conv.roles
            
        idx = image_id
        raw_question = line["question"]
        bbox = raw_question.split("[objects].")[-1]
        qs = "In the conversation, objects in image are represented by <box_0>(0,100), (1000,1000)</box_0>. The number 0 in box_0 is image_id, (0,100),(1000,1000) is bbox formatted by (Xtopleft, Ytopleft), (Xbottomright, Ybottomright), which is normolized and then scaled from 0 to 1000. \n{}".format(
            raw_question.strip()).replace("<box_0></box_0>\n<image_0>", bbox).replace("<image_0>", "<image>")
        # qs += "\nGenerate a non-interrogative English caption: "   ## 
        # qs = raw_question.strip().replace("<box_0></box_0>\n<image_0>", bbox).replace("<image_0>", "<image>")
        # qs += "\nGenerate the caption to describe the [objects] in English: "
        # qs += "\nGenerate the caption to describe the box_0 in English: "
        # qs += "\nGenerate a non-interrogative English caption for a given set of coordinates:  "
        # qs = "Describe the content of the red box: "
        # qs += "\n" 
        # print(qs)
        cur_prompt = qs
        if "image" in line:
            image = load_image(os.path.join(args.image_folder, line["image"]))
            if "<box_0>" in raw_question:
                # add draw bbox 
                image, bbox_prompt_color_map = visual_prompt([raw_question], image)
                # image.save(os.path.join('/data3/ljj/proj/omchat/img_output', str(idx) + ".jpg"))
                # print(idx, "--------")
                # break
                for bbox_prompt in bbox_prompt_color_map.keys():
                    question = raw_question.replace(bbox_prompt, "")
                bbox_prompt_color_list = list(bbox_prompt_color_map.values())
                question = question.replace("[objects]", " and ".join(bbox_prompt_color_list))
                bbox = raw_question.split("[objects].")[-1]
                # qs = "In the conversation, objects in image are represented by <box_0>(0,100), (1000,1000)</box_0>. The number 0 in box_0 is image_id, (0,100),(1000,1000) is bbox formatted by (Xtopleft, Ytopleft), (Xbottomright, Ybottomright), which is normolized and then scaled from 0 to 1000. \n{}".format(question.strip()).replace("<box_0></box_0>\n<image_0>", "")
                qs = "In the conversation, objects in image are represented by <box_0>(0,100), (1000,1000)</box_0>. The number 0 in box_0 is image_id, (0,100),(1000,1000) is bbox formatted by (Xtopleft, Ytopleft), (Xbottomright, Ybottomright), which is normolized and then scaled from 0 to 1000. \n{}".format(question.strip()).replace("<box_0></box_0>\n<image_0>", bbox).replace("<image_0>", "<image>")
                # qs += "\nGenerate a non-interrogative English caption to describe the red box: " 
                # qs += "\nGenerate a non-interrogative English caption for a given set of coordinates:  "
                # qs = "In the conversation, objects in image are represented by <box_0>(0,100), (1000,1000)</box_0>. The number 0 in box_0 is image_id, (0,100),(1000,1000) is bbox formatted by (Xtopleft, Ytopleft), (Xbottomright, Ybottomright), which is normolized and then scaled from 0 to 1000. \n"
                # qs = "Generate a non-interrogative English caption to describe the red box: "
                # qs = "Generate the caption to describe the red box in English: "
                # qs = "Describe the content of the red box: "
                # qs = question.strip().replace("<box_0></box_0>\n<image_0>", bbox).replace("<image_0>", "<image>")
                cur_prompt = qs
                # print(qs)
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
                inp = f"{roles[0]}: " + qs.replace("<image>", "").strip()

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
                images=image_tensor,
                do_sample=True,
                temperature=args.temperature,
                max_new_tokens=512,
                streamer=streamer,
                use_cache=True,
                no_repeat_ngram_size=6,
                stopping_criteria=[stopping_criteria],
            )

        output = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
        if "qllama" not in model_name.lower() and "yi"  not in model_name.lower():
            conv.messages[-1][-1] = output

        # print(output)
        results.append(
            {
                "image_id": idx,
                "prompt": cur_prompt,
                "caption": output.split("</s>")[0] if "qllama" not in model_name.lower() and "yi" not in model_name.lower() else output.split("<|im_end|>", 1)[0],
            }
        )
    print("====================== Eval Pointing Task ======================")
    print(f"Evaluating {args.dataset} ...")

    json.dump(results, open(answers_file, "w"), ensure_ascii=False, indent=2)

    annotation_path = args.annotations_file
    score = eval_coco(annotation_path, answers_file)

    print(score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, default="")
    parser.add_argument("--answers-file", type=str, default="")
    parser.add_argument("--annotations-file", type=str, default="")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--dataset", type=str, default="")

    args = parser.parse_args()
    main(args)
