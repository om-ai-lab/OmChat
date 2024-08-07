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
from typing import Optional
from omchat.eval_scripts.make_context import make_context, make_context_yi

ds_collections = {
    "test_human": {
        "test": "data/chartqa/test_human.jsonl",
        "metric": "relaxed_accuracy",
        "max_new_tokens": 100,
    },
    "test_augmented": {
        "test": "data/chartqa/test_augmented.jsonl",
        "metric": "relaxed_accuracy",
        "max_new_tokens": 100,
    },
}

# https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py#L81
def relaxed_correctness(
    target: str, prediction: str, max_relative_change: float = 0.05
) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith("%"):
                # Convert percentages to floats.
                return float(text.rstrip("%")) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()

def evaluate_relaxed_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem["annotation"], str):
            elem["annotation"] = [elem["annotation"]]
        score = max(
            [
                relaxed_correctness(elem["answer"].strip(), ann)
                for ann in elem["annotation"]
            ]
        )
        scores.append(score)
    return str(round((sum(scores) / len(scores)) * 100, 3))



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

    outputs = []
    for i, line in enumerate(tqdm(questions)):
        conv = conv_templates[args.conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ("user", "assistant")
        else:
            roles = conv.roles

        if "id" not in line:
            idx = line["question_id"]
            question = line["question"]
            qs = question 
        else:
            idx = line["id"]
            question = line["conversations"][0]
            # gt_ans = line["conversations"][1]
            qs = question["value"]
        qs = qs + "\nBrief answer."  
        cur_prompt = qs
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
                max_new_tokens=ds_collections[args.dataset]["max_new_tokens"],
                streamer=streamer,
                use_cache=True,
                no_repeat_ngram_size=6,
                stopping_criteria=[stopping_criteria],
            )

        answers = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
        if "qllama" not in model_name.lower() and "yi"  not in model_name.lower():
            conv.messages[-1][-1] = outputs
        
        outputs.append(
            {
                "question_id": idx,
                "prompt": cur_prompt,
                "answer": answers.split("</s>")[0]
                if "qllama" not in model_name.lower() and "yi" not in model_name.lower()
                else answers.split("<|im_end|>", 1)[0],
                "annotation": line["answer"],
                "model_id": model_name,
            }
        )
    print("====================== Eval ChartQA ======================")
    print(f"Evaluating {args.dataset} ...")
    json.dump(outputs, open(answers_file, "w"), ensure_ascii=False, indent=2)

    print("relaxed_accuracy: {} %".format(evaluate_relaxed_accuracy(outputs)))


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
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="")

    args = parser.parse_args()
    main(args)
