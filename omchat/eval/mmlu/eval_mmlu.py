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
from typing import List, Tuple
from transformers import TextStreamer, PreTrainedTokenizer
from omchat.eval_scripts.make_context import make_context, make_context_yi
TASK_NAME_MAPPING = {
    "stem": [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "electrical_engineering",
        "elementary_mathematics",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_statistics",
        "machine_learning",
    ],
    "Humanities": [
        "formal_logic",
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "moral_disputes",
        "moral_scenarios",
        "philosophy",
        "prehistory",
        "professional_law",
        "world_religions",
    ],
    "other": [
        "business_ethics",
        "college_medicine",
        "human_aging",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "nutrition",
        "professional_accounting",
        "professional_medicine",
        "virology",
        "global_facts",
        "clinical_knowledge",
    ],
    "social": [
        "econometrics",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "high_school_psychology",
        "human_sexuality",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
    ],
}

SUBJECTS = [v for vl in TASK_NAME_MAPPING.values() for v in vl]

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


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

    for subject_name in tqdm(SUBJECTS):
        choose_eval = args.choose_eval_path
        val_file_path = os.path.join(args.eval_data_path, choose_eval, f"{subject_name}_{choose_eval}.json")
        questions = json.load(open(os.path.expanduser(val_file_path), "r"))
        answers_file = os.path.join(args.answers_file, choose_eval, model_name.rsplit("_", 1)[-1], f"{subject_name}_{choose_eval}_result.jsonl")
        
        questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

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
            qs = question["value"]
            if args.single_pred_prompt:
                qs = (
                    qs
                    + "\n"
                    + "Answer with the option's letter from the given choices directly."
                )
            cur_prompt = qs

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
                    max_new_tokens=1024,
                    streamer=streamer,
                    use_cache=True,
                    no_repeat_ngram_size=6,
                    stopping_criteria=[stopping_criteria],
                )

            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
            if "qllama" not in model_name.lower() and "yi"  not in model_name.lower():
                conv.messages[-1][-1] = outputs

            if args.debug:
                print("\n", {"prompt": cur_prompt, "outputs": outputs}, "\n")
            
            ans_file.write(
                json.dumps(
                    {
                        "question_id": idx,
                        "prompt": cur_prompt,
                        "text": outputs.split("</s>")[0]
                        if "qllama" not in model_name.lower() and "yi" not in model_name.lower()
                        else outputs.split("<|im_end|>", 1)[0],
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
    parser.add_argument("--eval-data-path", type=str, default="")
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--choose-eval-path", type=str, default="val")
    args = parser.parse_args()
    main(args)
