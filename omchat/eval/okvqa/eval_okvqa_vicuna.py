from tqdm import tqdm
import argparse
import torch
import json
import os
import math
from omchat.model.builder import load_pretrained_model
from omchat.utils import disable_torch_init
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from omchat.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from omchat.conversation import conv_templates, SeparatorStyle
from omchat.mm_utils import tokenizer_image_token, get_model_name_from_path
from omchat.eval_scripts.okvqa.vqa import VQA
from omchat.eval_scripts.okvqa.vqa_eval import VQAEval


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(
        self, questions, image_folder, tokenizer, image_processor, model_config
    ):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        anno = line["answer"]
        if self.model_config.mm_use_im_start_end:
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + qs
            )
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        # qs = qs + "\nBrief answer."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )

        return input_ids, image_tensor, anno

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(
    questions,
    image_folder,
    tokenizer,
    image_processor,
    model_config,
    batch_size=1,
    num_workers=4,
):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(
        questions, image_folder, tokenizer, image_processor, model_config
    )
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    return data_loader


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == "pad":
        for image in images:
            image = expand2square(
                image, tuple(int(x * 255) for x in image_processor.image_mean)
            )
            image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors="pt")["pixel_values"]
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        load_4bit=False,
        image_size=args.image_size,
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
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    data_loader = create_data_loader(
        questions, args.image_folder, tokenizer, image_processor, model.config
    )

    outputs = []
    for (input_ids, image_tensor, anno), line in tqdm(
        zip(data_loader, questions), total=len(questions)
    ):
        idx = line["question_id"]
        cur_prompt = line["text"]
        cur_prompt += "Answer the question using a single word or phrase."
        stop_str = (
            conv_templates[args.conv_mode].sep
            if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO
            else conv_templates[args.conv_mode].sep2
        )
        input_ids = input_ids.to(device="cuda", non_blocking=True)

        keywords = [stop_str]
        stopping_criteria = None

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=1024,
                use_cache=True,
                no_repeat_ngram_size=6,
                stopping_criteria=stopping_criteria,
            )
        input_token_len = input_ids.shape[1]
        output = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        output = output.strip()
        if output.endswith(stop_str):
            output = output[: -len(stop_str)]
        answers = output.strip()
        print(answers)
        outputs.append(
            {
                "question_id": idx,
                "prompt": cur_prompt,
                "answer": answers,
                "model_id": model_name,
            }
        )
        # print(outputs)
        
    if args.is_cn:
        print("====================== Eval OKVQA (cn version)======================")
    else:
        print("====================== Eval OKVQA (en version)======================")
        
    print(f"Evaluating okvqa_val ...")
    json.dump(outputs, open(answers_file, "w"), ensure_ascii=False, indent=2)

    vqa = VQA(args.annotation_file, args.question)
    results = vqa.loadRes(resFile=answers_file, quesFile=args.question)
    vqa_scorer = VQAEval(vqa, results, n=2)
    vqa_scorer.evaluate()

    print(vqa_scorer.accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/omchat-llava-qwen-7b-v1-1-finetune_zh_n48",
    )
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument(
        "--image-folder", type=str, default="/data3/kyusong/data/test_images"
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--question-file", type=str, default="sciQA/scienceqa/llava_test_QCM-LEPA.json"
    )
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--annotation-file", type=str, default="")
    parser.add_argument("--question", type=str, default="")
    parser.add_argument("--is-cn", action="store_true")
    args = parser.parse_args()
    main(args)
