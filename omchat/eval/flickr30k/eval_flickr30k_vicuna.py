import argparse
import json
import os
from PIL import Image
import torch
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from tqdm import tqdm
from omchat.model.builder import load_pretrained_model
from omchat.utils import disable_torch_init
from omchat.mm_utils import tokenizer_image_token, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
from omchat.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from omchat.conversation import conv_templates, SeparatorStyle


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

        if self.model_config.mm_use_im_start_end:
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
            )
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n"
        qs = qs + "\nGenerate the caption in English: "

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model_config)[
            0
        ]

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )

        return input_ids, image_tensor

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
            image = image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors="pt")["pixel_values"]
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


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
    if args.is_cn:
        prompt = "生成中文标题："
    else:
        prompt = "Generate the caption in English:"

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

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))["images"]
    image_ids = [_["id"] for _ in questions]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    data_loader = create_data_loader(
        questions, args.image_folder, tokenizer, image_processor, model.config
    )

    results = []
    for (input_ids, image_tensor), image_id in tqdm(
        zip(data_loader, image_ids), total=len(image_ids)
    ):
        cur_prompt = prompt
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
                max_new_tokens=30,  ## 1024
                use_cache=True,
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

        results.append(
            {
                "image_id": int(image_id),
                "prompt": cur_prompt,
                "caption": answers
                if "qllama" not in model_name.lower()
                else answers.split("<|im_end|>", 1)[0],
            }
        )
    
    if args.is_cn:
        print("====================== Eval Flickr (cn version)======================")
    else:
        print("====================== Eval Flickr (en version)======================")

    print(f"Evaluating flickr30k_karpathy_test ...")
        
    json.dump(results, open(answers_file, "w"), ensure_ascii=False, indent=2)
    coco = COCO(args.question_file)
    coco_result = coco.loadRes(answers_file)
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.evaluate()

    print(coco_eval.eval.items())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/omchat-llava-qwen-7b-v1-1-finetune_zh_n48",
    )
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument(
        "--question-file", type=str, default="sciQA/scienceqa/llava_test_QCM-LEPA.json"
    )
    parser.add_argument("--answers-file", type=str, default="answer.json")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--is-cn", action="store_true")
    args = parser.parse_args()
    main(args)
