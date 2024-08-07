from tqdm import tqdm
import argparse
import torch
import json
import os, re
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
from torchvision.ops.boxes import box_area

ds_collections = {
    "refcoco_val": "data/refcoco/refcoco_val_normalized.jsonl",
    "refcoco_testA": "data/refcoco/refcoco_testA_normalized.jsonl",
    "refcoco_testB": "data/refcoco/refcoco_testB_normalized.jsonl",
    "refcoco+_val": "data/refcoco+/refcoco+_val_normalized.jsonl",
    "refcoco+_testA": "data/refcoco+/refcoco+_testA_normalized.jsonl",
    "refcoco+_testB": "data/refcoco+/refcoco+_testB_normalized.jsonl",
    "refcocog_val": "data/refcocog/refcocog_val_normalized.jsonl",
    "refcocog_test": "data/refcocog/refcocog_test_normalized.jsonl",
}


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
        qs = line["sent"]
        anno = line["bbox"]
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
        qs = "In the conversation, objects in image are represented by <box_0>(0,100),(1000,1000)</box_0>. The number 0 in box_0 is image_id, (0,100),(1000,1000) is bbox formatted by (Xtopleft, Ytopleft), (Xbottomright, Ybottomright), which is normolized and then scaled from 0 to 1000. Spotlight the location of {}".format(
            qs.strip()
        )
        # qs = "Where is {}? Give me the location of question. Answer the format of <box>(x0, y0), (x1, y1)</box>.".format(qs)
        # qs = "<ref>" + qs + "</ref> Give me the coordinates of <ref>. Only answer the coordinates. "
        # qs = "<ref>" + qs + "</ref><box>"
        # qs = "<ref>" + qs + "</ref> give me the location of question"

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
            image = image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]
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
        idx = line["image"]
        cur_prompt = line["sent"]
        # sys_prompt = "In the conversation, objects in image are represented by <box_0>(0,100),(1000,1000)</box_0>. The number 0 in box_0 is image_id, (0,100),(1000,1000) is bbox formatted by (Xtopleft, Ytopleft), (Xbottomright, Ybottomright), which is normolized and then scaled from 0 to 1000. Spotlight the location of {}".format(cur_prompt.strip()) + "\n<image>"

        bbox = line["bbox"]
        hw = (line["height"], line["width"])
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
                max_new_tokens=28,
                min_new_tokens=10,
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

        outputs.append(
            {
                "image": idx,
                "answer": answers,
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
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--is-cn", action="store_true")
    args = parser.parse_args()
    main(args)
