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
from omchat.model.builder import load_pretrained_model
from omchat.utils import disable_torch_init
from omchat.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
)
from PIL import Image
from typing import List, Tuple
import requests
from io import BytesIO
from transformers import TextStreamer
from omchat.make_context import make_context,  get_image_context, get_context


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def get_response(text, image_path=None, initial_prompt="You are a helpful assistant.", image_processor=None, model_name=None, tokenizer=None,image_grid_pinpoints=None):
    image = load_image(image_path) if image_path else None
    inp, context_tokens, image_tensor = get_context(text=text, image=image, image_processor=image_processor, image_grid_pinpoints=image_grid_pinpoints, tokenizer=tokenizer, initial_prompt=initial_prompt)
    return context_tokens, image_tensor


def main(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path=args.model_path, model_name=model_name)
    image_grid_pinpoints = model.config.image_grid_pinpoints
    context_tokens, image_tensor = get_response(args.question, args.image_path, model_name=model_name,tokenizer=tokenizer,image_processor=image_processor,image_grid_pinpoints=image_grid_pinpoints)
    input_ids = torch.tensor([context_tokens]).cuda()
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
            eos_token_id=151645
        )

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/data2/omchat_dev/omchat/checkpoints/omchat-qwen2-7b-qllama-internvit6b-fk58")
    parser.add_argument("--image-path", type=str, default="images/extreme_ironing.jpg")
    parser.add_argument(
        "--question",
        type=str,
        default="What is unusual about this image? can you explain this to a 5-year-old kid?",
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)







