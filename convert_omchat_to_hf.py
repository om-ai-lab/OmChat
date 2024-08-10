
import argparse
import gc
import glob
import json
from pathlib import Path

import requests
import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
from safetensors import safe_open

from transformers import (
    AddedToken,
    AutoConfig,
    AutoTokenizer
)
from configuration_omchat import OmChatConfig
from configuration_omchat import InternVisionConfig
from modeling_omchat import OmChatForConditionalGeneration
from processing_omchat import OmChatProcessor
from image_processing_omchat import  OmChatImageProcessor

KEYS_TO_MODIFY_MAPPING = {
    "model.vision_tower.": "",
    "model.mm_projector": "multi_modal_projector",
    "model": "model.model",
    "vision_model.model": "vision_model",
    "lm_head": "language_model.lm_head",
    "model.model": "language_model.model",
    "multi_modal_projector.0": "multi_modal_projector.linear_1",
    "multi_modal_projector.2": "multi_modal_projector.linear_2",
}


def load_original_state_dict(directory_path):
    original_state_dict = {}
    for path in glob.glob(f"{directory_path}/*"):
        if path.endswith(".safetensors"):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    original_state_dict[key] = f.get_tensor(key)

    return original_state_dict


def convert_state_dict_to_hf(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith(".inv_freq"):
            continue
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        new_state_dict[key] = value.to(torch.float16)
    return new_state_dict


def load_image():
    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw)
    return image


def convert_omchat_to_hf(filepath, pytorch_dump_folder_path, push_to_hub=False):
    # load original config
    #filepath = hf_hub_download(repo_id=model_id, filename="config.json", repo_type="model")

    # read json
    with open(filepath) as f:
        data = json.load(f)
        print(data)

    vision_model_id = data["mm_vision_tower"]
    text_model_id = "../resources/Qwen2-7B"

    torch.set_default_dtype(torch.float16)
    text_config = AutoConfig.from_pretrained(text_model_id)

    use_fast = True
    tokenizer = AutoTokenizer.from_pretrained(text_model_id, use_fast=use_fast)
    #tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False), special_tokens=True)

    image_processor = OmChatImageProcessor.from_pretrained("../resources/InternViT-6B-448px-V1-5")
    processor = OmChatProcessor(tokenizer=tokenizer, image_processor=image_processor)
    print (data)
    
    vision_config = InternVisionConfig()
    config = OmChatConfig(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        image_grid_pinpoints=data["image_grid_pinpoints"]
    )
    with init_empty_weights():
        model = OmChatForConditionalGeneration(config)

    # load original state dict
    state_dict = load_original_state_dict("omchat-beta2")
    state_dict = convert_state_dict_to_hf(state_dict)
    model.load_state_dict(state_dict, assign=True)
    model.eval()

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)
    processor.save_pretrained(pytorch_dump_folder_path)
    exit()
    #processor.save_pretrained(pytorch_dump_folder_path)

    # Make space so we can load the model properly now.
    #del state_dict
    #gc.collect()

    # Load everything back for inference tests in float32 because prev script was written as that
    # Though it's mostly loaded in fp16 as original weights are in fp16

    #model = OmChatForConditionalGeneration.from_pretrained(pytorch_dump_folder_path, device_map="auto")
    #processor = OmChatProcessor.from_pretrained(pytorch_dump_folder_path)
if __name__ == "__main__":
    convert_omchat_to_hf("omchat-beta2/config.json", "omchat-beta2_hf")
