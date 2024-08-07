#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel
from omchat.model import LlavaLlamaForCausalLM, OmChatQwen2ForCausalLM, OmChatLlamaForCausalLM

def load_pretrained_model(model_path, model_name, device_map="auto", image_size=448):
    kwargs = {"device_map": device_map}
    kwargs['torch_dtype'] = torch.float16

    if 'omchat' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_base or model_path, use_fast=False, trust_remote_code=True)
        model = load_omchat_model(model_path, kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_base or model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_base or model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = load_image_processor(model, image_size)
    context_len = getattr(model.config, "max_sequence_length", 2048)
    return tokenizer, model, image_processor, context_len


def load_omchat_model(model_path, kwargs):
    if "qwen" in model_path.lower():
        return OmChatQwen2ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    return OmChatLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)


def load_image_processor(model, image_size):
    vision_tower = model.get_vision_tower() if hasattr(model, 'get_vision_tower') else None
    if vision_tower and not vision_tower.is_loaded:
        vision_tower.load_model(image_size=image_size, is_train=False)
        vision_tower.to(device='cuda', dtype=torch.float16)
    return vision_tower.image_processor if vision_tower else None

