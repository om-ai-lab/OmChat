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
from omchat.model import OmChatQwen2ForCausalLM

def load_pretrained_model(model_path, model_name, device_map="auto", device="cuda", **kwargs):
    kwargs = {"device_map": device_map, **kwargs}
    if device != "cuda":
        kwargs['device_map'] = {"":device}
    

    kwargs['torch_dtype'] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = load_omchat_model(model_path, kwargs)

    image_processor = load_image_processor(model, device)
    context_len = getattr(model.config, "max_sequence_length", 2048)
    return tokenizer, model, image_processor, context_len


def load_omchat_model(model_path, kwargs):
    return OmChatQwen2ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, delay_load=False, **kwargs)


def load_image_processor(model, device):
    vision_tower = model.get_vision_tower() if hasattr(model, 'get_vision_tower') else None
    if vision_tower and not vision_tower.is_loaded:
        vision_tower.load_model(is_train=False)
        vision_tower.to(device=device, dtype=torch.float16)
    return vision_tower.image_processor if vision_tower else None

