import os
import re
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

import transformers

from omchat.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from omchat.train.omchat_trainer import OmChatTrainer
from omchat import conversation as conversation_lib
from omchat.model.language_model.omchat_llama import OmChatLlamaForCausalLM
from omchat.model import *
from omchat.mm_utils import tokenizer_image_token, process_anyres_image, process_dynamic_image
from transformers.trainer_pt_utils import LabelSmoother
from transformers import AutoModelForCausalLM
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from omchat.model.processing_omclip import OmClipImageProcessor
from omchat.model.multimodal_encoder.omclip.model import load_from_pretrained,resize_pos_embed_to_new_image_size

from PIL import Image, ImageDraw

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

local_rank = None

def split_and_add_number(s):
    numbers = re.findall(r'\d+', s)
    if numbers:
        split_str = re.split(r'(\d+)', s)
        return split_str[0] + numbers[0]
    else:
        return s

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    trainable_layers: List[int] = field(default_factory=lambda: [])
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    mm_projector_type: Optional[str] = field(default='linear')
    mm_projector_n_query: Optional[int] = field(default=144)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    norm_class: str = field(default="LlamaRMSNorm")
    rotary_type: str = field(default="fused_rotary")
    max_position_embeddings: int = field(default=512000)  # 512000 for 512k, 4096 for origin Yi-6B
    rope_theta: float = field(default=50000000.0)   # 50000000.0 for 512k, 5000000.0 for origin Yi-6B
    num_experts: Optional[int] = field(default=1)
    num_selected: Optional[int] = field(default=1)
    num_layers: Optional[int] = field(default=3)
    balance_loss_coef: Optional[float] = field(default=0.0)
    router_z_loss_coef: Optional[float] = field(default=0.0)
    dropout: Optional[bool] = field(default=False)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    scales: Optional[str] = field(default=None)
    scale_factor: int = 2

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    big_data: bool = False
    image_size: int = field(default=448)
    max_num: int = field(default=6)
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    data_type: str = 'qlamma' #qlamma or vicuna
    data_ratio: List[int] = field(default_factory=lambda: [])
    image_grid_pinpoints: str = "[(672, 672), (336, 672), (672, 336), (336, 1008), (1008, 336)]"


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    end_learning_rate: float = 0.0
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    use_flash_attention_2: bool = field(default=True)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    trainer: Optional[str] = field(default="OmChatTrainer")
    beta: Optional[float] = field(
        default=0.1,
        metadata={
            "help":
            "Only used for OmChatDPOTrainer"
        },
    )
    include_tokens_per_second: Optional[bool] = field(
        default=False,
        metadata={"help": "If set to `True`, the speed metrics will include `tgs` (tokens per second per device)."},
    )
    torch_compile: bool = field(
        default=False, metadata={"help": "If set to `True`, the model will be wrapped in `torch.compile`."}
    )
    is_aim: bool = field(
        default=False, metadata={"help": "If set to `True`, traing metrics will be sended to aim."}
    )
    aim_url: str = field(
        default="aim://121.52.244.248:53800/", metadata={"help": "aim url"}
    )
    aim_experiment: str = field(
        default="omchat", metadata={"help": "aim_experiment"}
    )

    


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, name=k) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        """
        keys_to_match = ['vision_tower']
        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "vision_tower")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'vision_tower.bin'))
        """
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation

def preprocess_pretrain_batch(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    system_message: Optional[str]=None,
    has_image: bool=True
) -> Dict:
    conversations = []
    for source in sources:        
        conversation = source[0]['value']
        conversations.append(conversation)

    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') if isinstance(prompt, str) else torch.tensor(prompt, dtype=torch.long) for prompt in conversations][0]
    targets = copy.deepcopy(input_ids)
    return dict(input_ids=input_ids, labels=targets)

def sample_qwen_prompt(source):
    if "system_message" not in source.keys():
        system_message = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    else:
        system_message = source["system_message"]
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant", "human": "<|im_start|>user",
             "gpt": "<|im_start|>assistant"}
    nl_tokens = '\n'
    im_start, im_end = '<|im_start|>', '<|im_end|>'
    _system = 'system' + nl_tokens
    prompt = ""
    system = im_start + _system + system_message + im_end + nl_tokens
    prompt += system
    for j, sentence in enumerate(source["conversations"]):
        rank0_print(f"sample data {j}: {sentence}")
        role = roles[sentence["from"]]
        _prompt = role + nl_tokens + sentence["value"] + im_end + nl_tokens
        prompt += _prompt
    rank0_print((f"default prompt: {prompt}"))

def preprocess_qwen_batch(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    system_message: Optional[str]=None,
    has_image: bool=True
) -> Dict:
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant", "human":"<|im_start|>user", "gpt":"<|im_start|>assistant"}
    if not system_message:
        system_message = "You are a helpful assistant."
    
    im_start = 151644
    im_end = 151645
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    
    # Apply prompt templates
    max_len = 0 
    input_ids, rejected_input_ids, targets, rejected_targets = [], [], [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        rejected_input_id,rejected_target = [], []
        rejected_input_id += system
        rejected_target += [im_start] + [IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens
        assert len(rejected_input_id) == len(rejected_target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            if isinstance(sentence["value"], bool):
                sentence["value"] = str(sentence["value"])
            if isinstance(sentence["value"], str):
                if not has_image:
                    _input_id = tokenizer(role).input_ids + nl_tokens + \
                        tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
                else:
                    _input_id = tokenizer(role).input_ids + nl_tokens + \
                        tokenizer_image_token(sentence["value"], tokenizer) + [im_end] + nl_tokens

                input_id += _input_id
                if role == '<|im_start|>user':
                    _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
                elif role == '<|im_start|>assistant':
                    _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                        _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
                else:
                    raise NotImplementedError
                target += _target
            elif isinstance(sentence["value"], dict):
                #choosen value
                if not has_image:
                    _input_id = tokenizer(role).input_ids + nl_tokens + \
                        tokenizer(sentence["value"]["chosen"]).input_ids + [im_end] + nl_tokens
                else:
                    _input_id = tokenizer(role).input_ids + nl_tokens + \
                        tokenizer_image_token(sentence["value"]["chosen"], tokenizer) + [im_end] + nl_tokens

                input_id += _input_id
                if role == '<|im_start|>user':
                    _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
                elif role == '<|im_start|>assistant':
                    _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                        _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
                #rejected value
                if not has_image:
                    _rejcted_input_id = tokenizer(role).input_ids + nl_tokens + \
                        tokenizer(sentence["value"]["rejected"]).input_ids + [im_end] + nl_tokens
                else:
                    _rejcted_input_id = tokenizer(role).input_ids + nl_tokens + \
                        tokenizer_image_token(sentence["value"]["rejected"], tokenizer) + [im_end] + nl_tokens

                rejected_input_id += _rejcted_input_id
                if role == '<|im_start|>user':
                    _rejected_target = [im_start] + [IGNORE_TOKEN_ID] * (len(_rejcted_input_id)-3) + [im_end] + nl_tokens
                elif role == '<|im_start|>assistant':
                    _rejected_target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                        _rejcted_input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens

                target += _target
                rejected_target += _rejected_target
            else:
                raise ValueError(f"sentence['value'] should be str or dict")


        max_len = max(max_len, len(input_id), len(target), len(rejected_input_id))
        input_ids.append(input_id)
        targets.append(target)
        rejected_input_ids.append(rejected_input_id)
        rejected_targets.append(rejected_target)
    
    for i in range(len(input_ids)):
        input_ids[i] += [tokenizer.pad_token_id] * (max_len - len(input_ids[i]))
        targets[i] += [IGNORE_TOKEN_ID] * (max_len - len(targets[i]))
        rejected_input_ids[i] += [tokenizer.pad_token_id] * (max_len - len(rejected_input_ids[i]))
        rejected_targets[i] += [IGNORE_TOKEN_ID] * (max_len - len(rejected_targets[i]))

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    if sources[0][0]["from"] != roles["user"] and isinstance(sources[0][0]["value"], dict):
        rejected_input_ids = torch.tensor(rejected_input_ids, dtype=torch.long)
        rejected_targets = torch.tensor(rejected_targets, dtype=torch.long)

        return dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
            rejected_input_ids=rejected_input_ids,
            rejected_labels=rejected_targets,
            rejected_attention_mask=rejected_input_ids.ne(tokenizer.pad_token_id),
        )
    else:
        return dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments,
    num_images: list = []
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources
    output_list = []
    tag_index = 0
    if len(num_images) > 0:
        for source in sources:
            for sentence in source:
                if isinstance(sentence["value"], str):
                    if DEFAULT_IMAGE_TOKEN in sentence['value']:
                        sentence['value'] = sentence['value'].replace("<image>","").strip()+"\n"+"<image>"
                        sentence['value'] = sentence['value'].strip()
                        parts = sentence['value'].split('<image>')
                        output = ""
                        for i in range(1, len(parts)):
                            if tag_index < len(num_images):
                                count,best_res = num_images[tag_index]
                                tag_index += 1
                            else:
                                count = 1  # Default to one <image> tag if num_of_tags runs out
                            if data_args.mm_use_im_start_end:
                                output += f"{DEFAULT_IM_START_TOKEN}<image>{DEFAULT_IM_END_TOKEN}\n"+"\n".join([f"patch:{DEFAULT_IM_START_TOKEN}<image>{DEFAULT_IM_END_TOKEN}"]*(count-1)) +"\n"+ parts[0].strip()
                                #output += DEFAULT_IM_START_TOKEN+"<image>\n"+"\n".join(["patch:<image>"]*(count-1)) +"\n"+ parts[0].strip()+DEFAULT_IM_END_TOKEN
                            else:
                                output += "<image>\n"+"\n".join(["patch:<image>"]*(count-1)) +"\n"+ parts[0].strip()
                        sentence["value"] = output 

            
    else:
        for source in sources:
            for sentence in source:
                if isinstance(sentence["value"], str):
                    if DEFAULT_IMAGE_TOKEN in sentence['value']:
                        sentence['value'] = sentence['value'].strip()
                    sentence["value"] = re.sub(r'<image_\d+>', '<image>', sentence["value"])
                    if data_args.mm_use_im_start_end:
                        replace_token = DEFAULT_IM_START_TOKEN + '<image>' + DEFAULT_IM_END_TOKEN
                        sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
    return sources

def remove_value(lst, value):
    return [x for x in lst if x != value]

def sample_v1_prompt(source):
    if "system_message" not in source.keys():
        system_message = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    else:
        system_message = source["system_message"]
    conv = conversation_lib.default_conversation.copy()
    conversations = []
    conv.messages = []
    for j, sentence in enumerate(source["conversations"]):
        rank0_print(f"sample data {j}: {sentence}")
        role = roles[sentence["from"]]
        conv.append_message(role, sentence["value"])
    conversations.append(conv.get_prompt(system_message))
    rank0_print((f"default prompt: {conversations}"))


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    system_setting: Optional[str] = None,
    data_type: str = "qlamma"
) -> Dict:
    if data_type == "pretrain":
        return preprocess_pretrain_batch(sources, tokenizer, system_message=system_setting, has_image=has_image)
    if data_type == "vision":
        return preprocess_plain(sources, tokenizer)
    else: 
        return preprocess_qwen_batch(sources, tokenizer, system_message=system_setting, has_image=has_image)



class BigPretainDataset(Dataset):
    def __init__(self, data_path: str,tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):
        super(BigPretainDataset, self).__init__()
        self.file_info = []  # To store file paths and their corresponding offsets
        self.tokenizer = tokenizer
        self.data_args = data_args

        for did,d in enumerate(data_path.split(":")):
            print (d)
            if d.endswith(".jsonl"):
                offsets = self._precompute_offsets(d)
                self.file_info.append({
                    'path': d,
                    'offsets': offsets,
                    'length': len(offsets),
                    'dataset_id': did
                })
    def _load_image(self, image_path):
        """Load an image from the given path, or return a blank white image on error."""
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            #print(f"Error loading image {image_path}: {e}")
            # Create a blank white image of size 336x336 pixels
            image = Image.new('RGB', (448, 448), color='white')
        return image
    def _precompute_offsets(self, file_path):
        """Precompute the byte offsets of each line in a file."""
        offsets = []
        with open(file_path, 'r') as f:
            while True:
                offset = f.tell()
                if not f.readline():
                    break  # End of file
                offsets.append(offset)
        return offsets

    def __len__(self):
        # Total length is the sum of the lengths of all files
        return sum(file['length'] for file in self.file_info)

    def __getitem__(self, idx):
        import time
        start_time = time.perf_counter()
        # Determine which file and what line number `idx` corresponds to
        for file in self.file_info:
            if idx < file['length']:
                break
            idx -= file['length']  # Move index to the correct file's range

        # Now `file` is the target file and `idx` is the line number within that file
        image = None
        with open(file['path'], 'r') as f:
            f.seek(file['offsets'][idx])
            line = f.readline()
            data = json.loads(line)
        dataset_id = file['dataset_id']
        sources = data    
        if isinstance(idx, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        image_time = time.perf_counter()
        if 'image' in sources[0] or 'images' in sources[0]:
            if 'image' in sources[0]:
                image_file = data['image']
            else:
                image_file = [x["image"] for x in data['images']]

            image_folder = self.data_args.image_folder.split(":")
            if len(image_folder) == 1: 
                image_folder = image_folder[0] 
            else: 
                image_folder =image_folder[dataset_id] 
 
            processor = self.data_args.image_processor
            conversations = data["conversations"]
            is_image = True
            if isinstance(image_file, list) and len(image_file) == 0:
                is_image = False
            elif isinstance(image_file, list) and len(image_file) > 0:
                try:
                    images = []
                    for i_file in image_file:
                        image_path = os.path.join(image_folder, i_file)
                        image = Image.open(image_path).convert("RGB")
                        images.append(image)
                except Exception as e:
                    # Handle the error (e.g., log, skip problematic image, etc.)
                    images = []
                    sources = [[remove_value(sentence['value'], -200) for sentence in source] for source in sources]

                                         
            if is_image:
                n_image = []
                if self.data_args.image_aspect_ratio == 'pad':
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
                    
                    images = [expand2square(image, tuple(int(x*255) for x in processor.image_mean)) for image in images]
                    try:
                        image = [processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in images]
                    except Exception as e:
                        print (e)
                        images = []
                        image = None
                        print (sources)
                        for source in sources:
                            del source["images"]
                            for sentence in source["conversations"]:
                                sentence["value"] = remove_value(sentence['value'], -200)
                        print (sources)
                        if "images" in data:
                            del data["images"]
                        if "image" in data:
                            del data["image"]

                elif self.data_args.image_aspect_ratio == 'anyres':
                    image = []
                    for img in images:
                        t, best_res = process_anyres_image(img,processor, self.data_args.image_grid_pinpoints, return_type_list=True, return_best_res=True)
                        image.extend(t)
                        n_image.append((len(t),best_res))
                else:
                    image = [processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in images]
                    
                sources = preprocess_multimodal(
                    copy.deepcopy([e["conversations"] for e in sources]),
                    self.data_args, n_image)
            else:
                sources = copy.deepcopy([e["conversations"] for e in sources])
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess(
            sources,
            self.tokenizer,
            system_setting=data.get("system_setting"),
            has_image=('image' in data or 'images' in data),
            data_type=self.data_args.data_type)
        image_end_time = time.perf_counter()

        if image:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            try:
                crop_size = self.data_args.image_processor.crop_size
            except:
                crop_size = {"height":self.data_args.image_size, "width":self.data_args.image_size}

            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict

class LazySupervisedPretrainDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedPretrainDataset, self).__init__()

        logging.warning("Loading data...")
        list_data_dict = [] 
        index = 0
        for idx,d in enumerate(data_path.split(":")): 
            if d.endswith(".json"): 
                print (d) 
                j = 0
                local_data = []
                for t in json.load(open(d)): 
                    t["dataset_id"] = idx 
                    t["id"] = index 
                    index+=1
                    local_data.append(t) 
                    if len(data_args.data_ratio) > 0:
                        if data_args.data_ratio[idx] < j:
                            break
                    j+=1
                list_data_dict.extend(local_data) 
            else: 
                f = open(d, "r") 
                print (d) 
                local_data = []
                j = 0
                stop = False
                while True: 
                    lines = f.readlines(1000000) 
                    if not lines: 
                        break 
                    for i, line in enumerate(lines): 
                        t = json.loads(line) 
                        t["id"] = index 
                        index+=1
                        t["dataset_id"] = idx 
                        local_data.append(t) 
                        if len(data_args.data_ratio) > 0:
                            if data_args.data_ratio[idx] < j:
                                stop = True
                                break
                        j+=1
                    if stop:
                        break
                list_data_dict.extend(local_data) 

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(str(conv['value']).split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list 

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        num_patches = 0
        if 'image' in sources[0] or 'images' in sources[0]:
            if 'image' in sources[0]:
                image_file = [self.list_data_dict[i]['image']]
            else:
                image_file = [x["image"] for x in self.list_data_dict[i]['images']]

            dataset_id =  self.list_data_dict[i]['dataset_id']
            image_folder = self.data_args.image_folder.split(":")
            if len(image_folder) == 1: 
                image_folder = image_folder[0] 
            else: 
                image_folder =image_folder[dataset_id] 
 
            processor = self.data_args.image_processor
            conversations = self.list_data_dict[i]["conversations"]
            is_image = True
            if isinstance(image_file, list) and len(image_file) == 0:
                is_image = False
            elif isinstance(image_file, list) and len(image_file) > 0:
                images = [Image.open(os.path.join(image_folder, i_file)).convert('RGB') for i_file in image_file]
                                         
            if is_image:
                n_image = []
                if self.data_args.image_aspect_ratio == 'pad':
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
                    
                    images = [expand2square(image, tuple(int(x*255) for x in processor.image_mean)) for image in images]
                    image = [processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in images]
                elif self.data_args.image_aspect_ratio == 'anyres':
                    image = []
                    for img in images:
                        t, best_res = process_anyres_image(img,processor, self.data_args.image_grid_pinpoints, return_type_list=True, return_best_res=True)
                        image.extend(t)
                        n_image.append((len(t),best_res))
                elif self.data_args.image_aspect_ratio == 'dynamic':
                    image = []
                    for img in images:
                        t, best_res = process_dynamic_image(img, processor, max_num=self.data_args.max_num, image_size=self.data_args.image_size, return_type_list=True, return_best_res=True)
                        image.extend(t)
                        n_image.append((len(t),best_res))
                else:
                    image = [processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in images]
                    
                sources = preprocess_multimodal(
                    copy.deepcopy([e["conversations"] for e in sources]),
                    self.data_args,n_image)
            else:
                sources = copy.deepcopy([e["conversations"] for e in sources])
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess(
            sources,
            self.tokenizer,
            system_setting=self.list_data_dict[i].get("system_setting"),
            has_image=('image' in self.list_data_dict[i] or "images" in self.list_data_dict[i] ),
            data_type=self.data_args.data_type)
        
        if isinstance(i, int):
            for key, value in data_dict.items():
                data_dict[key] = value[0]
  
        if 'image' in self.list_data_dict[i] or 'images' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            try:
                crop_size = self.data_args.image_processor.crop_size
            except:
                crop_size = {"height":self.data_args.image_size, "width":self.data_args.image_size}

            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            # adapt to multi-image 
            new_images = []
            for image in images:
                if type(image) is list:
                    for i in image:
                        new_images.append(i)
                else:
                    new_images.append(image)
            images = new_images
            batch['images'] = images
        return batch


@dataclass
class DataCollatorForPretrainingSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            # adapt to multi-image 
            new_images = []
            for image in images:
                if type(image) is list:
                    for i in image:
                        new_images.append(i)
                else:
                    new_images.append(image)
            images = new_images
            batch['images'] = images
        
        return batch


@dataclass
class DataCollatorForPretrainingSupervisedDataset2(object):

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            # adapt to multi-image 
            new_images = []
            for image in images:
                if type(image) is list:
                    for i in image:
                        new_images.append(i)
                else:
                    new_images.append(image)
            images = new_images
            batch['images'] = images
        
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if data_args.data_type == "pretrain":
        train_dataset = BigPretainDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)

        
        data_collator = DataCollatorForPretrainingSupervisedDataset(tokenizer=tokenizer)
    else:
        train_dataset = LazySupervisedPretrainDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def train():
    
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {"rotary_type": model_args.rotary_type}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:        
        if 'qwen' in model_args.model_name_or_path.lower() and not 'a14b' in model_args.model_name_or_path.lower():
            model = OmChatQwen2ForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    delay_load=not "omchat" in model_args.model_name_or_path.lower(),
                    **bnb_model_from_pretrained_args
                )
        elif 'qwen' in model_args.model_name_or_path.lower() and 'a14b' in model_args.model_name_or_path.lower():
            model = OmChatQwen2MoeForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    delay_load=not "omchat" in model_args.model_name_or_path.lower(),
                    **bnb_model_from_pretrained_args
                )
        else:
            model = OmChatLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                max_position_embeddings=model_args.max_position_embeddings,
                rope_theta=model_args.rope_theta,
                use_flash_attention_2=training_args.use_flash_attention_2, ignore_mismatched_sizes=False,
                **bnb_model_from_pretrained_args
        )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            use_flash_attention_2=training_args.use_flash_attention_2,
            **bnb_model_from_pretrained_args
        )

    model.config.use_cache = False
    model.config.training = True
    model.config.mlp_smoe = model_args.mlp_smoe
    model.config.clip_smoe = model_args.clip_smoe
    model.config.balance_loss_coef = model_args.balance_loss_coef
    model.config.router_z_loss_coef = model_args.router_z_loss_coef

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if 'qwen' in model_args.model_name_or_path.lower() and '2' in model_args.model_name_or_path.lower():
        tokenizer.add_special_tokens({'unk_token': '<|extra_0|>'})
    
    tokenizer.pad_token = tokenizer.unk_token
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model_args.is_train = True                
        
        model.get_model().initialize_vision_modules(model_args=model_args, data_args=data_args)
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
                
        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter

        #if True:
        if data_args.data_type == "pretrain":
            model.requires_grad_(True) 
            for n,p in model.get_model().vision_tower.named_parameters():                
                p.requires_grad = True
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
            if len(model_args.trainable_layers) == 0:
                model.get_model().vision_tower.requires_grad_(False)
            elif len(model_args.trainable_layers) == 1 and model_args.trainable_layers[0] == -1:
                model.get_model().vision_tower.requires_grad_(True)
            else:
                for n,p in model.get_model().vision_tower.named_parameters():                
                    if any(f"{layer_num}" in n for layer_num in model_args.trainable_layers):
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
 
            if len(model_args.trainable_layers) == 0:
                model.get_model().vision_tower.requires_grad_(False)
            else:
                for n,p in model.get_model().vision_tower.named_parameters():                                         
                    if any(f"{layer_num}" in n for layer_num in model_args.trainable_layers):
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
        
        elif data_args.data_type == "vision":
            model.requires_grad_(False) 
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
            if len(model_args.trainable_layers) == 0:
                model.get_model().vision_tower.requires_grad_(False)
            elif len(model_args.trainable_layers) == 1 and model_args.trainable_layers[0] == -1:
                model.get_model().vision_tower.requires_grad_(True)
            else:
                for n,p in model.get_model().vision_tower.named_parameters():                
                    if any(f"{layer_num}" in n for layer_num in model_args.trainable_layers):
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
        else:
            model.requires_grad_(True) 
            model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
 
            model.get_model().vision_tower.requires_grad_(False)
            if len(model_args.trainable_layers) == 0:
                model.get_model().vision_tower.requires_grad_(False)
            elif len(model_args.trainable_layers) == 1 and model_args.trainable_layers[0] == -1:
                model.get_model().vision_tower.requires_grad_(True)
            else:
                for n,p in model.get_model().vision_tower.named_parameters():                
                    if any(f"{layer_num}" in n for layer_num in model_args.trainable_layers):
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
 
        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    

    if training_args.trainer == "OmChatTrainer":
        data_module = make_supervised_data_module(tokenizer=tokenizer,
                                                  data_args=data_args)
        trainer = OmChatTrainer(model=model,
                        tokenizer=tokenizer,
                        args=training_args,
                        **data_module)
        
    else:
        raise ValueError(f"trainer should be selected from OmChatTrainer and OmChatDPOTrainer ")
    not_frozen = []
    for name, param in model.named_parameters():
        status = 'frozen' if not param.requires_grad else 'not frozen'
        if status == 'not frozen':
            n = split_and_add_number(name)
            if not n in not_frozen:
                not_frozen.append(n)

    for name in not_frozen:
        print(f'{name} is not frozen')

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
