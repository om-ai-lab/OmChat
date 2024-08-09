from transformers import PreTrainedTokenizer
from typing import List, Tuple
from omchat.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN
)
from omchat.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from omchat.mm_utils import (
    tokenizer_image_token,
)
import torch
from omchat.mm_utils import tokenizer_image_token, process_anyres_image

def get_context(
    text: str,
    tokenizer: PreTrainedTokenizer,
    initial_prompt="You are a helpful assistant.",
    image=None, 
    image_processor=None, 
    image_grid_pinpoints=None
    ):
    if image:
        image_patches, best_res = process_anyres_image(image, image_processor, image_grid_pinpoints, True,return_best_res=True)
        n = len(image_patches)
        image_tensor = torch.stack(image_patches, dim=0)
        image_tensor = image_tensor.half().cuda()
 
        inp, context_tokens = make_context(
            tokenizer,
            "<image>\n"+"\n".join(["patch:<image>"]*(n-1)) +"\n"+ text.replace("<image>", "").strip(),
            None,
            initial_prompt,
        )
    else:
        inp, context_tokens = make_context(
            tokenizer,
            qs.replace("<image>", "").strip(),
            None,
            initial_prompt,
        )
        image_tensor = None
 
    return inp, context_tokens, image_tensor
 
def get_image_context(
    model_name, 
    image, 
    image_processor, 
    image_grid_pinpoints,
    tokenizer: PreTrainedTokenizer,
    qs: str
    ):
    image_patches, best_res = process_anyres_image(image, image_processor, image_grid_pinpoints, True,return_best_res=True)
    n = len(image_patches)
    image_tensor = torch.stack(image_patches, dim=0)
    image_tensor = image_tensor.half().cuda()
 
    inp, context_tokens = make_context(
        tokenizer,
        "<image>\n"+"\n".join(["patch:<image>"]*(n-1)) +"\n"+ qs.replace("<image>", "").strip(),
        None,
        "You are a helpful assistant.",
    )
    return inp, context_tokens, image_tensor
 
def make_context(
    tokenizer: PreTrainedTokenizer,
    query: str,
    history: List[Tuple[str, str]] = None,
    system: str = "",
    max_window_size: int = 6144,
    chat_format: str = "chatml",
):
    if history is None:
        history = []

    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [151644]
        im_end_tokens = [151645]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            if DEFAULT_IMAGE_TOKEN in content:
                return f"{role}\n{content}", tokenizer.encode(
                    role
                ) + nl_tokens + tokenizer_image_token(
                    content, tokenizer, IMAGE_TOKEN_INDEX
                )
            else:
                return f"{role}\n{content}", tokenizer.encode(
                    role
                ) + nl_tokens + tokenizer.encode(content)

        def _tokenize_str2(role, content):
            return f"{role}\n{content}", tokenizer.encode(
                role,
            ) + nl_tokens + tokenizer.encode(content)

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            response_text, response_tokens_part = _tokenize_str(
                "assistant", turn_response
            )
            response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

            next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
            prev_chat = (
                f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
            )

            current_context_size = (
                len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            )
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (
            nl_tokens
            + im_start_tokens
            + _tokenize_str("user", query)[1]
            + im_end_tokens
            + nl_tokens
            + im_start_tokens
            + tokenizer.encode("assistant")
            + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")

    return raw_text, context_tokens
