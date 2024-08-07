
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import Qwen2MoeConfig, Qwen2MoeModel, Qwen2MoeForCausalLM, AutoConfig, AutoModelForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..omchat_arch import OmChatMetaModel, OmChatMetaForCausalLM
import torch.distributed as dist


class OmChatQwen2MoeConfig(Qwen2MoeConfig):
    model_type = "omchat_qwen2_moe"
    rotary_type = "normal_rotary"
    multi_scale_im = None


class OmChatQwen2MoeModel(OmChatMetaModel, Qwen2MoeModel):
    config_class = OmChatQwen2MoeConfig

    def __init__(self, config: Qwen2MoeConfig):
        super(OmChatQwen2MoeModel, self).__init__(config)


class OmChatQwen2MoeForCausalLM(Qwen2MoeForCausalLM, OmChatMetaForCausalLM):
    config_class = OmChatQwen2MoeConfig

    def __init__(self, config):
        super(Qwen2MoeForCausalLM, self).__init__(config)
        self.model = OmChatQwen2MoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # import ipdb
        # ipdb.set_trace()
        # print(f'rank {dist.get_rank()}', 'before prepare_inputs_labels_for_multimodal')
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        # dist.barrier()
        # print(f'rank {dist.get_rank()}', 'after prepare_inputs_labels_for_multimodal')
        out = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        # dist.barrier()
        # print(f'rank {dist.get_rank()}', 'after LLM')
        return out


    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

AutoConfig.register("omchat_qwen2_moe", OmChatQwen2MoeConfig)
AutoModelForCausalLM.register(OmChatQwen2MoeConfig, OmChatQwen2MoeForCausalLM)
