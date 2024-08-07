
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM, AutoConfig, AutoModelForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..omchat_arch import OmChatMetaModel, OmChatMetaForCausalLM
import torch.distributed as dist
from omchat.model.multimodal_projector.builder import build_vision_projector


class OmChatQwen2Config(Qwen2Config):
    model_type = "omchat_qwen2"
    rotary_type = "normal_rotary"
    multi_scale_im = None


class OmChatQwen2Model(OmChatMetaModel, Qwen2Model):
    config_class = OmChatQwen2Config

    def __init__(self, config: Qwen2Config):
        super(OmChatQwen2Model, self).__init__(config)


class OmChatQwen2ForCausalLM(Qwen2ForCausalLM, OmChatMetaForCausalLM):
    config_class = OmChatQwen2Config

    def __init__(self, config, delay_load=True):
        if not hasattr(config, 'delay_load'):
            config.delay_load = delay_load
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = OmChatQwen2Model(config)
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
        return out


    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

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

AutoConfig.register("omchat_qwen2", OmChatQwen2Config)
AutoModelForCausalLM.register(OmChatQwen2Config, OmChatQwen2ForCausalLM)
