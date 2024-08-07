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

import torch
from abc import ABC, abstractmethod
from omchat.model.multimodal_encoder.builder import build_vision_tower 
from .multimodal_projector.builder import build_vision_projector
from omchat.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

class OmChatMetaModel:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_tower = build_vision_tower(config, delay_load=True) if config.mm_vision_tower else None
        self.mm_projector = build_vision_projector(config) if config.mm_vision_tower else None

    def get_vision_tower(self):
        if isinstance(self.vision_tower, list):
            return self.vision_tower[0]
        return self.vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None, data_args=None):
        vision_tower = build_vision_tower(model_args) if self.get_vision_tower() is None else self.vision_tower
        if fsdp:
            self.vision_tower = [vision_tower] if len(fsdp) > 0 else vision_tower
        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_tower.hidden_size if vision_tower else -1
        self.mm_projector = build_vision_projector(self.config, image_size=data_args.image_size, mm_projector_n_query=model_args.mm_projector_n_query)

        if model_args.pretrain_mm_mlp_adapter:
            self.load_pretrained_projector_weights(model_args.pretrain_mm_mlp_adapter)

    def load_pretrained_projector_weights(self, pretrain_mm_mlp_adapter):
        mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
        self.mm_projector.load_state_dict({k.split('mm_projector.')[1]: v for k, v in mm_projector_weights.items()})

class OmChatMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        vision_tower = self.get_vision_tower()
        image_features = vision_tower(images)
        return self.get_model().mm_projector(image_features)

    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images):
        vision_tower = self.get_vision_tower()
        if not (vision_tower and images and input_ids.shape[1] != 1):
            if past_key_values and vision_tower and images and input_ids.shape[1] == 1:
                attention_mask = self.extend_attention_mask(attention_mask, past_key_values)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        image_features = self.encode_image_batch(images)
        input_ids, labels = self.adjust_for_image_tokens(input_ids, labels, image_features)

        max_len = max(x.shape[0] for x in input_ids)
        batch_size = len(input_ids)

        input_embeds_padded, labels_padded, attention_mask, position_ids = self.pad_inputs_labels(input_ids, labels, max_len, batch_size, attention_mask, position_ids)
        
        return None, position_ids, attention_mask, past_key_values, input_embeds_padded, labels_padded

    def extend_attention_mask(self, attention_mask, past_key_values):
        target_shape = past_key_values[-1][-1].shape[-2] + 1
        return torch.cat((attention_mask, torch.ones((attention_mask.shape[0], target_shape - attention_mask.shape[1]), dtype=attention_mask.dtype, device=attention_mask.device)), dim=1)

    def encode_image_batch(self, images):
        image_idx = [idx for idx, img in enumerate(images) if img.ndim == 3]
        images_minibatch = torch.stack([images[idx] for idx in image_idx]) if image_idx else []
        image_features_minibatch = self.encode_images(images_minibatch) if images_minibatch.ndim == 4 else []
        return [image_features_minibatch[i] if i in image_idx else None for i in range(len(images))]

    def adjust_for_image_tokens(self, input_ids, labels, image_features):
        new_input_embeds, new_labels, cur_image_idx = [], [], 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = self.get_image_token_indices(cur_input_ids)
            cur_input_ids_noim, cur_labels_noim = self.get_cur_input_labels_noim(cur_input_ids, labels[batch_idx], image_token_indices)
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds_no_im = torch.split(self.get_model().embed_tokens(torch.cat(cur_input_ids_noim)), split_sizes, dim=0)

            cur_new_input_embeds, cur_new_labels = self.insert_image_features(cur_input_embeds_no_im, cur_labels_noim, image_features, cur_image_idx, num_images)

            new_input_embeds.append(torch.cat(cur_new_input_embeds))
            new_labels.append(torch.cat(cur_new_labels))

        return new_input_embeds, new_labels

    def get_image_token_indices(self, cur_input_ids):
        return [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]

    def get_cur_input_labels_noim(self, cur_input_ids, cur_labels, image_token_indices):
        cur_input_ids_noim, cur_labels_noim = [], []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
            cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])
        return cur_input_ids_noim, cur_labels_noim

    def insert_image_features(self, cur_input_embeds_no_im, cur_labels_noim, image_features, cur_image_idx, num_images):
        cur_new_input_embeds, cur_new_labels = [], []
        for i in range(num_images + 1):
            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            cur_new_labels.append(cur_labels_noim[i])
            if i < num_images:
                cur_image_features = image_features[cur_image_idx].to(self.device)
                cur_image_idx += 1
                cur_new_input_embeds.append(cur_image_features)
                cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels_noim[0].device, dtype=cur_labels_noim[0].dtype))
        return cur_new_input_embeds, cur_new_labels

    def pad_inputs_labels(self, input_embeds, labels, max_len, batch_size, attention_mask, position_ids):
        input_embeds_padded = []
        labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=labels[0].dtype, device=labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(input_embeds, labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        input_embeds_padded = torch.stack(input_embeds_padded, dim=0)
        attention_mask = attention_mask.to(dtype=attention_mask.dtype)
        return input_embeds_padded, labels_padded, attention_mask, position_ids

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        num_new_tokens = tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))
        self.initialize_new_tokens(model_args, tokenizer, num_new_tokens)

    def initialize_new_tokens(self, model_args, tokenizer, num_new_tokens):
        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

        if model_args.tune_mm_mlp_adapter:
            for p in self.get_input_embeddings().parameters():
                p.requires_grad = True
            for p in self.get_output_embeddings().parameters():
                p.requires_grad = False

        if model_args.pretrain_mm_mlp_adapter:
            self.load_pretrained_tokens(model_args.pretrain_mm_mlp_adapter, num_new_tokens)

    def load_pretrained_tokens(self, pretrain_mm_mlp_adapter, num_new_tokens):
        mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
        embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
        input_embeddings = self.get_input_embeddings().weight.data
        assert num_new_tokens == 2
        if input_embeddings.shape == embed_tokens_weight.shape:
            input_embeddings[-num_new_tokens:] = embed_tokens_weight
        elif embed_tokens_weight.shape[0] == num_new_tokens:
            input_embeddings[-num_new_tokens:] = embed_tokens_weight
        else:
            raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Number of new tokens: {num_new_tokens}.")

