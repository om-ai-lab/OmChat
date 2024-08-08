import torch
import torch.nn as nn
from omchat.model.multimodal_encoder.base_encoder import AbsVisionTower
from transformers import AutoModel, CLIPImageProcessor, CLIPVisionConfig, AutoConfig
from .intern_vit_6b.configuration_intern_vit import InternVisionConfig
from .intern_vit_6b.modeling_intern_vit import InternVisionModel
from accelerate import Accelerator

class InternVITVisionTower(AbsVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = InternVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, is_train=False):
        crop_size = 448 if "448" in self.vision_tower_name else 336
        self.image_processor = CLIPImageProcessor(
                crop_size=crop_size, do_center_crop=True, do_normalize=True, do_resize=True,
                image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225], size=crop_size
            )
        #self.vision_tower = InternVisionModel.from_pretrained(self.vision_tower_name,torch_dtype=torch.bfloat16)
        self.vision_tower = InternVisionModel(InternVisionConfig())
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features
    
    def forward(self, images):        
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=torch.float16), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return torch.float
        #return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

