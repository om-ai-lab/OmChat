from .clip_encoder import CLIPVisionTower
from .siglip_encoder import SiglipVisionTower
from .internVIT_encoder import InternVITVisionTower
from .internVIT300m_encoder import InternVIT300mVisionTower 


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    print(vision_tower_cfg)
    mm_projector_type =  getattr(vision_tower_cfg,'mm_projector_type')
    if "internvit-300m" in vision_tower.lower():
        return InternVIT300mVisionTower(vision_tower, args=vision_tower_cfg, **kwargs) 
    elif "internvit-6b" in vision_tower.lower():
        return InternVITVisionTower(vision_tower, args=vision_tower_cfg, **kwargs) 
    else: 
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
