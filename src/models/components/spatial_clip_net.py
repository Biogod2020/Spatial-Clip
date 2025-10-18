# src/models/components/spatial_clip_net.py

# src/models/components/spatial_clip_net.py
from dataclasses import is_dataclass, asdict
from typing import Dict, Optional, Any, Union
import torch
import torch.nn as nn
import open_clip
from omegaconf import DictConfig, ListConfig, OmegaConf

class SpatialClipNet(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: str,
        aug_cfg: Optional[Union[Dict[str, Any], DictConfig, Any]] = None,
        cache_dir: Optional[str] = None
    ):
        super().__init__()

        # Robust handling of aug_cfg from Hydra or user code
        aug_cfg_obj = None
        if aug_cfg is not None:
            if isinstance(aug_cfg, (DictConfig, ListConfig)):
                # 1) OmegaConf -> dict
                aug_dict = OmegaConf.to_container(aug_cfg, resolve=True)
                # 2) dict -> AugmentationCfg dataclass
                aug_cfg_obj = open_clip.AugmentationCfg(**aug_dict)
            elif isinstance(aug_cfg, dict):
                aug_cfg_obj = open_clip.AugmentationCfg(**aug_cfg)
            elif is_dataclass(aug_cfg):
                aug_cfg_obj = aug_cfg  # already AugmentationCfg
            else:
                raise TypeError(f"Unsupported type for aug_cfg: {type(aug_cfg)}")

        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
            aug_cfg=aug_cfg_obj,   # 传入数据类实例，兼容上游 open_clip
            cache_dir=cache_dir,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def forward(self, images: torch.Tensor, texts: torch.Tensor) -> Dict[str, torch.Tensor]:
        image_features = self.model.encode_image(images, normalize=True)
        text_features = self.model.encode_text(texts, normalize=True)
        
        return {
            "image_features": image_features,
            "text_features": text_features,
            "logit_scale": self.model.logit_scale.exp(),
            "logit_bias": getattr(self.model, 'logit_bias', None),
        }