# src/models/components/losses.py
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from open_clip.loss import gather_features, ClipLoss as OpenClipLoss


class SpatialLoss(nn.Module):
    """
    CodeGuardian: This is the 'Thick Implementation' for your spatial loss.
    It is a pure nn.Module, containing only the logic for WHAT it does.
    """
    def __init__(
        self,
        local_loss: bool = False,
        gather_with_grad: bool = False,
        rank: int = 0,
        world_size: int = 1,
        use_horovod: bool = False,
        cap_logit_scale: Optional[float] = None,
        temp_reg_weight: float = 0.0,
        float32_logits: bool = False,
        neighbor_alpha_scale: float = 1.0,
    ):
        super().__init__()
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = rank
            self.world_size = world_size
        
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.use_horovod = use_horovod
        self.cap_logit_scale = cap_logit_scale
        self.temp_reg_weight = temp_reg_weight
        self.float32_logits = float32_logits
        self.neighbor_alpha_scale = neighbor_alpha_scale

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
        image_tile_ids: torch.Tensor,
        text_tile_ids: torch.Tensor,
        neighbor_tile_ids: torch.Tensor,
        neighbor_alphas: torch.Tensor,
        logit_bias: Optional[torch.Tensor] = None,
        output_dict: bool = True, # For consistency with LightningModule logging
    ) -> Dict[str, torch.Tensor]:
        device = image_features.device

        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features=image_features, text_features=text_features, 
                local_loss=self.local_loss, gather_with_grad=self.gather_with_grad,
                rank=self.rank, world_size=self.world_size, use_horovod=self.use_horovod)
            gathered_img_ids = [torch.empty_like(image_tile_ids) for _ in range(self.world_size)]
            gathered_txt_ids = [torch.empty_like(text_tile_ids) for _ in range(self.world_size)]
            dist.all_gather(gathered_img_ids, image_tile_ids)
            dist.all_gather(gathered_txt_ids, text_tile_ids)
            all_image_tile_ids = torch.cat(gathered_img_ids)
            all_text_tile_ids = torch.cat(gathered_txt_ids)
        else:
            all_image_features, all_text_features = image_features, text_features
            all_image_tile_ids, all_text_tile_ids = image_tile_ids, text_tile_ids

        s_eff = logit_scale
        if self.cap_logit_scale is not None:
            s_clipped = torch.clamp(logit_scale, max=self.cap_logit_scale)
            s_eff = logit_scale + (s_clipped - logit_scale).detach()

        z_i_t = image_features @ all_text_features.T
        z_t_i = text_features @ all_image_features.T
        logits_per_image = s_eff * z_i_t
        logits_per_text = s_eff * z_t_i

        if logit_bias is not None:
            logits_per_image += logit_bias
            logits_per_text += logit_bias
        
        if self.float32_logits:
            logits_per_image = logits_per_image.float()
            logits_per_text = logits_per_text.float()

        B_local, N_global = image_features.size(0), all_image_features.size(0)
        txt_id_to_idx = {tid.item(): i for i, tid in enumerate(all_text_tile_ids)}
        img_id_to_idx = {tid.item(): i for i, tid in enumerate(all_image_tile_ids)}
        ground_truth = torch.arange(B_local, device=device, dtype=torch.long) + B_local * self.rank
        labels_i_t = torch.zeros(B_local, N_global, device=device)
        labels_t_i = torch.zeros(B_local, N_global, device=device)
        labels_i_t.scatter_(1, ground_truth.unsqueeze(1), 1.0)
        labels_t_i.scatter_(1, ground_truth.unsqueeze(1), 1.0)

        alphas = (neighbor_alphas * self.neighbor_alpha_scale).clamp_min(0)

        for i in range(B_local):
            for nbr_id, alpha in zip(neighbor_tile_ids[i], alphas[i]):
                if alpha.item() <= 0: continue
                col_txt = txt_id_to_idx.get(int(nbr_id.item()))
                if col_txt is not None: labels_i_t[i, col_txt] += float(alpha.item())
                col_img = img_id_to_idx.get(int(nbr_id.item()))
                if col_img is not None: labels_t_i[i, col_img] += float(alpha.item())

        labels_i_t = F.normalize(labels_i_t, p=1, dim=1)
        labels_t_i = F.normalize(labels_t_i, p=1, dim=1)

        loss_i = -torch.sum(F.log_softmax(logits_per_image, dim=1) * labels_i_t, dim=1).mean()
        loss_t = -torch.sum(F.log_softmax(logits_per_text, dim=1) * labels_t_i, dim=1).mean()
        total_loss = 0.5 * (loss_i + loss_t)

        if self.temp_reg_weight > 0:
            p_i, p_t = F.softmax(logits_per_image, dim=1), F.softmax(logits_per_text, dim=1)
            Ez_p_i, Ez_q_i = (p_i * z_i_t).sum(dim=1).mean(), (labels_i_t * z_i_t).sum(dim=1).mean()
            Ez_p_t, Ez_q_t = (p_t * z_t_i).sum(dim=1).mean(), (labels_t_i * z_t_i).sum(dim=1).mean()
            gap = 0.5 * ((Ez_p_i - Ez_q_i) + (Ez_p_t - Ez_q_t))
            total_loss += self.temp_reg_weight * (gap ** 2)

        return {"contrastive_loss": total_loss}

class ClipLoss(OpenClipLoss):
    """
    CodeGuardian: 一个简单的包装器，现在拥有一个干净、明确的接口。
    它不再需要 **kwargs。
    """
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
        logit_bias: Optional[torch.Tensor] = None,
        # **kwargs,  <-- 移除这一行
    ) -> Dict[str, torch.Tensor]:
        # 确保 output_dict=False 以获取原始损失张量
        loss = super().forward(image_features, text_features, logit_scale, logit_bias, output_dict=False)
        return {"contrastive_loss": loss}