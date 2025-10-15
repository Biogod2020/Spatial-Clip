# /path/to/your/open_clip/src/open_clip_train/spatial_loss.py
import torch
import torch.distributed as dist
import torch.nn.functional as F

from open_clip import ClipLoss
from open_clip.loss import gather_features


class GlobalMappingMultiPositiveClipLoss(ClipLoss):
    """
    空间多正样本版 CLIP 损失：
      - 支持把邻居权重映射到“全局”logits 上（跨 GPU）。
      - 温度 s 的上限采用直通估计（STE）：前向裁剪、反向对原值求梯度。
      - 可选把 logits 转为 float32 做 log_softmax，提升数值稳定性（混精下建议开）。
      - 可选温度正则：把 E_{p_s}[z] 拉向 <q,z>，避免 s 过热。
    返回：
      - 当 output_dict=True，仅返回 {"contrastive_loss": total_loss}
        （避免你的训练循环把指标也相加进总损失）
    """

    def __init__(
        self,
        *args,
        cap_logit_scale: float = None,     # 有效温度上限 s_max（对 exp 后的 s 生效）
        temp_reg_weight: float = 0.0,      # 温度正则权重（0 关闭）
        float32_logits: bool = False,      # 在 log-softmax 前把 logits 转为 fp32
        neighbor_alpha_scale: float = 1.0, # 邻居权重整体缩放
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.cap_logit_scale = cap_logit_scale
        self.temp_reg_weight = temp_reg_weight
        self.float32_logits = float32_logits
        self.neighbor_alpha_scale = neighbor_alpha_scale

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        image_tile_ids: torch.Tensor,
        text_tile_ids: torch.Tensor,
        neighbor_tile_ids: torch.Tensor,
        neighbor_alphas: torch.Tensor,
        logit_scale: torch.Tensor,         # 注意：这里接收的是 exp 后的 s（标量张量或可广播）
        logit_bias: torch.Tensor = None,
        output_dict: bool = False,
    ):
        device = image_features.device

        # ===== 1) 跨 GPU gather（与 OpenCLIP 的实现保持一致）=====
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features=image_features,
                text_features=text_features,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )
            # 同步 tile_ids 用来定位全局列号
            gathered_img_ids = [torch.empty_like(image_tile_ids) for _ in range(self.world_size)]
            gathered_txt_ids = [torch.empty_like(text_tile_ids) for _ in range(self.world_size)]
            dist.all_gather(gathered_img_ids, image_tile_ids)
            dist.all_gather(gathered_txt_ids, text_tile_ids)
            all_image_tile_ids = torch.cat(gathered_img_ids)
            all_text_tile_ids = torch.cat(gathered_txt_ids)
        else:
            all_image_features, all_text_features = image_features, text_features
            all_image_tile_ids, all_text_tile_ids = image_tile_ids, text_tile_ids

        # ===== 2) 有效温度 s 的“软上限”（直通估计，前向裁剪、反向保梯度）=====
        s_eff = logit_scale
        if self.cap_logit_scale is not None:
            s_clipped = torch.clamp(logit_scale, max=self.cap_logit_scale)
            # STE： (s_clipped - s).detach() 不参与反传，等价于前向用裁剪值，反向对原 s 求梯度
            s_eff = logit_scale + (s_clipped - logit_scale).detach()

        # ===== 3) 相似度 -> logits（可选转 fp32 再做 log-softmax 更稳）=====
        z_i_t = image_features @ all_text_features.T   # [B, G]
        z_t_i = text_features  @ all_image_features.T  # [B, G]
        logits_per_image = s_eff * z_i_t
        logits_per_text  = s_eff * z_t_i
        if logit_bias is not None:
            logits_per_image = logits_per_image + logit_bias
            logits_per_text  = logits_per_text  + logit_bias
        if self.float32_logits:
            logits_per_image = logits_per_image.float()
            logits_per_text  = logits_per_text.float()

        # ===== 4) 构造软标签（把邻居权重映射到“全局列”）=====
        B_local = image_features.size(0)
        N_global = all_image_features.size(0)

        # 建立 tile_id -> 全局列号 的查找表
        txt_id_to_idx = {tid.item(): i for i, tid in enumerate(all_text_tile_ids)}
        img_id_to_idx = {tid.item(): i for i, tid in enumerate(all_image_tile_ids)}

        # 对角（同位）正样本的全局列号（local_loss 语义：行是本地样本，列是全局）
        ground_truth = torch.arange(B_local, device=device) + B_local * self.rank

        # 初始化 one-hot，再叠加邻居权重
        labels_i_t = torch.zeros(B_local, N_global, device=device)
        labels_t_i = torch.zeros(B_local, N_global, device=device)
        labels_i_t.scatter_(1, ground_truth.unsqueeze(1), 1.0)
        labels_t_i.scatter_(1, ground_truth.unsqueeze(1), 1.0)

        # 邻居整体缩放 & 非负裁剪
        if self.neighbor_alpha_scale != 1.0:
            neighbor_alphas = neighbor_alphas * self.neighbor_alpha_scale
        neighbor_alphas = neighbor_alphas.clamp_min(0)

        # 将邻居权重加到对应全局列
        for i in range(B_local):
            for nbr_id, alpha in zip(neighbor_tile_ids[i], neighbor_alphas[i]):
                if alpha.item() <= 0:
                    continue
                # 图像->文本（行是 image，列是 all_text）
                col_txt = txt_id_to_idx.get(int(nbr_id.item()))
                if col_txt is not None:
                    labels_i_t[i, col_txt] += float(alpha.item())
                # 文本->图像（行是 text，列是 all_image）
                col_img = img_id_to_idx.get(int(nbr_id.item()))
                if col_img is not None:
                    labels_t_i[i, col_img] += float(alpha.item())

        # 归一化成分布
        labels_i_t = F.normalize(labels_i_t, p=1, dim=1)
        labels_t_i = F.normalize(labels_t_i, p=1, dim=1)

        # ===== 5) 主损失（对称 KL：-<q, log p>）=====
        loss_i = -torch.sum(F.log_softmax(logits_per_image, dim=1) * labels_i_t, dim=1).mean()
        loss_t = -torch.sum(F.log_softmax(logits_per_text , dim=1) * labels_t_i, dim=1).mean()
        total_loss = 0.5 * (loss_i + loss_t)

        # ===== 6) 可选：温度正则（把 E_{p_s}[z] 拉近 E_q[z]）=====
        if self.temp_reg_weight and self.temp_reg_weight > 0:
            # 注意这里用“未缩放”的 z 来衡量温度引起的“过拟合软目标”的偏移
            p_i = F.softmax(logits_per_image, dim=1)
            p_t = F.softmax(logits_per_text , dim=1)

            Ez_p_i = (p_i * z_i_t).sum(dim=1).mean()
            Ez_q_i = (labels_i_t * z_i_t).sum(dim=1).mean()
            Ez_p_t = (p_t * z_t_i).sum(dim=1).mean()
            Ez_q_t = (labels_t_i * z_t_i).sum(dim=1).mean()

            gap = 0.5 * ((Ez_p_i - Ez_q_i) + (Ez_p_t - Ez_q_t))
            total_loss = total_loss + self.temp_reg_weight * (gap ** 2)

        # ===== 7) 输出 =====
        if output_dict:
            # 只返回真正参与反传的损失，避免被训练循环“全量相加”影响总损失
            return {"contrastive_loss": total_loss}
        return total_loss
