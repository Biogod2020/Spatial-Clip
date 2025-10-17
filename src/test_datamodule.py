# src/test_datamodule.py (最终正确版本 v2)

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra
import torch
import open_clip
from omegaconf import DictConfig
from PIL import Image
import torchvision.transforms as T

def get_dummy_preprocess():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

# --- 核心修改在这里：加载我们自己的测试配置 ---
@hydra.main(version_base=None, config_path="../configs", config_name="test_datamodule.yaml")
def main(cfg: DictConfig) -> None:
    # 现在 cfg 就是一个干净的配置，只包含 data 和 paths
    print("--- 正在使用以下数据配置进行测试 ---")
    print(cfg.data)
    
    # 实例化 DataModule
    datamodule = hydra.utils.instantiate(cfg.data)
    
    # 模拟从模型模块传入预处理器和分词器
    datamodule.preprocess_fn = get_dummy_preprocess()
    datamodule.tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    # 运行 setup
    datamodule.setup(stage="fit")
    
    # 获取一个训练批次
    print("\n--- 正在获取一个训练批次... ---")
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    
    print("✅ 成功获取一个批次！")
    print("--- 批次内容检查 ---")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  - {key}: type={type(value)}")

    # 检查邻居填充是否正确
    k = cfg.data.k_neighbors
    b = cfg.data.batch_size
    assert batch["neighbor_tile_ids"].shape == (b, k)
    assert batch["neighbor_alphas"].shape == (b, k)
    print(f"\n✅ 邻居张量形状正确 (batch_size, k) -> ({b}, {k})")

if __name__ == "__main__":
    main()