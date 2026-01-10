from __future__ import annotations
import torch
from ..register import register_node      # 若你项目无 register，可换成 from comfyui_nodes import register_node 或自行 import

class ImageBatchCollector:
    """
    通用批量收图器
    将多个上游节点的 IMAGE 输出收集为一张 batch 大图，
    下游可接 SaveImage、SaveImageGrid 等节点一次性保存。
    前端默认 9 个插口，需要更多请改 MAX_SLOTS。
    """

    MAX_SLOTS = 9

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "collect"
    CATEGORY = "utils/batch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                f"image_{i}": ("IMAGE",)
                for i in range(1, cls.MAX_SLOTS + 1)
            }
        }

    def collect(self, **kwargs):
        # 收集所有非空 IMAGE
        images = [
            kwargs[f"image_{i}"]
            for i in range(1, self.MAX_SLOTS + 1)
            if kwargs.get(f"image_{i}") is not None
        ]
        if not images:
            raise RuntimeError("ImageBatchCollector: 未收到任何图片输入！")

        # 拼成 [N, H, W, C] 的批次张量
        batch = torch.cat(images, dim=0)
        return (batch,)

# 注册节点（把字符串显示名也中性化）
register_node(ImageBatchCollector, "ImageBatchCollector")