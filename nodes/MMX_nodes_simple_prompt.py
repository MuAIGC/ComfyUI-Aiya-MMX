# ~/ComfyUI/custom_nodes/Aiya_mmx/nodes/MMX-nodes_simple_prompt.py
"""
💕 哎呀✦MMX 简易提示词 & 分辨率节点
"""
from __future__ import annotations
import torch
from ..register import register_node


class MMXSimplePromptResolution:
    """
    简单的提示词和分辨率设置节点
    提供常见尺寸比例选择，支持手动输入数值，输出空潜在向量
    """

    DESCRIPTION = (
        "💕 哎呀✦一步到位设置提示词 + 分辨率 + 空潜变量\n\n"
        "使用方法：\n"
        "1. 写好提示词，选好比例（或 Custom 手动填宽高）\n"
        "2. 宽高会自动对齐到 8 的倍数，符合潜模型要求\n"
        "3. batch_size 可一次生成多张，省显存就选 1\n\n"
        "比例清单：\n"
        "• 1:1 / 3:4 / 4:3 / 2:3 / 3:2 / 9:16 / 16:9\n"
        "• 选 Custom 可完全手动控制宽高\n\n"
        "输出：\n"
        "• prompt: 直接接 KSampler\n"
        "• width/height: 接任意需要像素的节点\n"
        "• latent: 空潜变量，直接喂给 KSampler\n\n"
        "English:\n"
        "Quick prompt & resolution picker. "
        "Aspect ratios auto-lock to 8-multiple. "
        "Outputs prompt, W/H, and empty latent ready for KSampler."
    )

    @classmethod
    def INPUT_TYPES(cls):
        aspect_ratios = [
            "Custom",
            "1:1 (Square)",
            "3:4 (Portrait)",
            "4:3 (Landscape)",
            "2:3 (Portrait)",
            "3:2 (Landscape)",
            "9:16 (Mobile)",
            "16:9 (Widescreen)",
        ]

        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "a beautiful landscape, masterpiece, best quality",
                    "multiline": True,
                    "placeholder": "Enter your prompt here..."
                }),
                "aspect_ratio": (aspect_ratios,),
                "width": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 8
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 8
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "LATENT")
    RETURN_NAMES = ("prompt", "width", "height", "latent")
    FUNCTION = "process_inputs"
    CATEGORY = "哎呀✦MMX/文本"
    OUTPUT_NODE = False

    def process_inputs(self, prompt, aspect_ratio, width, height, batch_size=1):
        if aspect_ratio != "Custom":
            ratio_map = {
                "1:1 (Square)": (1, 1),
                "3:4 (Portrait)": (3, 4),
                "4:3 (Landscape)": (4, 3),
                "2:3 (Portrait)": (2, 3),
                "3:2 (Landscape)": (3, 2),
                "9:16 (Mobile)": (9, 16),
                "16:9 (Widescreen)": (16, 9),
            }
            if aspect_ratio in ratio_map:
                ratio_w, ratio_h = ratio_map[aspect_ratio]
                height = int(width * ratio_h / ratio_w)
                height = (height // 8) * 8

        width = (width // 8) * 8
        height = (height // 8) * 8
        width = max(64, width)
        height = max(64, height)

        latent_width = width // 8
        latent_height = height // 8
        latent = torch.zeros([batch_size, 4, latent_height, latent_width])

        return (prompt, width, height, {"samples": latent})


# ---------- 注册 ----------
register_node(MMXSimplePromptResolution, "简易提示词&分辨率")
