# ~/ComfyUI/custom_nodes/Aiya_mmx/nodes/img_tools.py
from __future__ import annotations
import os
import json
import uuid
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import folder_paths
from ..register import register_node

# --------------------------------------------------
#  1. 通用批量收图器  ImageBatchCollector_mmx
# --------------------------------------------------
class ImageBatchCollector_mmx:
    """
    将多个上游 IMAGE 输出收集为一张 batch 大图，
    下游可接 SaveImage / SaveImageGrid 等节点一次性保存。
    默认 9 个插口，需要更多请改 MAX_SLOTS。
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
        images = [
            kwargs[f"image_{i}"]
            for i in range(1, self.MAX_SLOTS + 1)
            if kwargs.get(f"image_{i}") is not None
        ]
        if not images:
            raise RuntimeError("ImageBatchCollector_mmx: 未收到任何图片输入！")
        batch = torch.cat(images, dim=0)
        return (batch,)

# --------------------------------------------------
#  2. 一键保存 JPG  save2JPG_mmx
# --------------------------------------------------
class save2JPG_mmx:
    DESCRIPTION = (
        "🖼 一键保存 JPG 并可选附加提示词文本\n\n"
        "参数说明：\n"
        "• optimize  — 压缩优化，文件更小，画质无损，耗时略增（默认开）\n"
        "• progressive — 渐进式 JPG，网页大图加载\"由模糊到清晰\"，文件稍大，老设备可能不兼容（默认关）\n"
        "• save_prompt_as_txt — 同步生成同名 *_prompt.txt，记录当时提示词，方便后期归档（默认开）"
    )

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "quality": ("INT", {
                    "default": 95,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "slider"
                }),
                "optimize": ("BOOLEAN", {"default": True}),
                "progressive": ("BOOLEAN", {"default": False}),
                "save_prompt_as_txt": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt_text", "jpg_path")
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "哎呀✦MMX/图像"

    def save_images(self,
                    images,
                    filename_prefix="ComfyUI",
                    quality=95,
                    optimize=True,
                    progressive=False,
                    save_prompt_as_txt=True,
                    prompt=None,
                    extra_pnginfo=None):
        # 日期变量替换
        from ..date_variable import replace_date_vars
        filename_prefix = replace_date_vars(filename_prefix)

        os.makedirs(self.output_dir, exist_ok=True)
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])

        prompt_text = self._extract_prompt_text(prompt)
        saved_paths = []
        results = []

        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            file = f"{filename}_{counter:05}_.jpg"
            save_path = os.path.join(full_output_folder, file)
            img.save(save_path, format='JPEG', quality=quality,
                     optimize=optimize, progressive=progressive)
            saved_paths.append(save_path)

            if save_prompt_as_txt:
                txt_path = save_path.replace(".jpg", "_prompt.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(prompt_text)

            results.append({"filename": file,
                           "subfolder": subfolder, "type": self.type})
            counter += 1

        return {"ui": {"images": results},
                "result": (prompt_text, saved_paths[0] if saved_paths else "")}

    def _extract_prompt_text(self, prompt):
        if not isinstance(prompt, dict):
            return ""
        texts = []
        for node in prompt.values():
            if isinstance(node, dict) and isinstance(node.get("inputs"), dict):
                t = node["inputs"].get("prompt")
                if isinstance(t, str):
                    texts.append(t.strip())
        return "\n".join(texts)

# --------------------------------------------------
#  3. 路径读图  LoadImageFromPath_mmx
# --------------------------------------------------
CACHE_DIR = Path(folder_paths.get_output_directory()) / "Aiya/Aiya_path"

class LoadImageFromPath_mmx:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load"
    CATEGORY = "哎呀✦MMX/图像"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                "path": ("STRING", {"default": "", "multiline": False}),
                "cache_name": ("STRING", {"default": "default", "multiline": False})
        }}

    def load(self, path, cache_name):
        from ..date_variable import replace_date_vars

        path = path.strip()
        cache_name = cache_name.strip() or "default"
        path_file = CACHE_DIR / f"{cache_name}.path"

        # 1. 空输入 → 读缓存
        if not path:
            if path_file.exists():
                path = path_file.read_text(encoding="utf-8").strip()
            if not path:
                raise RuntimeError(f"LoadImageFromPath_mmx: 缓存「{cache_name}」为空！")
        # 2. 非空输入 → 写缓存
        else:
            path = replace_date_vars(path)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path_file.write_text(path, encoding="utf-8")

        # 3. 加载
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"LoadImageFromPath_mmx: 文件不存在 → {path}")

        img = Image.open(path).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        rgb = torch.from_numpy(img_np).unsqueeze(0)
        return (rgb,)

# --------------------------------------------------
#  4. 图像等分切割  ImageSplitGrid_mmx
# --------------------------------------------------
class ImageSplitGrid_mmx:
    """
    将图像按网格等分切割，支持 1×1 到 3×3 共9种输出组合
    宽切分数 × 高切分数 = 输出图片数量（最大9张）
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width_split": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 3,
                    "step": 1,
                    "display": "number",
                    "label": "宽度切分数"
                }),
                "height_split": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 3,
                    "step": 1,
                    "display": "number", 
                    "label": "高度切分数"
                }),
            }
        }

    RETURN_TYPES = tuple(["IMAGE"] * 9)
    RETURN_NAMES = tuple([f"image_{i}" for i in range(1, 10)])
    FUNCTION = "split_image"
    CATEGORY = "哎呀✦MMX/图像"

    def split_image(self, image, width_split, height_split):
        if width_split < 1 or width_split > 3 or height_split < 1 or height_split > 3:
            raise ValueError("ImageSplitGrid_mmx: 切分数必须在 1-3 之间")

        total_parts = width_split * height_split
        if total_parts > 9:
            raise ValueError(f"ImageSplitGrid_mmx: 总切割数 {total_parts} 超过最大值9")

        if len(image.shape) == 4:
            batch_size, height, width, channels = image.shape
            if batch_size != 1:
                raise ValueError("ImageSplitGrid_mmx: 暂不支持 batch > 1 的输入")
            img_tensor = image[0]
        else:
            height, width, channels = image.shape
            img_tensor = image

        part_width = width // width_split
        part_height = height // height_split

        width_positions = []
        height_positions = []

        for i in range(width_split):
            start = i * part_width
            if i == width_split - 1:
                end = width
            else:
                end = (i + 1) * part_width
            width_positions.append((start, end))

        for i in range(height_split):
            start = i * part_height
            if i == height_split - 1:
                end = height
            else:
                end = (i + 1) * part_height
            height_positions.append((start, end))

        parts = []
        for h_idx in range(height_split):
            for w_idx in range(width_split):
                h_start, h_end = height_positions[h_idx]
                w_start, w_end = width_positions[w_idx]

                part = img_tensor[h_start:h_end, w_start:w_end, :]
                part = part.unsqueeze(0)
                parts.append(part)

        result = []
        for i in range(9):
            if i < len(parts):
                result.append(parts[i])
            else:
                empty = torch.zeros((1, 1, 1, 3), dtype=img_tensor.dtype, device=img_tensor.device)
                result.append(empty)

        return tuple(result)

# --------------------------------------------------
#  统一注册
# --------------------------------------------------
register_node(ImageBatchCollector_mmx, "ImageBatchCollector_mmx")
register_node(save2JPG_mmx, "save2JPG_mmx")
register_node(LoadImageFromPath_mmx, "LoadImageFromPath_mmx")
register_node(ImageSplitGrid_mmx, "ImageSplitGrid_mmx")
