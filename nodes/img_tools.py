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
    MAX_SLOTS = 9
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "collect"
    CATEGORY = "utils/batch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {f"image_{i}": ("IMAGE",) for i in range(1, cls.MAX_SLOTS + 1)}
        }

    def collect(self, **kwargs):
        images = [kwargs[f"image_{i}"] for i in range(1, self.MAX_SLOTS + 1) if kwargs.get(f"image_{i}") is not None]
        if not images:
            raise RuntimeError("ImageBatchCollector_mmx: 未收到任何图片输入！")
        base_h, base_w = images[0].shape[1], images[0].shape[2]
        resized = []
        for img in images:
            if img.shape[1] != base_h or img.shape[2] != base_w:
                img = torch.nn.functional.interpolate(img, size=(base_h, base_w), mode="bilinear", align_corners=False)
            resized.append(img)
        batch = torch.cat(resized, dim=0)
        return (batch,)

# --------------------------------------------------
#  2. 一键保存 JPG  save2JPG_mmx
# --------------------------------------------------
class save2JPG_mmx:
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
                "quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1, "display": "slider"}),
                "optimize": ("BOOLEAN", {"default": True}),
                "progressive": ("BOOLEAN", {"default": False}),
                "save_prompt_as_txt": ("BOOLEAN", {"default": True}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt_text", "jpg_path")
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "哎呀✦MMX/图像"

    def save_images(self, images, filename_prefix="ComfyUI", quality=95, optimize=True, progressive=False,
                    save_prompt_as_txt=True, prompt=None, extra_pnginfo=None):
        from ..date_variable import replace_date_vars
        filename_prefix = replace_date_vars(filename_prefix)
        os.makedirs(self.output_dir, exist_ok=True)
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        prompt_text = self._extract_prompt_text(prompt)
        saved_paths, results = [], []
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            file = f"{filename}_{counter:05}_.jpg"
            save_path = os.path.join(full_output_folder, file)
            img.save(save_path, format='JPEG', quality=quality, optimize=optimize, progressive=progressive)
            saved_paths.append(save_path)
            if save_prompt_as_txt:
                with open(save_path.replace(".jpg", "_prompt.txt"), "w", encoding="utf-8") as f:
                    f.write(prompt_text)
            results.append({"filename": file, "subfolder": subfolder, "type": self.type})
            counter += 1
        return {"ui": {"images": results}, "result": (prompt_text, saved_paths[0] if saved_paths else "")}

    def _extract_prompt_text(self, prompt):
        if not isinstance(prompt, dict):
            return ""
        texts = []
        for node in prompt.values():
            if isinstance(node, dict) and isinstance(node.get("inputs"), dict):
                t = node["inputs"].get("prompt")
                # 只处理字符串，跳过 list / None
                if isinstance(t, str) and t.strip():
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
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "", "multiline": False}),
                "cache_name": ("STRING", {"default": "default", "multiline": False}),
            },
            "optional": {"force_run": ("BOOLEAN", {"default": True})}
        }

    def load(self, path, cache_name, force_run=True):
        from ..date_variable import replace_date_vars
        path = path.strip()
        cache_name = cache_name.strip() or "default"
        path_file = CACHE_DIR / f"{cache_name}.txt"

        if path:
            path = replace_date_vars(path)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path_file.write_text(path, encoding="utf-8")

        if path_file.exists():
            path = path_file.read_text(encoding="utf-8").strip()
        if not path:
            print(f"[LoadImageFromPath_mmx] 无有效路径，返回空图 | cache={cache_name}")
            empty = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            return (empty,)

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
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width_split": ("INT", {"default": 2, "min": 1, "max": 3, "step": 1, "display": "number", "label": "宽度切分数"}),
                "height_split": ("INT", {"default": 2, "min": 1, "max": 3, "step": 1, "display": "number", "label": "高度切分数"}),
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
            if image.shape[0] != 1:
                raise ValueError("ImageSplitGrid_mmx: 暂不支持 batch > 1 的输入")
            image = image[0]
        height, width, channels = image.shape

        new_width = (width // width_split) * width_split
        new_height = (height // height_split) * height_split
        if new_width != width or new_height != height:
            image = image.permute(2, 0, 1).unsqueeze(0)
            image = torch.nn.functional.interpolate(image, size=(new_height, new_width), mode='bilinear', align_corners=False)
            image = image.squeeze(0).permute(1, 2, 0)

        part_w = new_width // width_split
        part_h = new_height // height_split
        parts = []
        for i in range(height_split):
            for j in range(width_split):
                sy, ey = i * part_h, (i + 1) * part_h
                sx, ex = j * part_w, (j + 1) * part_w
                parts.append(image[sy:ey, sx:ex, :].unsqueeze(0))

        result = []
        for i in range(9):
            result.append(parts[i] if i < len(parts) else
                          torch.zeros((1, part_h, part_w, channels), dtype=image.dtype, device=image.device))
        return tuple(result)

# --------------------------------------------------
#  统一注册
# --------------------------------------------------
register_node(ImageBatchCollector_mmx, "ImageBatchCollector_mmx")
register_node(save2JPG_mmx, "save2JPG_mmx")
register_node(LoadImageFromPath_mmx, "LoadImageFromPath_mmx")
register_node(ImageSplitGrid_mmx, "ImageSplitGrid_mmx")
