from __future__ import annotations
import os
import json
import folder_paths
from PIL import Image
import numpy as np
import torch
from ..register import register_node


class ImageSaveJPG:
    DESCRIPTION = (
        "🖼 一键保存 JPG 并可选附加提示词文本\n\n"
        "参数说明：\n"
        "• optimize  — 压缩优化，文件更小，画质无损，耗时略增（默认开）\n"
        "• progressive — 渐进式 JPG，网页大图加载“由模糊到清晰”，文件稍大，老设备可能不兼容（默认关）\n"
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

    # ★ 新增一路 STRING：返回 jpg 绝对路径
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt_text", "jpg_path")
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "哎呀✦MMX/输出"

    def save_images(self,
                    images,
                    filename_prefix="ComfyUI",
                    quality=95,
                    optimize=True,
                    progressive=False,
                    save_prompt_as_txt=True,
                    prompt=None,
                    extra_pnginfo=None):
        # ===== 先把 %哎呀:xxx% 变量替换成真实日期 =====
        from ..date_variable import replace_date_vars
        filename_prefix = replace_date_vars(filename_prefix)
        # ============================================

        os.makedirs(self.output_dir, exist_ok=True)
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])

        prompt_text = self._extract_prompt_text(prompt)

        # ★ 保存绝对路径列表（多张图时返回首张路径）
        saved_paths = []

        results = []
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            file = f"{filename}_{counter:05}_.jpg"
            save_path = os.path.join(full_output_folder, file)
            img.save(save_path, format='JPEG', quality=quality,
                     optimize=optimize, progressive=progressive)
            saved_paths.append(save_path)          # ★ 记录路径

            # 只要开关打开就一定写 txt（空也写，保持旧习惯）
            if save_prompt_as_txt:
                txt_path = save_path.replace(".jpg", "_prompt.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(prompt_text)   # 空字符串也落盘

            results.append({"filename": file,
                           "subfolder": subfolder, "type": self.type})
            counter += 1

        # ★ 返回：prompt_text + 首张 jpg 绝对路径
        return {"ui": {"images": results},
                "result": (prompt_text, saved_paths[0] if saved_paths else "")}

    # ---------- 只抓「inputs.prompt」字段 ----------
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

register_node(ImageSaveJPG, "保存为JPG")
