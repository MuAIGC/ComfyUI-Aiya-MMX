"""
💕 哎呀✦MMX SeeDream4.5 图生图对齐版（10图）
官方 JSON + image[]（URL/Base64）方式，与示例完全一致
四件套：清晰度 / 比例 / 网址 / key  
运行日志打印完整参数，其余写死
"""
from __future__ import annotations
import io
import requests
import base64
import time
import os
import re
from datetime import datetime
import numpy as np
from PIL import Image
from io import BytesIO
import random
import torch
from ..register import register_node


# ---------- utils ----------
def tensor2pil(t):
    if t.ndim == 4:
        t = t.squeeze(0)
    if t.ndim == 3 and t.shape[2] == 3:
        t = (t * 255).clamp(0, 255).byte() if t.is_floating_point() else t
        return Image.fromarray(t.cpu().numpy(), "RGB")
    raise ValueError("Unsupported tensor shape")


def pil2tensor(img: Image.Image):
    return torch.from_numpy(
        np.array(img).astype(np.float32) / 255.0
    ).unsqueeze(0)


# --------------------------------------------------
# 官方推荐尺寸（与示例完全一致，≤ 16777216 像素）
# --------------------------------------------------
RECOMMENDED_SIZE = {
    ("1K", "1:1"): "1024x1024",
    ("1K", "4:3"): "1152x864",
    ("1K", "3:4"): "864x1152",
    ("1K", "16:9"): "1280x720",
    ("1K", "9:16"): "720x1280",
    ("1K", "3:2"): "1224x816",
    ("1K", "2:3"): "816x1224",
    ("1K", "21:9"): "1440x600",

    ("2K", "1:1"): "2048x2048",
    ("2K", "4:3"): "2304x1728",
    ("2K", "3:4"): "1728x2304",
    ("2K", "16:9"): "2560x1440",
    ("2K", "9:16"): "1440x2560",
    ("2K", "3:2"): "2496x1664",
    ("2K", "2:3"): "1664x2496",
    ("2K", "21:9"): "3024x1296",

    ("4K", "1:1"): "4096x4096",
    ("4K", "4:3"): "4096x3072",
    ("4K", "3:4"): "3072x4096",
    ("4K", "16:9"): "4096x2304",
    ("4K", "9:16"): "2304x4096",
    ("4K", "3:2"): "4096x2731",
    ("4K", "2:3"): "2731x4096",
    ("4K", "21:9"): "4096x1714",
}


# ---------- 节点 ----------
class SeeDream4_5_DMX:
    DESCRIPTION = (
        "💕 哎呀✦MMX SeeDream4.5 图生图对齐版（10图）\n\n"
        "官方 JSON + image[]（URL/Base64）方式，与示例完全一致\n"
        "四件套：清晰度 / 比例 / 网址 / key  \n"
        "运行日志打印完整参数，9:16 已对齐\n\n"
        "English: DMX-native doubao-seedream-4-5-251128 official JSON+image[] / 10 imgs / full logs"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "endpoint_url": ("STRING", {
                    "default": "https://www.dmxapi.cn/v1/images/generations",
                    "placeholder": "https://your-domain/v1/images/generations"
                }),
                "api_key": ("STRING", {"default": "", "placeholder": "Your API key"}),
                "prompt": ("STRING", {"forceInput": True}),
                "clarity": (["1K", "2K", "4K"], {"default": "2K"}),
                "aspect_ratio": (
                    ["1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3", "21:9"],
                    {"default": "1:1"}
                ),
            },
            "optional": {
                "seed": ("INT", {"default": -1, "min": -1, "max": 2_147_483_647, "step": 1, "hidden": True}),
                # 10 张参考图，前端默认只显示 2 个，其余隐藏
                "input_image_1": ("IMAGE",),
                "input_image_2": ("IMAGE",),
                "input_image_3": ("IMAGE",),
                "input_image_4": ("IMAGE",),
                "input_image_5": ("IMAGE",),
                "input_image_6": ("IMAGE",),
                "input_image_7": ("IMAGE",),
                "input_image_8": ("IMAGE",),
                "input_image_9": ("IMAGE",),
                "input_image_10": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "generate"
    CATEGORY = "哎呀✦MMX/SeeDream"

    # ---------- 内部 ----------
    def resolve_size(self, clarity, ar):
        """清晰度+比例→像素"""
        return RECOMMENDED_SIZE[(clarity, ar)]

    def build_json(self, prompt, imgs, clarity, ar, seed):
        """文生图（无图）"""
        port_map = {idx + 1: idx + 1 for idx, img in enumerate(imgs) if img is not None}
        for port, arr in port_map.items():
            prompt = re.sub(rf"图{port}(?!\d)", f"图{arr}", prompt)
        parts = []
        for img in imgs:
            if img is not None:
                pil = tensor2pil(img)
                buf = BytesIO()
                pil.save(buf, format="PNG")
                parts.append(base64.b64encode(buf.getvalue()).decode())
        size = self.resolve_size(clarity, ar)
        payload = {
            "model": "doubao-seedream-4-5-251128",  # 写死模型
            "prompt": prompt,
            "aspect_ratio": ar,
            "size": size,
            "n": 1,
            "response_format": "b64_json",
            "optimize_prompt_options": {"mode": "standard"},
            "watermark": False,
        }
        if seed != -1:
            payload["seed"] = seed
        if parts:
            payload["image"] = parts
        return payload

    def build_json_url(self, prompt, imgs, clarity, ar, seed):
        """图生图（URL 方式，与官方示例一致）"""
        port_map = {idx + 1: idx + 1 for idx, img in enumerate(imgs) if img is not None}
        for port, arr in port_map.items():
            prompt = re.sub(rf"图{port}(?!\d)", f"图{arr}", prompt)
        # 把 tensor 转成 base64 URL，模拟官方示例
        url_list = []
        for img in imgs:
            if img is not None:
                pil = tensor2pil(img)
                buf = BytesIO()
                pil.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                url_list.append(f"data:image/png;base64,{b64}")
        size = self.resolve_size(clarity, ar)
        payload = {
            "model": "doubao-seedream-4-5-251128",  # 写死模型
            "prompt": prompt,
            "aspect_ratio": ar,
            "size": size,
            "n": 1,
            "response_format": "b64_json",
            "optimize_prompt_options": {"mode": "standard"},
            "watermark": False,
        }
        if seed != -1:
            payload["seed"] = seed
        if url_list:
            payload["image"] = url_list  # 官方示例：image 为 URL/Base64 数组
        return payload

    def decode_all(self, result: dict) -> list[Image.Image]:
        images = []
        for item in result.get("data", []):
            if "b64_json" in item:
                images.append(Image.open(BytesIO(base64.b64decode(item["b64_json"]))).convert("RGB"))
            elif "url" in item:
                images.append(Image.open(BytesIO(requests.get(item["url"], timeout=60).content)).convert("RGB"))
        if not images:
            raise RuntimeError("No image returned")
        return images

    def call_api(self, url, key, ar, **kwargs):
        """带降级重试（503 时删 aspect_ratio+size）"""
        headers = {"Authorization": f"Bearer {key}"}
        if "json" in kwargs:
            headers["Content-Type"] = "application/json"
            resp = requests.post(url, headers=headers, json=kwargs["json"], timeout=180)
        else:
            resp = requests.post(url, headers=headers, data=kwargs["data"], files=kwargs["files"], timeout=180)

        if resp.status_code == 200:
            return resp
        if "rix_api_error" in resp.text and "bad_response_status_code" in resp.text:
            print("[SeeDream4.5-DMX] 后端限流，自动降级重试…")
            if "json" in kwargs:
                payload = kwargs["json"].copy()
                payload.pop("aspect_ratio", None)
                payload.pop("size", None)
                return requests.post(url, headers=headers, json=payload, timeout=180)
            else:
                data = kwargs["data"].copy()
                data.pop("aspect_ratio", None)
                data.pop("size", None)
                return requests.post(url, headers=headers, data=data, files=kwargs["files"], timeout=180)
        return resp

    # ---------- 主入口 ----------
    def generate(self, endpoint_url, api_key, prompt, clarity, aspect_ratio, seed: int = -1, **imgs):
        if not api_key:
            raise RuntimeError("[SeeDream4.5-DMX] api_key 不能为空！")
        img_list = [imgs.get(f"input_image_{i}") for i in range(1, 11)]
        cnt = len([i for i in img_list if i is not None])
        mode = "图生图/编辑（URL）" if cnt else "文生图"
        size = self.resolve_size(clarity, aspect_ratio)
        # 随机种子
        if seed == -1:
            seed = random.randint(0, 2_147_483_647)

        url = endpoint_url.rstrip("/")
        print(f"\n[SeeDream4.5-DMX] ===== {mode} =====")
        print(f"[SeeDream4.5-DMX] clarity: {clarity} | ratio: {aspect_ratio} → size: {size}")
        print(f"[SeeDream4.5-DMX] imgs: {cnt} | seed: {seed}")

        # 官方示例：有图时走 JSON + image[]（URL/Base64）
        if cnt:
            payload = self.build_json_url(prompt, img_list, clarity, aspect_ratio, seed)
            resp = self.call_api(url, api_key, aspect_ratio, json=payload)
        else:
            payload = self.build_json(prompt, img_list, clarity, aspect_ratio, seed)
            resp = self.call_api(url, api_key, aspect_ratio, json=payload)

        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

        images = self.decode_all(resp.json())
        best = max(images, key=lambda im: im.width * im.height)
        txt = (f"🍌 SeeDream4.5-DMX {mode}  {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
               f"endpoint: {url}\nclarity: {clarity}  ratio: {aspect_ratio}  size: {size}  seed: {seed}\n"
               f"input: {cnt}  output: {len(images)}")
        return (pil2tensor(best), txt)


# ---------- 注册 ----------
register_node(SeeDream4_5_DMX, "SeeDream4_5_DMX")