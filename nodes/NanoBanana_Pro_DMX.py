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

# ===================================================================
#  1.  NanoBanana_Pro_OA_DMX  ——  DMX OpenAI 兼容通道（1K/2K/4K 字段）
# ===================================================================
class NanoBanana_Pro_OA_DMX:
    DESCRIPTION = "💕 哎呀✦NanoBanana_Pro_OA_DMX —— DMX OpenAI 兼容 / 14 图 / 1-4K"

    DEFAULT_ENDPOINT = "https://www.dmxapi.cn/v1"   # ← 只留前缀

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "endpoint_url": ("STRING", {"default": cls.DEFAULT_ENDPOINT}),
                "api_key": ("STRING", {"default": "", "placeholder": "sk-***"}),
                "model": ("STRING", {"default": "nano-banana-2", "placeholder": "model name"}),
                "prompt": ("STRING", {"forceInput": True, "multiline": True}),
                "aspect_ratio": (["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"], {"default": "1:1"}),
                "size": (["1K", "2K", "4K"], {"default": "1K"}),
            },
            "optional": {f"input_image_{i}": ("IMAGE",) for i in range(1, 15)}
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "generate"
    CATEGORY = "哎呀✦MMX/DMXAPI"

    # ---------- 构造 payload ----------
    def build_payload(self, prompt, imgs, ar, size, model):
        prompt = f"{prompt} [var-{np.random.randint(10000, 99999)}]"
        parts = []
        for img in imgs:
            if img is not None:
                pil = tensor2pil(img)
                buf = BytesIO()
                pil.save(buf, format="PNG")
                parts.append(base64.b64encode(buf.getvalue()).decode())
        payload = {
            "model": model,
            "prompt": prompt,
            "aspect_ratio": ar,
            "size": size.lower(),
            "n": 1,
            "response_format": "b64_json"
        }
        if parts:
            payload["image"] = parts
        return payload

    # ---------- DMX 专用解码 ----------
    def decode_all(self, result: dict) -> list[Image.Image]:
        images = []
        # DMX 图生图：顶层 image
        if "image" in result:
            images.append(Image.open(BytesIO(base64.b64decode(result["image"]))).convert("RGB"))
            return images
        # DMX 文生图：顶层 b64_json
        if "b64_json" in result:
            images.append(Image.open(BytesIO(base64.b64decode(result["b64_json"]))).convert("RGB"))
            return images
        # 兜底：OpenAI 格式 data 数组
        for item in result.get("data", []):
            if "b64_json" in item:
                images.append(Image.open(BytesIO(base64.b64decode(item["b64_json"]))).convert("RGB"))
            elif "url" in item:
                images.append(Image.open(BytesIO(requests.get(item["url"], timeout=60).content)).convert("RGB"))
        if not images:
            raise RuntimeError("No image returned")
        return images

    # ---------- 主入口 ----------
    def generate(self, endpoint_url, api_key, model, prompt, aspect_ratio, size, **img_ports):
        imgs = [img_ports.get(f"input_image_{i}") for i in range(1, 15) if img_ports.get(f"input_image_{i}") is not None]
        cnt = len(imgs)
        print(f"[NanoBanana_Pro_OA_DMX] model={model} imgs={cnt} ratio={aspect_ratio} res={size}")

        base_url = endpoint_url.strip().rstrip("/")
        url = base_url + "/images/" + ("edits" if cnt else "generations")

        payload = self.build_payload(prompt, imgs, aspect_ratio, size, model)
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        resp = requests.post(url, headers=headers, json=payload, timeout=300)

        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

        images = self.decode_all(resp.json())
        best = max(images, key=lambda im: im.width * im.height)
        info = (f"🍌 NanoBanana_Pro_OA_DMX {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"endpoint: {url}\nmodel: {model}\nratio: {aspect_ratio}  size: {size.upper()}\n"
                f"input: {cnt}  output: {len(images)}")
        return (pil2tensor(best), info)
    
# ===================================================================
#  2.  NanoBanana_Pro_GN_DMX  ——  DMX 原生 4K 通道（Gemini 原生格式）
# ===================================================================
class NanoBanana_Pro_GN_DMX:
    DESCRIPTION = (
        "💕 哎呀✦NanoBanana_Pro_GN_DMX —— DMX 原生 / 真 1K·2K·4K / 14 图输入\n"
        "endpoint: /v1beta/models/gemini-3-pro-image-preview:generateContent"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "endpoint_url": ("STRING", {
                    "default": "https://www.dmxapi.cn/v1beta/models/gemini-3-pro-image-preview:generateContent",
                    "placeholder": "https://xxx/v1beta/models/gemini-3-pro-image-preview:generateContent"
                }),
                "api_key": ("STRING", {"default": "", "placeholder": "sk-***"}),
                "model": ("STRING", {"default": "gemini-3-pro-image-preview", "placeholder": "model name"}),
                "prompt": ("STRING", {"forceInput": True, "multiline": True}),
                "aspect_ratio": (["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"], {"default": "1:1"}),
                "resolution": (["1K", "2K", "4K"], {"default": "2K"}),
            },
            "optional": {f"input_image_{i}": ("IMAGE",) for i in range(1, 15)}
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "generate"
    CATEGORY = "哎呀✦MMX/DMXAPI"

    def add_random(self, p: str) -> str:
        return f"{p} [var-{random.randint(10000, 99999)}]"

    def tensor2pil(self, t):
        if t.ndim == 4:
            t = t.squeeze(0)
        if t.ndim == 3 and t.shape[2] == 3:
            t = (t * 255).clamp(0, 255).byte() if t.is_floating_point() else t
            return Image.fromarray(t.cpu().numpy(), "RGB")
        raise ValueError("Unsupported tensor shape")

    def pil2tensor(self, img: Image.Image):
        return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)

    def build_gn_payload(self, prompt, imgs, ar, res, model):
        prompt = self.add_random(prompt)
        parts = [{"text": prompt}]
        for img in imgs:
            if img is not None:
                pil = self.tensor2pil(img)
                buf = io.BytesIO()
                pil.save(buf, format="PNG")
                parts.append({
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": base64.b64encode(buf.getvalue()).decode()
                    }
                })
        return {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {
                    "aspectRatio": ar,
                    "imageSize": res.upper()   # 原生 4K 字段
                }
            }
        }

    def decode_gn(self, resp_json):
        for cand in resp_json.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                if "inlineData" in part:
                    b64_data = part["inlineData"]["data"]
                    im = Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("RGB")
                    return self.pil2tensor(im)
        return None

    def generate(self, endpoint_url, api_key, model, prompt, aspect_ratio, resolution, **img_ports):
        imgs = [img_ports.get(f"input_image_{i}") for i in range(1, 15) if img_ports.get(f"input_image_{i}") is not None]
        cnt = len(imgs)
        print(f"[NanoBanana_Pro_GN_DMX] model={model} imgs={cnt} ratio={aspect_ratio} res={resolution}")

        payload = self.build_gn_payload(prompt, imgs, aspect_ratio, resolution, model)
        headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
        resp = requests.post(endpoint_url, headers=headers, json=payload, timeout=300)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

        best = self.decode_gn(resp.json())
        if best is None:
            raise RuntimeError("No inlineData image returned")

        info = (f"🍌 NanoBanana_Pro_GN_DMX {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"endpoint: {endpoint_url}\nmodel: {model}\n"
                f"ratio: {aspect_ratio}  size: {resolution.upper()}\ninput: {cnt}  success: True")
        return (best, info)


# ---------- 统一注册 ----------
register_node(NanoBanana_Pro_OA_DMX, "NanoBanana_Pro_OA_DMX")
register_node(NanoBanana_Pro_GN_DMX, "NanoBanana_Pro_GN_DMX")
