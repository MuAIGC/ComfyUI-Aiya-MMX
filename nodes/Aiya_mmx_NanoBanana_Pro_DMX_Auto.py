"""
💕 哎呀✦MMX NanoBanana2-DMX 全自动节点
无图=文生图(/generations)  有图=图生图(/edits)
1K/2K/4K | 官方宽高比 | 自动降级 | 无保存选项
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


# ---------- 节点 ----------
class NanoBanana2_DMX:
    DESCRIPTION = (
        "💕 哎呀✦NanoBanana2-DMX 全自动节点\n\n"
        "自动识别：无图走文生图(/generations)，有图走图生图(/edits)\n"
        "字段与 DMXAPI 官方 1:1 映射，支持 1K/2K/4K\n\n"
        "English: DMX-native auto txt/img2img / 14 imgs / 1K・2K・4K / fallback."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "endpoint_url": ("STRING", {
                    "default": "",
                    "placeholder": "https://www.dmxapi.cn/v1/images/(generations|edits)"
                }),
                "api_key": ("STRING", {
                    "default": "", "placeholder": "Your API key"
                }),
                "prompt": ("STRING", {"forceInput": True, "multiline": True}),
                "aspect_ratio": (
                    ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4",
                     "9:16", "16:9", "21:9"],
                    {"default": "1:1"}
                ),
                "size": (["1K", "2K", "4K"], {"default": "2K"}),
            },
            "optional": {f"input_image_{i}": ("IMAGE",) for i in range(1, 15)}
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "generate"
    CATEGORY = "哎呀✦MMX/nano-banana-2"

    # ---------- 内部 ----------
    def build_json(self, prompt, imgs, ar, size):
        """文生图 /generations"""
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
        payload = {
            "model": "nano-banana-2",
            "prompt": prompt,
            "aspect_ratio": ar,
            "size": size.lower(),
            "n": 1,
            "response_format": "b64_json"
        }
        if parts:
            payload["image"] = parts
        return payload

    def build_multipart(self, prompt, imgs, ar, size):
        """图生图 /edits"""
        port_map = {idx + 1: idx + 1 for idx, img in enumerate(imgs) if img is not None}
        for port, arr in port_map.items():
            prompt = re.sub(rf"图{port}(?!\d)", f"图{arr}", prompt)
        files = []
        for img in imgs:
            if img is not None:
                pil = tensor2pil(img)
                buf = BytesIO()
                pil.save(buf, format="PNG")
                buf.seek(0)
                files.append(("image", ("nb2.png", buf, "image/png")))
        data = {
            "model": "nano-banana-2",
            "prompt": prompt,
            "aspect_ratio": ar,
            "size": size.lower(),
            "n": 1,
            "response_format": "b64_json"
        }
        return data, files

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

    def call_api(self, url, key, ar, size, **kwargs):
        """带降级的一次封装"""
        headers = {"Authorization": f"Bearer {key}"}
        # 第一次：完整参数
        if "json" in kwargs:
            headers["Content-Type"] = "application/json"
            resp = requests.post(url, headers=headers, json=kwargs["json"], timeout=180)
        else:
            resp = requests.post(url, headers=headers, data=kwargs["data"], files=kwargs["files"], timeout=180)

        if resp.status_code == 200:
            return resp
        # 识别 rix 错误
        if "rix_api_error" in resp.text and "bad_response_status_code" in resp.text:
            print("[NanoBanana2-DMX] 后端不支持当前分辨率，自动降级重试…")
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
    def generate(self, endpoint_url, api_key, prompt, aspect_ratio, size, **img_ports):
        if not endpoint_url or not api_key:
            raise RuntimeError("[NanoBanana2-DMX] endpoint_url 和 api_key 不能为空！")
        imgs = [img_ports.get(f"input_image_{i}") for i in range(1, 15)]
        cnt = len([i for i in imgs if i is not None])
        mode = "图生图/编辑" if cnt else "文生图"
        print(f"\n[NanoBanana2-DMX] ===== {mode} =====")
        print(f"[NanoBanana2-DMX] imgs: {cnt}  ratio: {aspect_ratio}  size: {size}")

        base_url = endpoint_url.rstrip("/")
        if mode == "文生图":
            url = base_url.replace("/edits", "/generations")
            payload = self.build_json(prompt, imgs, aspect_ratio, size)
            resp = self.call_api(url, api_key, aspect_ratio, size, json=payload)
        else:
            url = base_url.replace("/generations", "/edits")
            data, files = self.build_multipart(prompt, imgs, aspect_ratio, size)
            resp = self.call_api(url, api_key, aspect_ratio, size, data=data, files=files)

        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

        images = self.decode_all(resp.json())
        best = max(images, key=lambda im: im.width * im.height)
        txt = (f"🍌 NanoBanana2-DMX {mode}  {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
               f"endpoint: {url}\nratio: {aspect_ratio}  size: {size}\n"
               f"input: {cnt}  output: {len(images)}")
        return (pil2tensor(best), txt)


# ---------- 注册 ----------
register_node(NanoBanana2_DMX, "NanoBanana_Pro_DMX")