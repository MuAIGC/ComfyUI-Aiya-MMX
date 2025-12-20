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
        "💕 哎呀✦NanoBanana2-DMX 一键出图\n\n"
        "无图 = 文生图 (/generations)  |  有图 = 图生图 (/edits)\n"
        "模型：nano-banana-2  |  最多 14 张参考图\n"
        "分辨率：1K / 2K / 4K  |  宽高比：1:1 ~ 21:9\n"
        "字段与官方 1:1 映射，自动降级，免保存配置\n\n"
        "English: Auto txt|img2img, 14 imgs, 1-4K, fallback on error."
    )

    # 1. 预置默认 endpoint，想改只改这一行 -----------------------------
    DEFAULT_ENDPOINT = "https://www.dmxapi.cn/v1/images/generations"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "endpoint_url": ("STRING", {
                    "default": cls.DEFAULT_ENDPOINT,
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
    CATEGORY = "哎呀✦MMX/DMXAPI"

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
        """温柔重试：300 s 超时，最多 3 次，503/5xx/超时都重试"""
        headers = {"Authorization": f"Bearer {key}"}
        max_retry = 3
        for attempt in range(1, max_retry + 1):
            try:
                print(f"[NanoBanana2-DMX] 第 {attempt}/{max_retry} 次请求中… 请再等等我哦~")
                if "json" in kwargs:
                    headers["Content-Type"] = "application/json"
                    resp = requests.post(url, headers=headers, json=kwargs["json"], timeout=300)
                else:
                    resp = requests.post(url, headers=headers, data=kwargs["data"],
                                         files=kwargs["files"], timeout=300)

                # 503/5xx 重试
                if 500 <= resp.status_code < 600:
                    print(f"[NanoBanana2-DMX] 服务器开小差 ({resp.status_code})，{(2 ** attempt)} 秒后重试…")
                    time.sleep(2 ** attempt)
                    continue

                # rix 限流降级
                if "rix_api_error" in resp.text and "bad_response_status_code" in resp.text:
                    print("[NanoBanana2-DMX] 后端限流，自动降级（去掉 aspect_ratio & size）重试…")
                    if "json" in kwargs:
                        payload = kwargs["json"].copy()
                        payload.pop("aspect_ratio", None)
                        payload.pop("size", None)
                        resp = requests.post(url, headers=headers, json=payload, timeout=300)
                    else:
                        data = kwargs["data"].copy()
                        data.pop("aspect_ratio", None)
                        data.pop("size", None)
                        resp = requests.post(url, headers=headers, data=data,
                                             files=kwargs["files"], timeout=300)
                return resp

            except requests.exceptions.Timeout:
                print(f"[NanoBanana2-DMX] 请求超时 (>300 s)，别急，我再试试…（{attempt}/{max_retry}）")
                if attempt < max_retry:
                    time.sleep(5)
                continue
            except requests.exceptions.RequestException as e:
                print(f"[NanoBanana2-DMX] 网络波动：{e}，{attempt}/{max_retry} 次")
                if attempt < max_retry:
                    time.sleep(5)
                continue

        # 温柔地抛异常
        raise RuntimeError(
            "[NanoBanana2-DMX] 我已经很努力啦，可服务器还是木有响应～\n"
            "1. 高峰时段生成较慢，请 3~5 分钟后再试；\n"
            "2. 检查 API 额度是否充足；\n"
            "3. 调低清晰度（4K→2K）或减少参考图数量再试试～"
        )

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

register_node(NanoBanana2_DMX, "NanoBanana_Pro_DMX")
