from __future__ import annotations
import io
import requests
import base64
import time
import random
from PIL import Image
from io import BytesIO
import torch
import numpy as np
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
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)

# ---------- 节点 ----------
class BaiduAirDMX:
    DESCRIPTION = (
        "💕 哎呀✦MMX 百度 MuseSteamer-Air（文生图）\n\n"
        "模型固定：musesteamer-air-image\n\n"
        "1️⃣ 纯文生图\n"
        "  prompt≤500字；seed=-1随机\n\n"
        "2️⃣ 比例+清晰度一次选好，绝不错位\n"
        "  默认 3:4高清 1104×1472\n\n"
        "3️⃣ 返回最大图（单张）；503/超时重试3次\n"
        "  高峰失败请换普清或稍后再试～"
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
                "ratio_size": (["1:1  普清 1024×1024", "1:1  高清 1328×1328",
                               "4:3  普清 1152×864", "4:3  高清 1472×1104",
                               "3:4  普清  864×1152", "3:4  高清 1104×1472",
                               "16:9 普清 1280×720", "16:9 高清 1664×928",
                               "9:16 普清  720×1280", "9:16 高清  928×1664"], {"default": "3:4  高清 1104×1472"}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
            },
            "optional": {
                "seed": ("INT", {"default": -1, "min": -1, "max": 2_147_483_647, "step": 1, "hidden": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "generate"
    CATEGORY = "哎呀✦MMX/DMX"

    # label→value 映射
    RATIO_SIZE_MAP = {
        "1:1  普清 1024×1024": "1024x1024",
        "1:1  高清 1328×1328": "1328x1328",
        "4:3  普清 1152×864": "1152x864",
        "4:3  高清 1472×1104": "1472x1104",
        "3:4  普清  864×1152": "864x1152",
        "3:4  高清 1104×1472": "1104x1472",
        "16:9 普清 1280×720": "1280x720",
        "16:9 高清 1664×928": "1664x928",
        "9:16 普清  720×1280": "720x1280",
        "9:16 高清  928×1664": "928x1664",
    }

    # ---------- 网络请求：温柔重试 ----------
    def call_api(self, url: str, key: str, payload: dict):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}"
        }
        for attempt in range(1, 4):
            try:
                print(f"[BaiduAir-DMX] 第 {attempt}/3 次请求中…")
                resp = requests.post(url, headers=headers, json=payload, timeout=300)
                if 500 <= resp.status_code < 600:
                    print(f"[BaiduAir-DMX] 服务器开小差 ({resp.status_code})，{2 ** attempt}s 后重试…")
                    time.sleep(2 ** attempt)
                    continue
                return resp
            except requests.exceptions.Timeout:
                print(f"[BaiduAir-DMX] 请求超时 (>300s)，别急，我再试试…（{attempt}/3）")
                if attempt < 3:
                    time.sleep(5)
                continue
            except requests.exceptions.RequestException as e:
                print(f"[BaiduAir-DMX] 网络波动：{e}，{attempt}/3 次")
                if attempt < 3:
                    time.sleep(5)
                continue
        raise RuntimeError(
            "[BaiduAir-DMX] 我已经很努力啦，可服务器还是木有响应～\n"
            "1. 高峰时段生成较慢，请 3~5 分钟后再试；\n"
            "2. 检查 API 额度是否充足；\n"
            "3. 调低清晰度或稍后再试～"
        )

    def download_image(self, url: str) -> Image.Image:
        return Image.open(BytesIO(requests.get(url, timeout=60).content)).convert("RGB")

    # ---------- 主入口 ----------
    def generate(self, endpoint_url, api_key, prompt, ratio_size, response_format, seed=-1):
        if not api_key:
            raise RuntimeError("[BaiduAir-DMX] api_key 不能为空！")
        if not prompt or not prompt.strip():
            raise RuntimeError("[BaiduAir-DMX] prompt 为空，请先连接文本输入节点！")

        # 长度保护：百度文档 ≤500 字
        prompt = prompt.strip()[:500]
        if seed == -1:
            seed = random.randint(0, 2_147_483_647)
        size_clean = self.RATIO_SIZE_MAP[ratio_size]

        payload = {
            "model": "musesteamer-air-image",
            "prompt": prompt,
            "size": size_clean,
            "n": 1,
            "response_format": response_format,
        }
        url = endpoint_url.rstrip("/")
        print(f"\n[BaiduAir-DMX] ===== 文生图 =====")
        print(f"[BaiduAir-DMX] selected: {ratio_size}  |  size: {size_clean}  |  seed: {seed}")
        print(f"[BaiduAir-DMX] prompt[:100] = {prompt[:100]!r}")

        resp = self.call_api(url, api_key, payload)
        if resp.status_code != 200:
            print(f"[BaiduAir-DMX] 百度返回异常：{resp.text}")
            raise RuntimeError(f"HTTP {resp.status_code}：{resp.json().get('error', {}).get('message', 'unknown')}")

        data = resp.json()
        if response_format == "url":
            img_url = data["data"][0]["url"]
            img = self.download_image(img_url)
        else:
            b64 = data["data"][0]["b64_json"]
            img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")

        info = (f"🍉 BaiduAir-DMX 文生图  {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"endpoint: {url}\nselected: {ratio_size}  size: {size_clean}  seed: {seed}")
        return (pil2tensor(img), info)

register_node(BaiduAirDMX, "BaiduAir_DMX")
