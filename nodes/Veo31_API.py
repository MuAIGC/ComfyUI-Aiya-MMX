# Veo3_1_API.py
from __future__ import annotations
import json
import requests
import torch
import io
import base64
import time
import uuid
import cv2
import folder_paths
from pathlib import Path
from PIL import Image
from datetime import datetime
from ..register import register_node
from ..mmx_utils import pil2tensor, tensor2pil
from ..video_adapter import Video          # ComfyUI 标准 VIDEO 对象

VEO3_MODELS = [
    "veo3.1", "veo3.1-pro", "veo3.1-components",
    "veo3.1-4k", "veo3.1-pro-4k", "veo3.1-components-4k",
]

# --------------------------------------------------
# 通用工具（直接抄 DMX 节点的写法）
# --------------------------------------------------
def build_video_obj(video_path: Path) -> Video:
    """把本地 mp4 封装成 ComfyUI VIDEO 对象"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return Video(str(video_path), fps, w, h)

def download_file(url: str, dst: Path, max_retry: int = 3, timeout: int = 120):
    """带重试的下载"""
    for attempt in range(1, max_retry + 1):
        try:
            print(f"[Veo3.1 Download] 第 {attempt}/{max_retry} 次：{url}")
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(dst, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            print(f"[Veo3.1 Download] 成功 → {dst}")
            return
        except Exception as e:
            print(f"[Veo3.1 Download] 第 {attempt} 次失败：{e}")
            if attempt == max_retry:
                raise RuntimeError(f"下载失败（重试 {max_retry} 次）：{e}")
            time.sleep(2)

# --------------------------------------------------
# 节点主体
# --------------------------------------------------
class ComflyVeo3_1:
    DESCRIPTION = (
        "💕 哎呀✦MMX/Veo3.1 谷歌文生视频\n\n"
        "【功能】输入文本 → 输出视频张量 + URL + 任务信息\n"
        "【模型】veo3.1 / pro / components / 4K 全系支持\n"
        "【必填】API 密钥 + 提示词；其余按需调节\n"
        "【参数】时长 5-25s、分辨率 16:9 或 9:16、enhance、upsample\n"
        "【输入】可插 3 张参考图（自动转 base64）\n"
        "【输出】IO.VIDEO 标准张量 + 视频 URL + JSON 详情\n"
        "【异常】失败返回空视频适配器 + ❌ 信息，下游不崩\n\n"
        "========== 使用示例 ==========\n"
        "提示词：A drone flies over the Great Wall at sunrise, 4K cinematic\n"
        "模型：veo3.1-pro-4k → 25 s → 16:9 → enhance → upsample\n"
    )
    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "info")
    FUNCTION = "generate_video"
    CATEGORY = "哎呀✦MMX/Video"

    def __init__(self):
        self.timeout = 120
        self.poll_interval = 2
        self.max_poll = 150

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "placeholder": "sk-***************************"}),
                "base_url": ("STRING", {"default": "https://ai.t8star.cn", "placeholder": "API 根地址"}),
                "prompt": ("STRING", {"multiline": True, "default": "A cinematic aerial shot of a neon-lit cyberpunk city at night, 4K, ultra detailed"}),
                "model": (VEO3_MODELS, {"default": "veo3.1-pro"}),
                "duration": (["5", "10", "15", "20", "25"], {"default": "10"}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
                "enhance_prompt": ("BOOLEAN", {"default": True}),
                "enable_upsample": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image1": ("IMAGE",), "image2": ("IMAGE",), "image3": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }

    # ---------------- 工具 ----------------
    def image_to_base64(self, img_tensor):
        if img_tensor is None:
            return None
        pil = tensor2pil(img_tensor)[0]
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    # ---------------- 主入口 ----------------
    def generate_video(self, api_key, base_url, prompt, model, duration, aspect_ratio,
                       enhance_prompt, enable_upsample,
                       image1=None, image2=None, image3=None, seed=0):
        if not api_key.strip():
            return (Video.create_empty(), "", "❌ API Key 为空")

        root = base_url.rstrip("/")
        submit_url = f"{root}/v2/videos/generations"
        query_url  = f"{root}/v2/videos/generations/{{}}"

        images_b64 = []
        for img in (image1, image2, image3):
            if img is not None:
                b64 = self.image_to_base64(img)
                if b64:
                    images_b64.append(f"data:image/png;base64,{b64}")

        payload = {
            "model": model,
            "prompt": prompt,
            "duration": int(duration),
            "aspect_ratio": aspect_ratio,
            "enhance_prompt": enhance_prompt,
            "enable_upsample": enable_upsample,
        }
        if images_b64:
            payload["images"] = images_b64
        if seed > 0:
            payload["seed"] = seed

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key.strip()}"
        }

        try:
            # 1. 提交任务
            resp = requests.post(submit_url, headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            task_id = data.get("task_id")
            if not task_id:
                return (Video.create_empty(), "", "❌ 未返回 task_id")
            print(f"[Veo3.1] 任务已提交: {task_id}")

            # 2. 轮询状态
            for i in range(self.max_poll):
                time.sleep(self.poll_interval)
                st = requests.get(query_url.format(task_id), headers=headers, timeout=30)
                st.raise_for_status()
                st_data = st.json()
                status = st_data.get("status", "")

                if status == "SUCCESS":
                    video_url = st_data.get("data", {}).get("output", "")
                    if video_url:
                        # 下载到临时目录
                        temp_dir = Path(folder_paths.get_temp_directory())
                        temp_dir.mkdir(parents=True, exist_ok=True)
                        temp_file = temp_dir / f"veo3_1_{int(time.time()*1000)}.mp4"
                        download_file(video_url, temp_file)

                        # 封装成 ComfyUI VIDEO 对象
                        video_obj = build_video_obj(temp_file)
                        info_json = {
                            "task_id": task_id,
                            "model": model,
                            "prompt": prompt,
                            "duration": duration,
                            "aspect_ratio": aspect_ratio,
                            "enhance": enhance_prompt,
                            "upsample": enable_upsample,
                            "seed": seed if seed > 0 else "auto",
                            "video_url": video_url,
                        }
                        return (video_obj, video_url, json.dumps(info_json, ensure_ascii=False, indent=2))
                    else:
                        return (Video.create_empty(), "", f"❌ 状态成功但无视频 URL: {st_data}")

                elif status == "FAILURE":
                    reason = st_data.get("fail_reason", "Unknown")
                    return (Video.create_empty(), "", f"❌ 任务失败: {reason}")

            # 超时
            return (Video.create_empty(), "", f"❌ 轮询超时（>{self.max_poll * self.poll_interval}s）")

        except requests.exceptions.Timeout:
            return (Video.create_empty(), "", "❌ 请求超时 (120s)")
        except Exception as e:
            return (Video.create_empty(), "", f"❌ 异常: {str(e)}")

# ========== 注册节点 ==========
register_node(ComflyVeo3_1, "Comfly Veo3.1 文生视频")