# ~/ComfyUI/custom_nodes/ComfyUI-Aiya-MMX/nodes/sora2_API.py
from __future__ import annotations
import io
import json
import time
import base64
import requests
from pathlib import Path
from PIL import Image
import torch
from ..register import register_node
from ..mmx_utils import tensor2pil
from ..video_adapter import Video

# ---------- 通用工具 ----------
def _download_file(url: str, dst: Path, max_retry: int = 3, timeout: int = 120):
    """下载视频到本地临时文件（MMX 标准下载器）"""
    for attempt in range(1, max_retry + 1):
        try:
            print(f"[Sora2_mmx] 下载视频 {attempt}/{max_retry}: {url}")
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(dst, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            return
        except Exception as e:
            print(f"[Sora2_mmx] 下载失败: {e}")
            if attempt == max_retry:
                raise
            time.sleep(2)


# ===================================================================
#  Sora 2 / Sora 2 Pro 文/图生视频（模型名外显字符串版）
# ===================================================================
class Sora2_mmx:
    DESCRIPTION = (
        "💕 哎呀✦Sora 2 —— OpenAI Sora 视频生成\n"
        "模型名可自由输入（如 sora-2 / sora-2-pro / sora-3 等）"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "placeholder": "sk-***************************"}),
                "base_url": ("STRING", {"default": "https://ai.t8star.cn", "placeholder": "API 根地址"}),
                "prompt": ("STRING", {"multiline": True, "default": "A cinematic shot of..."}),
                "model": ("STRING", {"default": "sora-2", "placeholder": "sora-2 / sora-2-pro"}),  # ← 改为字符串输入
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
                "duration": (["10", "15", "25"], {"default": "15"}),
                "hd": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "private": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "info")
    FUNCTION = "generate_video"
    CATEGORY = "哎呀✦MMX/Video"

    def __init__(self):
        self.timeout = 900

    def image_to_base64(self, image_tensor):
        """Tensor → Base64 Data URI"""
        if image_tensor is None:
            return None
        pil_image = tensor2pil(image_tensor)[0]
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{base64_str}"

    def generate_video(self, api_key, base_url, prompt, model, aspect_ratio,
                       duration, hd, image1=None, image2=None, image3=None,
                       image4=None, seed=0, private=True):
        
        # 1. 基础校验
        if not api_key.strip():
            return (Video.create_empty(), "", json.dumps(
                {"status": "error", "message": "API key 未填写"}, ensure_ascii=False))
        
        root = base_url.rstrip("/")
        model_clean = model.strip()  # 去除首尾空格
        
        # 2. 参数兼容性检查（针对 sora-2 的硬性限制）
        if model_clean == "sora-2":
            if duration == "25":
                err = "sora-2 不支持 25 秒视频，请切换到 sora-2-pro 或修改模型名"
                print(f"[Sora2_mmx] {err}")
                return (Video.create_empty(), "", json.dumps(
                    {"status": "error", "message": err}, ensure_ascii=False))
            if hd:
                err = "sora-2 不支持 HD 模式，请切换到 sora-2-pro 或关闭 HD"
                print(f"[Sora2_mmx] {err}")
                return (Video.create_empty(), "", json.dumps(
                    {"status": "error", "message": err}, ensure_ascii=False))
        
        # 3. 构建 Payload
        payload = {
            "prompt": prompt,
            "model": model_clean,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
            "hd": hd,
            "private": private
        }
        if seed > 0:
            payload["seed"] = seed

        # 4. 处理多图输入
        images = []
        for img in (image1, image2, image3, image4):
            if img is not None:
                b64 = self.image_to_base64(img)
                if b64:
                    images.append(b64)
        
        if images:
            payload["images"] = images
        
        # 5. 提交任务
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key.strip()}"
        }
        endpoint = f"{root}/v2/videos/generations"
        
        try:
            print(f"[Sora2_mmx] 提交任务 | model={model_clean} | duration={duration}s | hd={hd} | imgs={len(images)}")
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            err_msg = f"提交失败: {str(e)}"
            print(f"[Sora2_mmx] {err_msg}")
            return (Video.create_empty(), "", json.dumps(
                {"status": "error", "message": err_msg}, ensure_ascii=False))
        
        task_id = result.get("task_id")
        if not task_id:
            err_msg = "API 未返回 task_id"
            print(f"[Sora2_mmx] {err_msg} | 响应: {result}")
            return (Video.create_empty(), "", json.dumps(
                {"status": "error", "message": err_msg, "raw": result}, ensure_ascii=False))
        
        print(f"[Sora2_mmx] 任务已提交: {task_id}")

        # 6. 轮询状态
        query_url = f"{root}/v2/videos/generations/{{}}"
        max_attempts = 300
        video_url = None
        
        for attempt in range(max_attempts):
            time.sleep(10)
            
            try:
                st_resp = requests.get(query_url.format(task_id), headers=headers, timeout=60)
                st_resp.raise_for_status()
                st_data = st_resp.json()
                
                status = st_data.get("status", "")
                progress_text = st_data.get("progress", "0%")
                
                if attempt % 10 == 0:
                    print(f"[Sora2_mmx] 轮询 {attempt}/{max_attempts} | 状态: {status} | 进度: {progress_text}")
                
                if status == "SUCCESS":
                    if "data" in st_data and "output" in st_data["data"]:
                        video_url = st_data["data"]["output"]
                        print(f"[Sora2_mmx] 生成成功 | URL: {video_url[:60]}...")
                        break
                    else:
                        err_msg = "状态为 SUCCESS 但未找到 output 字段"
                        print(f"[Sora2_mmx] {err_msg} | 数据: {st_data}")
                        return (Video.create_empty(), "", json.dumps(
                            {"status": "error", "message": err_msg, "raw": st_data}, ensure_ascii=False))
                
                elif status == "FAILURE":
                    fail_reason = st_data.get("fail_reason", "Unknown")
                    err_msg = f"生成失败: {fail_reason}"
                    print(f"[Sora2_mmx] {err_msg}")
                    return (Video.create_empty(), "", json.dumps(
                        {"status": "failed", "message": err_msg, "task_id": task_id}, ensure_ascii=False))
                        
            except Exception as e:
                print(f"[Sora2_mmx] 轮询异常: {e}")
                continue
        
        if not video_url:
            err_msg = f"轮询超时（{max_attempts}次尝试后仍未获取结果）"
            print(f"[Sora2_mmx] {err_msg}")
            return (Video.create_empty(), "", json.dumps(
                {"status": "timeout", "message": err_msg, "task_id": task_id}, ensure_ascii=False))
        
        # 7. 下载视频
        try:
            import folder_paths
            import cv2
            temp_dir = Path(folder_paths.get_temp_directory())
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_file = temp_dir / f"sora2_{task_id}_{int(time.time())}.mp4"
            
            _download_file(video_url, temp_file)
            
            cap = cv2.VideoCapture(str(temp_file))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            video_obj = Video(str(temp_file), fps, w, h)
            
            info = {
                "status": "success",
                "model": model_clean,
                "task_id": task_id,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "duration": duration,
                "hd": hd,
                "private": private,
                "seed": seed if seed > 0 else "auto",
                "video_url": video_url,
                "local_path": str(temp_file)
            }
            
            return (video_obj, video_url, json.dumps(info, ensure_ascii=False, indent=2))
            
        except Exception as e:
            err_msg = f"视频下载/处理失败: {e}"
            print(f"[Sora2_mmx] {err_msg}")
            return (Video.create_empty(), video_url, json.dumps(
                {"status": "partial_success", "message": err_msg, "video_url": video_url}, ensure_ascii=False))


# ===================================================================
# 统一注册
# ===================================================================
register_node(Sora2_mmx, "Sora2_mmx")