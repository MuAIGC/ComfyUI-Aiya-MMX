# ~/ComfyUI/custom_nodes/ComfyUI-Aiya-MMX/nodes/modelscope_api.py
from __future__ import annotations
import io
import json
import base64
import time
import requests
import torch
from PIL import Image
from ..register import register_node
from ..mmx_utils import pil2tensor

# ===================================================================
#  ModelScope 图像生成（文生图/图生图）
# ===================================================================
class ModelScope_Image:
    DESCRIPTION = (
        "💕 哎呀✦ModelScope 图像生成 —— 魔塔文生图/图生图\n"
        "支持 Tongyi-MAI/Z-Image-Turbo、Qwen-Image 等 ModelScope AIGC 模型"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_url": ("STRING", {"default": "https://api-inference.modelscope.cn"}),
                "api_key": ("STRING", {"default": "", "placeholder": "从 modelscope.cn/my/myaccesstoken 获取"}),
                "prompt": ("STRING", {"multiline": True, "forceInput": True, "default": ""}),
                "model": ("STRING", {"default": "Tongyi-MAI/Z-Image-Turbo"}),
                "width": ("STRING", {"default": "1024", "placeholder": "如 1024, 512, 768"}),
                "height": ("STRING", {"default": "1024", "placeholder": "如 1024, 512, 768"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 1.5, "max": 20.0, "step": 0.1}),
                "loras": ("STRING", {"multiline": True, "default": "", "placeholder": "单LoRA: repo-id\n多LoRA: {\"id1\":0.6,\"id2\":0.4}"}),
                "timeout": ("INT", {"default": 300, "min": 60, "max": 600, "step": 10}),
                **{f"image_{i}": ("IMAGE",) for i in range(1, 7)}  # image_1 到 image_6
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "generate"
    CATEGORY = "哎呀✦MMX/图像"

    def tensor2pil_single(self, t: torch.Tensor) -> Image.Image:
        if t.dim() == 4:
            t = t.squeeze(0)
        t = (t.clamp(0, 1) * 255).byte().cpu()
        return Image.fromarray(t.numpy())

    def create_empty(self):
        return pil2tensor(Image.new("RGB", (64, 64), color=(0, 0, 0)))

    def generate(self, api_url, api_key, prompt, model, width, height,
                 negative_prompt="", seed=-1, steps=30, guidance=3.5, 
                 loras="", timeout=300, **image_ports):
        
        if not api_key.strip():
            err = "❌ API Key 为空\n请访问: https://modelscope.cn/my/myaccesstoken"
            return (self.create_empty(), err)

        # 构建 size 参数 WxH
        w_str = str(width).strip()
        h_str = str(height).strip()
        size = f"{w_str}x{h_str}"
        base_url = api_url.strip().rstrip("/")
        
        # 收集所有非空图像（支持1-6张）
        images_b64 = []
        for i in range(1, 7):
            img_tensor = image_ports.get(f"image_{i}")
            if img_tensor is not None:
                try:
                    pil_img = self.tensor2pil_single(img_tensor)
                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    b64 = base64.b64encode(buf.getvalue()).decode()
                    images_b64.append(f"data:image/png;base64,{b64}")
                except Exception as e:
                    print(f"[ModelScope] image_{i} 转换失败: {e}")

        # 构建 payload
        payload = {
            "model": model,
            "prompt": prompt,
            "size": size
        }
        
        if negative_prompt.strip():
            payload["negative_prompt"] = negative_prompt
        if seed >= 0:
            payload["seed"] = seed
        if steps != 30:
            payload["steps"] = steps
        if abs(guidance - 3.5) > 0.01:
            payload["guidance"] = guidance
        if images_b64:
            payload["image_url"] = images_b64

        # LoRA 处理
        if loras.strip():
            try:
                payload["loras"] = json.loads(loras)
            except:
                payload["loras"] = loras.strip()

        # 提交异步任务
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
            "X-ModelScope-Async-Mode": "true"
        }

        try:
            print(f"[ModelScope] 提交任务: {model} | {size} | 图像: {len(images_b64)}张")
            resp = requests.post(
                f"{base_url}/v1/images/generations",
                headers=headers,
                data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                timeout=30
            )
            
            if resp.status_code == 401:
                err_data = resp.json()
                if "bind your Alibaba Cloud account" in err_data.get("errors", {}).get("message", ""):
                    err_msg = ("❌ 账户未绑定阿里云\n"
                              "1. 访问 https://www.aliyun.com 注册/登录\n"
                              "2. 访问 https://modelscope.cn/my/account 绑定\n"
                              "3. 完成实名认证后重新生成 Token")
                    return (self.create_empty(), err_msg)
            
            resp.raise_for_status()
            data = resp.json()
            task_id = data.get("task_id")
            
            if not task_id:
                return (self.create_empty(), f"❌ 无 task_id: {data}")

        except Exception as e:
            return (self.create_empty(), f"❌ 提交失败: {str(e)}")

        # 轮询结果
        query_headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "X-ModelScope-Task-Type": "image_generation"
        }
        
        max_poll = timeout // 5
        for i in range(max_poll):
            time.sleep(5)
            try:
                result = requests.get(
                    f"{base_url}/v1/tasks/{task_id}",
                    headers=query_headers,
                    timeout=30
                )
                result.raise_for_status()
                data = result.json()
                status = data.get("task_status", "")

                if i == 0 or status in ["SUCCEED", "FAILED"] or (i+1) % 6 == 0:
                    print(f"[ModelScope] 轮询 {i+1}/{max_poll} | {status}")

                if status == "SUCCEED":
                    urls = data.get("output_images", [])
                    if not urls:
                        return (self.create_empty(), "❌ 无输出图片")
                    
                    img_data = requests.get(urls[0], timeout=60).content
                    pil_img = Image.open(io.BytesIO(img_data)).convert("RGB")
                    
                    mode_str = "图生图" if images_b64 else "文生图"
                    info = (f"✅ ModelScope {mode_str}成功\n"
                           f"Task: {task_id}\n"
                           f"Model: {model}\n"
                           f"Size: {size}\n"
                           f"输入图像: {len(images_b64)}张")
                    
                    return (pil2tensor(pil_img), info)

                elif status == "FAILED":
                    reason = data.get("message", "未知错误")
                    return (self.create_empty(), f"❌ 任务失败: {reason}")

            except Exception as e:
                continue

        return (self.create_empty(), f"❌ 超时 | Task: {task_id} 仍在运行")


# 统一注册
register_node(ModelScope_Image, "ModelScope_Image_mmx")