# ~/ComfyUI/custom_nodes/ComfyUI-Aiya-MMX/nodes/openai_API.py
from __future__ import annotations
import io
import json
import base64
import time
import requests
import torch
from pathlib import Path
from PIL import Image
from ..register import register_node
from ..mmx_utils import pil2tensor, tensor2pil

# ---------- 通用工具 ----------
def tensor2pil_single(t: torch.Tensor) -> Image.Image:
    """严格单张转换"""
    if t.dim() == 4:
        t = t.squeeze(0)
    t = (t.clamp(0, 1) * 255).byte().cpu()
    return Image.fromarray(t.numpy())

def decode_b64_to_tensor(b64_str: str):
    """base64 转张量"""
    img_bytes = base64.b64decode(b64_str)
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return pil2tensor(pil)

def get_empty_image(h=1024, w=1024):
    """返回指定尺寸黑图"""
    return torch.zeros(1, h, w, 3)


# ===================================================================
#  1. GPT-Image 文生图（支持自动分批请求突破平台限制）
# ===================================================================
class GPTImageGenerate:
    DESCRIPTION = (
        "💕 哎呀✦GPT-Image 文生图\n"
        "支持 gpt-image-1.5，自动分批获取多张图"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_url": ("STRING", {"default": "https://ai.t8star.cn/v1/images/generations"}),
                "api_key": ("STRING", {"default": "", "placeholder": "sk-***"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": ("STRING", {"default": "gpt-image-1.5", "placeholder": "gpt-image-1.5"}),
                "size": ([
                    "1024x1024 (正方形)",
                    "1536x1024 (横版)", 
                    "1024x1536 (竖版)",
                    "auto (自动)"
                ], {"default": "1024x1024 (正方形)"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10}),
                "quality": ([
                    "auto (自动)",
                    "high (高)",
                    "medium (中)",
                    "low (低)"
                ], {"default": "auto (自动)"}),
            },
            "optional": {
                "background": ([
                    "auto (自动)",
                    "transparent (透明)",
                    "opaque (不透明)"
                ], {"default": "auto (自动)"}),
                "output_format": (["jpeg", "png", "webp"], {"default": "jpeg"}),
                "output_compression": ("INT", {"default": 90, "min": 0, "max": 100}),
                "moderation": ([
                    "auto (自动)",
                    "low (宽松)"
                ], {"default": "low (宽松)"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "generate"
    CATEGORY = "哎呀✦MMX/图像"

    def parse_option(self, option_str: str) -> str:
        """从中文标签提取实际值"""
        return option_str.split(" ")[0]

    def process_single_image(self, img_data: dict, index: int) -> torch.Tensor:
        """处理单张图片数据，失败返回 None"""
        try:
            if "b64_json" in img_data and img_data["b64_json"]:
                return decode_b64_to_tensor(img_data["b64_json"])
            elif "url" in img_data and img_data["url"]:
                url = img_data["url"]
                print(f"[GPTImage] 图{index+1} 下载URL: {url[:40]}...")
                img_resp = requests.get(url, timeout=60)
                img_resp.raise_for_status()
                pil_img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
                return pil2tensor(pil_img)
            else:
                print(f"[GPTImage] ⚠️ 图{index+1} 无有效数据")
                return None
        except Exception as e:
            print(f"[GPTImage] ⚠️ 图{index+1} 处理失败: {e}")
            return None

    def request_single_batch(self, api_url: str, headers: dict, payload: dict, 
                            batch_idx: int, total_batches: int) -> list:
        """发送单次请求，返回 tensor 列表"""
        try:
            print(f"[GPTImage] 第 {batch_idx}/{total_batches} 批请求 (n={payload['n']})...")
            resp = requests.post(api_url, headers=headers, json=payload, timeout=180)
            resp.raise_for_status()
            data = resp.json()

            if "data" not in data or not isinstance(data["data"], list):
                print(f"[GPTImage] ⚠️ 第 {batch_idx} 批返回异常: {list(data.keys())}")
                return []

            batch_tensors = []
            for idx, img_item in enumerate(data["data"]):
                tensor = self.process_single_image(img_item, idx)
                if tensor is not None:
                    if tensor.dim() == 3:
                        tensor = tensor.unsqueeze(0)
                    batch_tensors.append(tensor)
            
            print(f"[GPTImage] 第 {batch_idx} 批成功获取 {len(batch_tensors)} 张")
            return batch_tensors

        except Exception as e:
            print(f"[GPTImage] ⚠️ 第 {batch_idx} 批请求失败: {e}")
            return []

    def generate(self, api_url, api_key, prompt, model, size, n, quality,
                 background="auto (自动)", output_format="jpeg", 
                 output_compression=90, moderation="auto (自动)"):
        
        if not api_key.strip():
            print("[GPTImage] ❌ API Key 缺失")
            return (get_empty_image(), "Error: API Key 缺失")

        # 解析参数
        size_val = self.parse_option(size)
        quality_val = self.parse_option(quality)
        bg_val = self.parse_option(background)
        mod_val = self.parse_option(moderation)

        # 平台限制每批最多 2 张，计算分批
        MAX_PER_BATCH = 2
        total_needed = n
        batches = []
        
        remaining = total_needed
        while remaining > 0:
            current_batch = min(remaining, MAX_PER_BATCH)
            batches.append(current_batch)
            remaining -= current_batch

        print(f"[GPTImage] 需要 {total_needed} 张图，分 {len(batches)} 次请求: {batches}")

        headers = {
            "Authorization": f"{api_key}",
            "Content-Type": "application/json"
        }

        all_tensors = []
        success_count = 0
        
        # 循环发送请求
        for batch_idx, batch_n in enumerate(batches, 1):
            payload = {
                "model": model.strip(),
                "prompt": prompt,
                "n": batch_n,
                "size": size_val,
                "quality": quality_val,
                "background": bg_val,
                "output_format": output_format,
                "output_compression": output_compression,
                "moderation": mod_val,
            }
            
            batch_tensors = self.request_single_batch(
                api_url, headers, payload, batch_idx, len(batches)
            )
            
            all_tensors.extend(batch_tensors)
            success_count += len(batch_tensors)
            
            # 简单防速率限制，每批间隔 0.5 秒（最后一批不用等）
            if batch_idx < len(batches):
                time.sleep(0.5)

        # 如果全部失败，返回空图
        if not all_tensors:
            return (get_empty_image(), "Error: 所有批次请求均失败")

        # 如果成功数量不足，用黑图补齐（保持用户要求的 n 张）
        while len(all_tensors) < total_needed:
            all_tensors.append(get_empty_image(1024, 1536 if "1536" in size_val else 1024))
            print(f"[GPTImage] 用空白图补齐 1 张")

        # 合并为 batch: [B, H, W, 3]
        batched = torch.cat(all_tensors[:total_needed], dim=0)  # 只取前 n 张，防止 API 多给
        actual_returned = success_count

        print(f"[GPTImage] ✅ 总计成功 {actual_returned}/{total_needed} 张，batch形状: {batched.shape}")

        info = (
            f"🎨 GPT-Image Generate | {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"模型: {model} | 尺寸: {size_val} | 质量: {quality_val}\n"
            f"背景: {bg_val} | 格式: {output_format} | 压缩: {output_compression}\n"
            f"请求: {total_needed}张 | 分批: {len(batches)}次 | 实际获取: {actual_returned}张\n"
            f"提示词: {prompt[:50]}{'...' if len(prompt) > 50 else ''}"
        )
        
        return (batched, info)


# ===================================================================
#  2. GPT-Image 图像编辑（修复版：支持16张图输入，修复响应解析）
# ===================================================================
class GPTImageEdit:
    DESCRIPTION = (
        "💕 哎呀✦GPT-Image 图像编辑\n"
        "支持最多16张参考图，单张输出（编辑API不支持n参数）"
    )

    @classmethod
    def INPUT_TYPES(cls):
        # 动态生成16个图像输入端口
        optional_inputs = {
            f"reference_image_{i}": ("IMAGE",) 
            for i in range(1, 17)
        }
        
        return {
            "required": {
                "api_url": ("STRING", {"default": "https://ai.t8star.cn/v1/images/edits"}),
                "api_key": ("STRING", {"default": "", "placeholder": "sk-***"}),
                "prompt": ("STRING", {"multiline": True, "default": "给人物添加一副墨镜，保持风格一致"}),
                "model": ("STRING", {"default": "gpt-image-1.5"}),
                "size": ([
                    "1024x1024 (正方形)",
                    "1536x1024 (横版)",
                    "1024x1536 (竖版)",
                    "auto (自动)"
                ], {"default": "1024x1024 (正方形)"}),
            },
            "optional": {
                "quality": ([
                    "auto (自动)",
                    "high (高)",
                    "medium (中)",
                    "low (低)"
                ], {"default": "auto (自动)"}),
                "background": ([
                    "auto (自动)",
                    "transparent (透明)",
                    "opaque (不透明)"
                ], {"default": "auto (自动)"}),
                "output_format": (["jpeg", "png", "webp"], {"default": "jpeg"}),
                "output_compression": ("INT", {"default": 90, "min": 0, "max": 100}),
                "input_fidelity": (["low (低保真)", "high (高保真)"], {"default": "low (低保真)"}),
                **optional_inputs
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "edit"
    CATEGORY = "哎呀✦MMX/图像"

    def parse_option(self, option_str: str) -> str:
        return option_str.split(" ")[0]

    def tensor_to_bytes(self, tensor: torch.Tensor, fmt: str = "PNG") -> bytes:
        """张量转为字节流"""
        pil = tensor2pil_single(tensor)
        buf = io.BytesIO()
        pil.save(buf, format=fmt)
        buf.seek(0)
        return buf.getvalue()

    def edit(self, api_url, api_key, prompt, model, size, 
             quality="auto (自动)", background="auto (自动)", 
             output_format="jpeg", output_compression=90, 
             input_fidelity="low (低保真)", **kwargs):
        
        if not api_key.strip():
            print("[GPTImageEdit] ❌ API Key 缺失")
            return (get_empty_image(), "Error: API Key 缺失")

        # 收集所有输入的参考图（reference_image_1 到 reference_image_16）
        images = []
        for i in range(1, 17):
            key = f"reference_image_{i}"
            if key in kwargs and kwargs[key] is not None:
                images.append(kwargs[key])
        
        if not images:
            print("[GPTImageEdit] ❌ 至少需要提供1张参考图")
            return (get_empty_image(), "Error: 至少需要1张参考图")

        print(f"[GPTImageEdit] 收到 {len(images)} 张参考图，准备上传...")

        size_val = self.parse_option(size)
        quality_val = self.parse_option(quality)
        bg_val = self.parse_option(background)
        fidelity_val = self.parse_option(input_fidelity)

        # 构建 multipart/form-data，支持多图上传
        files = []
        for idx, img_tensor in enumerate(images):
            img_bytes = self.tensor_to_bytes(img_tensor, "PNG")
            files.append(
                ("image", (f"input_{idx+1}.png", io.BytesIO(img_bytes), "image/png"))
            )

        data = {
            "model": model.strip(),
            "prompt": prompt,
            "size": size_val,
            "quality": quality_val,
            "background": bg_val,
            "output_format": output_format,
            "output_compression": str(output_compression),
        }
        
        # fidelity 不支持 1-mini
        if "1-mini" not in model:
            data["input_fidelity"] = fidelity_val

        headers = {"Authorization": f"{api_key}"}

        try:
            print(f"[GPTImageEdit] 发送请求: {model} | 保真: {fidelity_val} | 上传 {len(images)} 张图")
            resp = requests.post(api_url, headers=headers, data=data, files=files, timeout=180)
            resp.raise_for_status()
            result = resp.json()

            # 🔍 关键调试：打印完整响应
            debug_str = json.dumps(result, ensure_ascii=False, indent=2)[:800]
            print(f"[GPTImageEdit] API 原始响应:\n{debug_str}...")

            # 检查错误
            if "error" in result:
                err_msg = result["error"].get("message", "未知错误")
                print(f"[GPTImageEdit] ❌ API 返回错误: {err_msg}")
                return (get_empty_image(), f"API Error: {err_msg}")

            if "data" not in result or not result["data"]:
                print(f"[GPTImageEdit] ⚠️ 响应无 data 字段，实际字段: {list(result.keys())}")
                return (get_empty_image(), "Error: 响应无图像数据")

            # 编辑API通常只返回1张图，取第一张处理
            img_data = result["data"][0]
            
            # 尝试多种格式解析
            tensor = None
            if "b64_json" in img_data and img_data["b64_json"]:
                try:
                    tensor = decode_b64_to_tensor(img_data["b64_json"])
                    print(f"[GPTImageEdit] ✅ 解码 base64 成功")
                except Exception as e:
                    print(f"[GPTImageEdit] ⚠️ base64 解码失败: {e}")
            
            elif "url" in img_data and img_data["url"]:
                try:
                    url = img_data["url"]
                    print(f"[GPTImageEdit] 下载URL: {url[:50]}...")
                    img_resp = requests.get(url, timeout=60)
                    img_resp.raise_for_status()
                    pil_img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
                    tensor = pil2tensor(pil_img)
                    print(f"[GPTImageEdit] ✅ URL 下载成功")
                except Exception as e:
                    print(f"[GPTImageEdit] ⚠️ URL 下载失败: {e}")
            
            else:
                # 检查是否有其他字段（如 revised_prompt 等元数据）
                print(f"[GPTImageEdit] ⚠️ 无 b64_json 或 url，可用字段: {list(img_data.keys())}")
                return (get_empty_image(), f"Error: 无法解析图像，字段: {list(img_data.keys())}")

            if tensor is None:
                return (get_empty_image(), "Error: 图像解码失败")

            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)

            info = (
                f"✏️ GPT-Image Edit | {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"模型: {model} | 尺寸: {size_val} | 保真: {fidelity_val}\n"
                f"上传: {len(images)}张参考图 | 输出格式: {output_format}\n"
                f"提示词: {prompt[:50]}{'...' if len(prompt) > 50 else ''}"
            )
            
            print(f"[GPTImageEdit] ✅ 编辑成功，输出形状: {tensor.shape}")
            return (tensor, info)

        except requests.exceptions.HTTPError as e:
            err_text = e.response.text if e.response else str(e)
            print(f"[GPTImageEdit] ❌ HTTP 错误: {err_text[:200]}")
            return (get_empty_image(), f"HTTP Error: {err_text[:200]}")
        except Exception as e:
            err_msg = f"[GPTImageEdit] ❌ 请求失败: {str(e)}"
            print(err_msg)
            return (get_empty_image(), err_msg)


# ===================================================================
#  统一注册
# ===================================================================
register_node(GPTImageGenerate, "GPTImage_Generate_mmx")
register_node(GPTImageEdit, "GPTImage_Edit_mmx")