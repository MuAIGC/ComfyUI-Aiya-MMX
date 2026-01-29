"""
💡 核心特性：
   ✅ 音画同步：人物口型/情绪/节奏与语音高度一致（开启音频后自动生效）
   ✅ 三模式支持：文生视频 / 图生视频 / 首尾帧生视频
   ✅ 智能宽高比：自动适配抖音(9:16)、YouTube(16:9)、小红书(3:4)等平台
   ✅ 运镜语法：支持自然语言指令（"360度环绕运镜"）或专业语法（[推进][右摇]）
   
⚠️  重要限制（平台硬性要求）：
   • 视频时长：仅支持 5 秒 或 10 秒（文档写 4-12 秒为误导，实际仅 5/10 可用）
   • 1080p 分辨率：仅支持 5 秒（10 秒强制降级为 720p）
   • 首尾帧模式：尾帧自动裁剪至首帧尺寸（保持宽高比）
   • Seed 范围：必须 ≤ 4294967295 (2^32)

🎬 运镜技巧速查：
   • 自然语言："镜头缓慢推进，人物微笑后360度环绕"
   • 专业语法："[推进] 人物微笑，[右摇] 背景虚化，[环绕] 3秒"
   • 口播推荐："[固定镜头] 人物说'茄子'，微笑点头" + 开启音频
"""

import os
import re
import json
import time
import uuid
import base64
import requests
from pathlib import Path
from io import BytesIO
from PIL import Image
import numpy as np
import torch

import folder_paths
from ..register import register_node
from ..video_adapter import Video

# ══════════════════════════════════════════════════════════════════════════════
# 🔑 全局默认配置
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_API_URL = "https://www.dmxapi.cn"
API_VERSION = "v1"
MODEL_NAME = "doubao-seedance-1-5-pro-responses"
QUERY_MODEL = "seedance-get"

# 宽高比选项（带平台说明）
RATIO_OPTIONS = [
    "16:9（横屏-YouTube/腾讯视频）",
    "9:16（竖屏-抖音/快手）",
    "1:1（正方形-Instagram）",
    "4:3（复古屏）",
    "3:4（小红书封面）",
    "21:9（电影宽屏）",
    "adaptive（自动适配首帧）"
]

RESOLUTION_OPTIONS = ["480p", "720p", "1080p"]
# ✅ 关键修复：Seedance 实际仅支持 5/10 秒（文档 4-12 为误导）
DURATION_OPTIONS = ["5", "10"]  # 严格遵循 API 实际限制

# ✅ 修复：Seedance seed 上限为 2^32-1 (4294967295)
MAX_SEED = 4294967295

# ══════════════════════════════════════════════════════════════════════════════
# 🛠️ 通用工具函数（复用 Hailuo 风格）
# ══════════════════════════════════════════════════════════════════════════════

def _download_file(url: str, dst: Path, max_retry: int = 3, timeout: int = 120):
    """带重试的下载（复用 Hailuo 风格）"""
    for attempt in range(1, max_retry + 1):
        try:
            print(f"[Download] 第 {attempt}/{max_retry} 次：{url}")
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(dst, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            print(f"[Download] 成功 → {dst}")
            return
        except Exception as e:
            print(f"[Download] 第 {attempt} 次失败：{e}")
            if attempt == max_retry:
                raise RuntimeError(f"下载失败（重试 {max_retry} 次）：{e}")
            time.sleep(2 ** attempt)

def image_to_base64(img_tensor) -> str:
    """
    ComfyUI IMAGE tensor → 标准 Data URL（自动压缩到 <20MB，边长≤7680）
    ✅ 严格遵循 Data URL 标准：data:image/jpeg;base64,...
    :param img_tensor: ComfyUI IMAGE 格式 (B, H, W, C)
    :return: data:image/jpeg;base64,... 格式字符串
    """
    # 转 PIL Image
    img = img_tensor[0]  # 取第一帧
    img = (img * 255).clamp(0, 255).numpy().astype('uint8')
    pil_img = Image.fromarray(img).convert("RGB")
    
    # 限制最大边长 ≤7680
    max_edge = 7680
    if max(pil_img.size) > max_edge:
        ratio = max_edge / max(pil_img.size)
        new_size = (int(pil_img.size[0] * ratio), int(pil_img.size[1] * ratio))
        pil_img = pil_img.resize(new_size, Image.LANCZOS)
    
    # 质量压缩循环
    buffer = BytesIO()
    quality = 95
    while True:
        buffer.seek(0)
        buffer.truncate()
        pil_img.save(buffer, format="JPEG", quality=quality, optimize=True)
        if buffer.tell() < 19 * 1024 * 1024 or quality <= 10:
            break
        quality -= 5
    
    base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    # ✅ 修复：添加标准 data: 前缀（符合 Data URL 规范）
    return f"data:image/jpeg;base64,{base64_str}"

def build_video_obj(video_path: Path) -> Video:
    """把本地 mp4 封装成 ComfyUI VIDEO 对象（复用 Hailuo 风格）"""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return Video(str(video_path), fps, w, h)

# ══════════════════════════════════════════════════════════════════════════════
# 🧩 核心任务类（修复 URL 提取逻辑 + 支持自定义 API URL）
# ══════════════════════════════════════════════════════════════════════════════

class SeedanceTask:
    """Seedance 任务统一处理类"""
    
    @staticmethod
    def submit_task(
        api_url: str,
        token: str,
        prompt: str,
        first_frame_b64: str = None,
        last_frame_b64: str = None,
        resolution: str = "720p",
        ratio: str = "16:9",
        duration: int = 5,
        seed: int = -1,
        camera_fixed: bool = False,
        watermark: bool = False,
        generate_audio: bool = True
    ) -> str:
        """提交视频生成任务，返回 task_id"""
        # 构建 input 数组
        input_arr = [{"type": "text", "text": prompt.strip()}]
        
        if first_frame_b64:
            input_arr.append({
                "type": "image_url",
                "image_url": {"url": first_frame_b64},
                "role": "first_frame"
            })
        if last_frame_b64:
            input_arr.append({
                "type": "image_url",
                "image_url": {"url": last_frame_b64},
                "role": "last_frame"
            })
        
        # 清理 ratio 选项（移除中文说明）
        ratio_clean = ratio.split("（")[0].strip()
        
        # ✅ 修复：确保 seed 在有效范围内 (-1 或 0~4294967295)
        if seed > MAX_SEED:
            seed = seed % (MAX_SEED + 1)  # 对大seed取模
        if seed < -1:
            seed = -1
            
        payload = {
            "model": MODEL_NAME,
            "input": input_arr,
            "callback_url": "",
            "return_last_frame": False,
            "generate_audio": generate_audio,
            "resolution": resolution,
            "ratio": ratio_clean,
            "duration": duration,
            "seed": seed,
            "camera_fixed": camera_fixed,
            "watermark": watermark
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token.strip()}"
        }
        
        url = f"{api_url.rstrip('/')}/{API_VERSION}/responses"
        
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            result = resp.json()
            
            if "id" not in result:
                raise ValueError(f"❌ 任务提交失败: {result.get('error', '未知错误')}")
            
            task_id = result["id"]
            print(f"[Seedance] 任务提交成功: {task_id}")
            return task_id
            
        except requests.exceptions.RequestException as e:
            if "401" in str(e):
                raise RuntimeError("❌ API 密钥无效或已过期，请检查 DMXAPI_KEY 配置")
            # 增强错误诊断：打印 API 返回的具体错误
            try:
                error_detail = resp.json().get("error", resp.text[:200])
                raise RuntimeError(f"❌ 任务提交失败 (HTTP {resp.status_code}): {error_detail}")
            except:
                raise RuntimeError(f"❌ 任务提交失败: {str(e)}")
    
    @staticmethod
    def query_task(api_url: str, task_id: str, token: str) -> str:
        """流式查询任务进度并提取视频URL（增强健壮性）"""
        url = f"{api_url.rstrip('/')}/{API_VERSION}/responses"
        payload = {
            "model": QUERY_MODEL,
            "input": task_id,
            "stream": True
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token.strip()}"
        }
        
        video_url = None
        last_progress = 0
        
        # 进度映射表
        PROGRESS_MAP = {
            "response.output_text.delta": 70,
            "response.output_text.done": 80,
            "response.content_part.done": 85,
            "response.output_item.done": 90,
            "response.completed": 100
        }
        
        try:
            with requests.post(url, json=payload, headers=headers, stream=True, timeout=180) as resp:
                resp.raise_for_status()
                
                for line in resp.iter_lines():
                    if not line:
                        continue
                    
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith('event:') or line_str == '[DONE]':
                        continue
                    if line_str.startswith('data: '):
                        line_str = line_str[6:]
                    
                    try:
                        data = json.loads(line_str)
                        event_type = data.get('type', '')
                        
                        # 提取视频URL（关键！三重保险策略）
                        if event_type == "response.completed":
                            # 策略1：从 response.output[0].content[0].text 提取
                            text_content = data.get('response', {}).get('output', [{}])[0] \
                                         .get('content', [{}])[0].get('text', '')
                            
                            # 三重 URL 提取（按优先级）
                            url_candidates = []
                            
                            # 优先级1：匹配"视频URL: https://..."格式
                            match1 = re.search(r'视频URL[:：]?\s*(https://[^\s\n\)\]\'"]+)', text_content)
                            if match1:
                                url_candidates.append(match1.group(1).rstrip('.,;):]\'"'))
                            
                            # 优先级2：匹配纯 https:// 开头的 URL（更通用）
                            match2 = re.findall(r'(https://[a-zA-Z0-9\-._~:/?#\[\]@!$&\'()*+,;=%]+\.mp4[^\s]*)', text_content)
                            url_candidates.extend(match2)
                            
                            # 优先级3：匹配所有 https 链接（兜底）
                            match3 = re.findall(r'(https://[^\s\n\)\]\'"]+)', text_content)
                            url_candidates.extend(match3)
                            
                            # 清洗并验证 URL
                            for url in url_candidates:
                                url = url.rstrip('.,;):]\'"').strip()
                                if url.startswith("https://") and (".mp4" in url or ".mov" in url):
                                    video_url = url
                                    break
                            
                            # 调试诊断：提取失败时打印响应片段
                            if not video_url:
                                print(f"\n⚠️  未提取到视频URL，响应片段（前500字符）：")
                                print(f"   {text_content[:500]}")
                                raise RuntimeError("❌ 未从响应中提取到有效的视频URL")
                        
                        # 简化进度提示（仅关键节点输出）
                        if event_type in PROGRESS_MAP:
                            progress = PROGRESS_MAP[event_type]
                            if progress > last_progress:
                                print(f"[Seedance] 生成进度 {progress}% ({event_type})")
                                last_progress = progress
                        
                    except json.JSONDecodeError:
                        continue  # 忽略无法解析的行
                    except Exception as e:
                        if "未提取到视频URL" not in str(e):
                            print(f"[Seedance] 流处理异常: {e}")
                        continue
                
                if not video_url:
                    raise RuntimeError("❌ 任务完成但未提取到视频URL，请检查API响应格式")
                
                print(f"[Seedance] 音画同步视频生成完成！")
                return video_url
                
        except requests.exceptions.Timeout:
            raise RuntimeError("❌ 视频生成超时（超过180秒），请重试或缩短时长")
        except Exception as e:
            raise RuntimeError(f"❌ 任务查询失败: {str(e)}")

# ══════════════════════════════════════════════════════════════════════════════
# 🎞️ 节点实现（统一 CATEGORY 和注册风格 + 外显 API URL）
# ══════════════════════════════════════════════════════════════════════════════

class SeedanceText2Video:
    """🎵 Seedance-文生视频（原生音画同步）"""
    
    DESCRIPTION = """
💡 豆包 Seedance 1.5 Pro - 原生音画同步视频生成
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ 核心优势：
   • 音频+视频联合生成：人物口型/情绪/节奏与语音高度一致
   • 专业级运镜：支持自然语言指令（"360度环绕运镜"）或专业语法（[推进][右摇]）
   • 多平台适配：自动优化抖音(9:16)、YouTube(16:9)、小红书(3:4)等尺寸

⚠️  重要限制（平台硬性要求）：
   • 视频时长：仅支持 5 秒 或 10 秒（文档写 4-12 秒为误导，实际仅 5/10 可用）
   • 1080p 分辨率：仅支持 5 秒（10 秒强制降级为 720p）
   • Seed 随机种子：必须 ≤ 4294967295 (2^32)
   • 生成音频后视频体积增大 30%~50%

🎬 运镜技巧：
   • 基础指令："[推进] 人物微笑"、"[右摇] 背景虚化"
   • 高级组合："[环绕] 3秒 + 人物说'茄子'，微笑点头"
   • 口播推荐：固定镜头 + 开启音频 → 生成专业口播视频
    """
    
    RETURN_TYPES = ("VIDEO", "STRING", "INT")
    RETURN_NAMES = ("video", "download_url", "seed")
    FUNCTION = "generate"
    CATEGORY = "哎呀✦MMX/DMXAPI"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_url": ("STRING", {"default": DEFAULT_API_URL, "multiline": False}),
                "dmxapi_key": ("STRING", {"default": "sk-", "multiline": False}),
                "prompt": ("STRING", {
                    "default": "图中女孩对着镜头说'茄子'，360度环绕运镜",
                    "multiline": True,
                    "placeholder": "描述画面+运镜+口播内容（支持自然语言运镜指令）"
                }),
                "resolution": (RESOLUTION_OPTIONS, {"default": "720p"}),
                "ratio": (RATIO_OPTIONS, {"default": "9:16（竖屏-抖音/快手）"}),
                "duration": (DURATION_OPTIONS, {"default": "5"}),
                # ✅ 修复：seed 上限改为 4294967295 (2^32-1)
                "seed": ("INT", {"default": -1, "min": -1, "max": MAX_SEED}),
            },
            "optional": {
                "generate_audio": (["开启（推荐）", "关闭"], {"default": "开启（推荐）"}),
                "camera_fixed": (["运镜移动", "固定镜头"], {"default": "运镜移动"}),
                "watermark": (["无水印", "添加水印"], {"default": "无水印"}),
            }
        }
    
    def generate(self, api_url, dmxapi_key, prompt, resolution, ratio, duration, seed,
                 generate_audio="开启（推荐）", camera_fixed="运镜移动", watermark="无水印"):
        # ✅ 关键校验：时长必须为 5 或 10
        if duration not in ["5", "10"]:
            raise RuntimeError("❌ 时长仅支持 5 秒或 10 秒（平台硬性限制）")
        
        # ✅ 1080p + 10秒 自动降级（避免报错）
        if resolution == "1080p" and duration == "10":
            print("⚠️  1080p 仅支持 5 秒，10 秒时长将自动降级为 720p")
            resolution = "720p"
        
        # 参数校验
        token = dmxapi_key.strip()
        if not token or token == "sk-":
            raise RuntimeError("❌ 请在 dmxapi_key 字段填入有效的 DMXAPI 密钥（格式: sk-xxxx）")
        
        if not prompt.strip():
            raise RuntimeError("❌ 提示词不能为空")
        
        # 提交任务
        task_id = SeedanceTask.submit_task(
            api_url=api_url,
            token=token,
            prompt=prompt.strip(),
            resolution=resolution,
            ratio=ratio,
            duration=int(duration),
            seed=seed,
            camera_fixed=(camera_fixed == "固定镜头"),
            watermark=(watermark == "添加水印"),
            generate_audio=(generate_audio == "开启（推荐）")
        )
        
        # 流式查询进度
        video_url = SeedanceTask.query_task(api_url, task_id, token)
        
        # 下载视频
        output_dir = Path(folder_paths.get_output_directory()) / "seedance"
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir / f"seedance_{uuid.uuid4().hex[:8]}.mp4"
        _download_file(video_url, video_path)
        
        # 封装为 ComfyUI VIDEO 对象
        video = build_video_obj(video_path)
        print(f"[Seedance-T2V] VIDEO 对象已生成：{video}")
        
        return (video, video_url, seed)


class SeedanceImage2Video:
    """🖼️ Seedance-图生视频（首帧控制+音画同步）"""
    
    DESCRIPTION = """
💡 豆包 Seedance 1.5 Pro - 基于首帧图片生成音画同步视频
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ 核心优势：
   • 首帧精准控制：基于上传图片生成连贯视频
   • 智能宽高比：自动检测图片尺寸推荐最佳 ratio
   • 音画同步：开启音频后，人物动作与语音节奏匹配

⚠️  重要限制：
   • 视频时长：仅支持 5 秒 或 10 秒
   • 1080p 分辨率：仅支持 5 秒（10 秒强制降级为 720p）
   • Seed 随机种子：必须 ≤ 4294967295 (2^32)
   • 首帧分辨率建议 ≥720p 以保证生成质量
    """
    
    RETURN_TYPES = ("VIDEO", "STRING", "INT")
    RETURN_NAMES = ("video", "download_url", "seed")
    FUNCTION = "generate"
    CATEGORY = "哎呀✦MMX/DMXAPI"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_url": ("STRING", {"default": DEFAULT_API_URL, "multiline": False}),
                "dmxapi_key": ("STRING", {"default": "sk-", "multiline": False}),
                "first_frame": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "镜头缓慢推进，人物微笑说'欢迎来到我的频道'",
                    "multiline": True,
                    "placeholder": "描述画面变化+运镜+口播内容"
                }),
                "resolution": (RESOLUTION_OPTIONS, {"default": "720p"}),
                "ratio": (RATIO_OPTIONS, {"default": "adaptive（自动适配首帧）"}),
                "duration": (DURATION_OPTIONS, {"default": "5"}),
                # ✅ 修复：seed 上限改为 4294967295 (2^32-1)
                "seed": ("INT", {"default": -1, "min": -1, "max": MAX_SEED}),
            },
            "optional": {
                "generate_audio": (["开启（推荐）", "关闭"], {"default": "开启（推荐）"}),
                "camera_fixed": (["运镜移动", "固定镜头"], {"default": "运镜移动"}),
                "watermark": (["无水印", "添加水印"], {"default": "无水印"}),
            }
        }
    
    def generate(self, api_url, dmxapi_key, first_frame, prompt, resolution, ratio, duration, seed,
                 generate_audio="开启（推荐）", camera_fixed="运镜移动", watermark="无水印"):
        # ✅ 关键校验：时长必须为 5 或 10
        if duration not in ["5", "10"]:
            raise RuntimeError("❌ 时长仅支持 5 秒或 10 秒（平台硬性限制）")
        
        # ✅ 1080p + 10秒 自动降级
        if resolution == "1080p" and duration == "10":
            print("⚠️  1080p 仅支持 5 秒，10 秒时长将自动降级为 720p")
            resolution = "720p"
        
        # 参数校验
        token = dmxapi_key.strip()
        if not token or token == "sk-":
            raise RuntimeError("❌ 请在 dmxapi_key 字段填入有效的 DMXAPI 密钥（格式: sk-xxxx）")
        
        if not prompt.strip():
            raise RuntimeError("❌ 提示词不能为空")
        
        # 处理首帧图片
        first_frame_b64 = image_to_base64(first_frame)
        
        # 智能宽高比推荐（仅当用户选择 adaptive 时）
        if ratio.startswith("adaptive"):
            w = first_frame.shape[2]
            h = first_frame.shape[1]
            ratio_val = w / h
            
            if abs(ratio_val - 9/16) < 0.1:
                ratio = "9:16（竖屏-抖音/快手）"
            elif abs(ratio_val - 16/9) < 0.1:
                ratio = "16:9（横屏-YouTube/腾讯视频）"
            elif abs(ratio_val - 3/4) < 0.1:
                ratio = "3:4（小红书封面）"
            else:
                ratio = "1:1（正方形-Instagram）"
            print(f"[Seedance-I2V] 检测到首帧尺寸 {w}x{h}，自动推荐宽高比: {ratio}")
        
        # 提交任务
        task_id = SeedanceTask.submit_task(
            api_url=api_url,
            token=token,
            prompt=prompt.strip(),
            first_frame_b64=first_frame_b64,
            resolution=resolution,
            ratio=ratio,
            duration=int(duration),
            seed=seed,
            camera_fixed=(camera_fixed == "固定镜头"),
            watermark=(watermark == "添加水印"),
            generate_audio=(generate_audio == "开启（推荐）")
        )
        
        # 流式查询进度
        video_url = SeedanceTask.query_task(api_url, task_id, token)
        
        # 下载视频
        output_dir = Path(folder_paths.get_output_directory()) / "seedance"
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir / f"seedance_{uuid.uuid4().hex[:8]}.mp4"
        _download_file(video_url, video_path)
        
        # 封装为 ComfyUI VIDEO 对象
        video = build_video_obj(video_path)
        print(f"[Seedance-I2V] VIDEO 对象已生成：{video}")
        
        return (video, video_url, seed)


class SeedanceFirstLastFrame2Video:
    """🎞️ Seedance-首尾帧生视频（双帧控制+音画同步）"""
    
    DESCRIPTION = """
💡 豆包 Seedance 1.5 Pro - 基于首尾帧生成过渡视频（含音画同步）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ 核心优势：
   • 双帧精准控制：指定起始/结束画面，AI 生成自然过渡
   • 自动尺寸适配：尾帧自动裁剪至首帧尺寸（保持比例）
   • 音画同步：音频节奏匹配画面过渡变化

⚠️  重要限制：
   • 视频时长：仅支持 5 秒 或 10 秒
   • 1080p 分辨率：仅支持 5 秒（10 秒强制降级为 720p）
   • Seed 随机种子：必须 ≤ 4294967295 (2^32)
   • 首尾帧建议使用相同主体（如人脸），否则过渡可能不自然
    """
    
    RETURN_TYPES = ("VIDEO", "STRING", "INT")
    RETURN_NAMES = ("video", "download_url", "seed")
    FUNCTION = "generate"
    CATEGORY = "哎呀✦MMX/DMXAPI"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_url": ("STRING", {"default": DEFAULT_API_URL, "multiline": False}),
                "dmxapi_key": ("STRING", {"default": "sk-", "multiline": False}),
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "镜头从首帧平滑过渡至尾帧，人物自然转身",
                    "multiline": True,
                    "placeholder": "描述过渡方式+运镜+口播内容"
                }),
                "resolution": (RESOLUTION_OPTIONS, {"default": "720p"}),
                "ratio": (RATIO_OPTIONS, {"default": "adaptive（自动适配首帧）"}),
                "duration": (DURATION_OPTIONS, {"default": "5"}),
                # ✅ 修复：seed 上限改为 4294967295 (2^32-1)
                "seed": ("INT", {"default": -1, "min": -1, "max": MAX_SEED}),
            },
            "optional": {
                "generate_audio": (["开启（推荐）", "关闭"], {"default": "开启（推荐）"}),
                "watermark": (["无水印", "添加水印"], {"default": "无水印"}),
            }
        }
    
    def generate(self, api_url, dmxapi_key, first_frame, last_frame, prompt, resolution, duration, seed, ratio,
                 generate_audio="开启（推荐）", watermark="无水印"):
        # ✅ 关键校验：时长必须为 5 或 10
        if duration not in ["5", "10"]:
            raise RuntimeError("❌ 时长仅支持 5 秒或 10 秒（平台硬性限制）")
        
        # ✅ 1080p + 10秒 自动降级
        if resolution == "1080p" and duration == "10":
            print("⚠️  1080p 仅支持 5 秒，10 秒时长将自动降级为 720p")
            resolution = "720p"
        
        # 参数校验
        token = dmxapi_key.strip()
        if not token or token == "sk-":
            raise RuntimeError("❌ 请在 dmxapi_key 字段填入有效的 DMXAPI 密钥（格式: sk-xxxx）")
        
        if not prompt.strip():
            raise RuntimeError("❌ 提示词不能为空")
        
        # 处理首帧
        first_frame_b64 = image_to_base64(first_frame)
        
        # 处理尾帧（自动适配首帧尺寸）
        first_w = first_frame.shape[2]
        first_h = first_frame.shape[1]
        
        last_img = last_frame[0]
        last_img = (last_img * 255).clamp(0, 255).numpy().astype('uint8')
        pil_last = Image.fromarray(last_img).convert("RGB")
        pil_last.thumbnail((first_w, first_h), Image.LANCZOS)
        
        if pil_last.size != (first_w, first_h):
            bg = Image.new('RGB', (first_w, first_h), (0, 0, 0))
            offset = ((first_w - pil_last.size[0]) // 2, (first_h - pil_last.size[1]) // 2)
            bg.paste(pil_last, offset)
            pil_last = bg
        
        last_np = np.array(pil_last).astype(np.float32) / 255.0
        last_tensor = torch.from_numpy(last_np).unsqueeze(0)
        last_frame_b64 = image_to_base64(last_tensor)
        
        print(f"[Seedance-FL2V] 尾帧已自动裁剪至 {first_w}x{first_h} 以匹配首帧")
        
        # 清理 ratio 参数（移除中文说明）
        ratio_clean = ratio.split("（")[0].strip()
        
        # 提交任务
        task_id = SeedanceTask.submit_task(
            api_url=api_url,
            token=token,
            prompt=prompt.strip(),
            first_frame_b64=first_frame_b64,
            last_frame_b64=last_frame_b64,
            resolution=resolution,
            ratio=ratio_clean,
            duration=int(duration),
            seed=seed,
            camera_fixed=False,
            watermark=(watermark == "添加水印"),
            generate_audio=(generate_audio == "开启（推荐）")
        )
        
        # 流式查询进度
        video_url = SeedanceTask.query_task(api_url, task_id, token)
        
        # 下载视频
        output_dir = Path(folder_paths.get_output_directory()) / "seedance"
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir / f"seedance_{uuid.uuid4().hex[:8]}.mp4"
        _download_file(video_url, video_path)
        
        # 封装为 ComfyUI VIDEO 对象
        video = build_video_obj(video_path)
        print(f"[Seedance-FL2V] VIDEO 对象已生成：{video}")
        
        return (video, video_url, seed)


# ══════════════════════════════════════════════════════════════════════════════
# 🔌 节点注册（统一使用 register_node 风格）
# ══════════════════════════════════════════════════════════════════════════════

register_node(SeedanceText2Video, "Seedance15Pro-文生视频-DMX")
register_node(SeedanceImage2Video, "Seedance15Pro-图生视频-DMX")
register_node(SeedanceFirstLastFrame2Video, "Seedance15Pro-首尾帧生视频-DMX")