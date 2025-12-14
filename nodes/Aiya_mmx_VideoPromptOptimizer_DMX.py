# Aiya_mmx_VideoPromptOptimizer_DMX.py
from __future__ import annotations
import io
import base64
import requests
from PIL import Image
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


def image_to_data_url(image_tensor) -> str:
    pil = tensor2pil(image_tensor)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{b64}"


# ---------- 20 套纯分镜预设 ----------
PRESET_PROMPTS = {
    "默认（纯分镜）": (
        "按 {} 纯分镜，仅输出两段≤200 字/词提示词本体，先中文后英文，各一段，用换行分隔："
        "0s-首帧/中段-主体动作/末段-落幅"
    ),
    "黑白分镜": (
        "按 {} 纯黑白分镜，仅输出两段≤200 字/词提示词本体，先中文后英文，各一段，用换行分隔："
        "0s-高反差硬影入场，中段-轮廓摇移，末段-颗粒定格"
    ),
    "超现实分镜": (
        "按 {} 纯超现实分镜，仅输出两段≤200 字/词提示词本体，先中文后英文，各一段，用换行分隔："
        "0s-反向缩放入场，中段-漂浮+粒子，末段-柔焦消散"
    ),
    "史诗电影": (
        "按 {} 纯史诗分镜，仅输出两段≤200 字/词提示词本体，先中文后英文，各一段，用换行分隔："
        "0s-广角推进，中段-英雄慢动作+尘埃光，末段-航拍拉远"
    ),
    "一镜到底": (
        "按 {} 纯一镜分镜，仅输出两段≤200 字/词提示词本体，先中文后英文，各一段，用换行分隔："
        "0s-手持进入，中段-穿越人群+回眸，末段-霓虹定格"
    ),
    "赛博朋克": (
        "按 {} 纯赛博分镜，仅输出两段≤200 字/词提示词本体，先中文后英文，各一段，用换行分隔："
        "0s-雨夜俯冲，中段-扫描线+剪影，末段-青品对撞定格"
    ),
    "极简留白": (
        "按 {} 纯极简分镜，仅输出两段≤200 字/词提示词本体，先中文后英文，各一段，用换行分隔："
        "0s-对称静帧，中段-主体缓移，末段-轻推留白"
    ),
    "恐怖惊悚": (
        "按 {} 纯惊悚分镜，仅输出两段≤200 字/词提示词本体，先中文后英文，各一段，用换行分隔："
        "0s-手电闪烁登场，中段-突回身+拉伸影，末段-低频鼓定格"
    ),
    "复古 8mm": (
        "按 {} 纯复古分镜，仅输出两段≤200 字/词提示词本体，先中文后英文，各一段，用换行分隔："
        "0s-颗粒漏光入场，中段-跳切+晃动，末段-暖黄褪色定格"
    ),
    "微距特写": (
        "按 {} 纯微距分镜，仅输出两段≤200 字/词提示词本体，先中文后英文，各一段，用换行分隔："
        "0s-焦点堆栈入场，中段-细节滑动，末段-浅景消散"
    ),
    "高速冲击": (
        "按 {} 纯高速分镜，仅输出两段≤200 字/词提示词本体，先中文后英文，各一段，用换行分隔："
        "0s-120 fps 入场，中段-撞击+灰尘环，末段-速度线定格"
    ),
    "蒸汽波": (
        "按 {} 纯蒸汽波分镜，仅输出两段≤200 字/词提示词本体，先中文后英文，各一段，用换行分隔："
        "0s-粉青网格入场，中段-慵懒摆动，末段-扫描线定格"
    ),
    "哥特暗黑": (
        "按 {} 纯哥特分镜，仅输出两段≤200 字/词提示词本体，先中文后英文，各一段，用换行分隔："
        "0s-烛光高对比入场，中段-慢移+推近，末段-暗角颗粒定格"
    ),
    "迪士尼 3D": (
        "按 {} 纯迪士尼分镜，仅输出两段≤200 字/词提示词本体，先中文后英文，各一段，用换行分隔："
        "0s-圆润饱和入场，中段-弹性动作，末段-镜头弹跳定格"
    ),
    "航拍广角": (
        "按 {} 纯航拍分镜，仅输出两段≤200 字/词提示词本体，先中文后英文，各一段，用换行分隔："
        "0s-无人机俯冲入镜，中段-小比例主体，末段-云影流动定格"
    ),
    "魔法粒子": (
        "按 {} 纯魔法分镜，仅输出两段≤200 字/词提示词本体，先中文后英文，各一段，用换行分隔："
        "0s-点光源+拖尾入场，中段-施法环绕，末段-紫青渐变定格"
    ),
    "废土末日": (
        "按 {} 纯废土分镜，仅输出两段≤200 字/词提示词本体，先中文后英文，各一段，用换行分隔："
        "0s-强逆光沙尘入场，中段-跋涉晃动，末段-灰尘前景定格"
    ),
    "柔焦少女": (
        "按 {} 纯少女分镜，仅输出两段≤200 字/词提示词本体，先中文后英文，各一段，用换行分隔："
        "0s-柔光粉调入镜，中段-微笑轻推，末段-光斑前景定格"
    ),
    "像素复古": (
        "按 {} 纯像素分镜，仅输出两段≤200 字/词提示词本体，先中文后英文，各一段，用换行分隔："
        "0s-8-bit 调色入场，中段-跳跃步进，末段-扫描线定格"
    ),
    "纪录片写实": (
        "按 {} 纯纪录分镜，仅输出两段≤200 字/词提示词本体，先中文后英文，各一段，用换行分隔："
        "0s-自然光中焦入场，中段-访谈呼吸，末段-环境留白定格"
    ),
}


# ---------- 节点 ----------
class VideoPromptOptimizer_DMX:
    DESCRIPTION = (
        "💕 哎呀✦纯分镜提示词优化（时分镜注入）\n\n"
        "下拉选风格 | 支持自定义系统提示 | 可设总时长（3-20s）\n"
        "输出中文+英文两份≤200 字「时分镜式」专业视频提示词\n"
    )
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("optimized_zh", "optimized_en")
    FUNCTION = "optimize"
    CATEGORY = "哎呀✦MMX/video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "placeholder": "sk-***************************"}),
                "user_prompt": ("STRING", {"multiline": True, "placeholder": "原始视频提示词，如：一只小猫在草地上奔跑"}),
                "target": (["图生视频", "文生视频"], {"default": "图生视频"}),
                "preset": (list(PRESET_PROMPTS.keys()), {"default": "默认（纯分镜）"}),
                "duration": ("INT", {"default": 5, "min": 3, "max": 20, "step": 1, "display": "number"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "custom_system": ("STRING", {"multiline": True, "placeholder": "（可选）自定义系统提示，留空即用上方预设"}),
            }
        }

    def optimize(self, api_key, user_prompt, target, preset, duration=5, image=None, custom_system=""):
        if not api_key.strip():
            raise RuntimeError("❌ api_key 不能为空")

        # 1. 构造 system 提示
        base = custom_system.strip() or PRESET_PROMPTS[preset]
        system = base.format(f"{duration}s")

        # 2. 组装 content
        content = [{"type": "input_text", "text": user_prompt}]
        if target == "图生视频" and image is not None:
            content.append({"type": "input_image", "image_url": image_to_data_url(image)})

        payload = {
            "model": "gpt-5-mini",
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system}]},
                {"role": "user", "content": content}
            ]
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key.strip()}"
        }

        resp = requests.post("https://www.dmxapi.cn/v1/responses",
                             headers=headers, json=payload, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(f"Responses 接口异常 HTTP {resp.status_code}: {resp.text[:200]}")

        try:
            out_list = resp.json()["output"]
            text_block = next(x for x in out_list if x["type"] == "message")["content"]
            full_text = next(x for x in text_block if x["type"] == "output_text")["text"]

            # 统一按行切：第一段中文，第二段英文
            lines = [ln.strip() for ln in full_text.strip().splitlines() if ln.strip()]
            if len(lines) < 2:
                raise ValueError("模型返回不足两行")
            zh, en = lines[0], lines[1]
        except Exception as e:
            raise RuntimeError(f"解析结果失败：{e}")

        return (zh, en)


register_node(VideoPromptOptimizer_DMX, "文图生视频提示词_DMX")