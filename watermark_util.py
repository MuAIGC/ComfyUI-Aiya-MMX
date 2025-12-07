# MMXTools/ComfyUI/custom_nodes/Aiya_mmx/watermark_util.py
"""
💕 哎呀✦MMX 水印小工具
"""
from __future__ import annotations
import os
import random
import cv2
from pathlib import Path
from PIL import Image
import numpy as np

# 水印素材目录
WATERMARK_DIR = Path(__file__).parent.parent / "watermarks"

# --------------------------------------------------
# 公共函数
# --------------------------------------------------
def list_watermark_files() -> list[str]:
    """返回所有 png 水印绝对路径（大小写不敏感）"""
    if not WATERMARK_DIR.is_dir():
        return []
    return [str(p) for p in WATERMARK_DIR.glob("*.[pP][nN][g]")]


def pick_random_watermark() -> str:
    """随机挑一张水印，找不到就抛异常"""
    files = list_watermark_files()
    if not files:
        raise FileNotFoundError(
            f"🚫 哎呀✦在 {WATERMARK_DIR} 里没找到任何 png 水印文件哦~"
        )
    return random.choice(files)


def fit_watermark(wm_pil: Image.Image, img_w: int, img_h: int) -> Image.Image:
    """
    1. 水印短边 = 图像短边 * 8 %
    2. 整体再缩小 90 %（四周各留 5 % 空白）
    3. 若仍超出画布，再二次缩小到「刚好塞下」
    """
    img_short = min(img_w, img_h)
    target_wm_short = int(img_short * 0.08)
    wm_w, wm_h = wm_pil.size
    scale = target_wm_short / min(wm_w, wm_h)
    new_w, new_h = int(wm_w * scale), int(wm_h * scale)

    # 统一留边：整体 90 % → 四周各 5 % 空白
    new_w, new_h = int(new_w * 0.90), int(new_h * 0.90)
    wm_pil = wm_pil.resize((new_w, new_h), Image.LANCZOS)

    # 二次保护
    scale2 = min(img_w / new_w, img_h / new_h, 1.0)
    if scale2 < 1.0:
        new_w, new_h = int(new_w * scale2), int(new_h * scale2)
        wm_pil = wm_pil.resize((new_w, new_h), Image.LANCZOS)

    return wm_pil


def apply_watermark_np(
    img_np: np.ndarray,
    wm_np: np.ndarray,
    position: str,
    alpha: float,
    margin_ratio: float = 0.02,
) -> np.ndarray:
    """
    将 4 通道水印合成到图像上
    position: 左上 / 左下 / 右上 / 右下 / 居中
    margin_ratio: 离边缘距离 = 图像短边 × ratio，默认 2 %
    """
    assert 0.5 <= alpha <= 1.0, "alpha 必须在 0.5~1.0 之间"
    ih, iw = img_np.shape[:2]
    wm_h, wm_w = wm_np.shape[:2]
    margin = int(min(ih, iw) * margin_ratio)

    # 计算左上角坐标
    if position == "左上":
        x, y = margin, margin
    elif position == "左下":
        x, y = margin, ih - wm_h - margin
    elif position == "右上":
        x, y = iw - wm_w - margin, margin
    elif position == "右下":
        x, y = iw - wm_w - margin, ih - wm_h - margin
    elif position == "居中":
        x, y = (iw - wm_w) // 2, (ih - wm_h) // 2
    else:
        raise ValueError(f"🚫 未知位置: {position}")

    # 越界保护
    x = max(0, min(x, iw - wm_w))
    y = max(0, min(y, ih - wm_h))

    #  alpha 融合
    roi = img_np[y : y + wm_h, x : x + wm_w]
    wm_alpha = wm_np[:, :, 3:4] / 255.0 * alpha
    wm_rgb = wm_np[:, :, :3]
    blended = (roi * (1 - wm_alpha) + wm_rgb * wm_alpha).astype(np.uint8)
    img_np[y : y + wm_h, x : x + wm_w] = blended
    return img_np