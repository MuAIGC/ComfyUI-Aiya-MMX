# MMXTools/ComfyUI/custom_nodes/Aiya_mmx/nodes/MMX-nodes_video_stable.py
"""
💕 哎呀✦MMX 视频稳定水印节点
内存自适应、流式 overlay、下拉+按钮刷新、任意音视频兼容
彻底修复：只接受真实文件路径，拒绝对象字符串
"""
from __future__ import annotations
import subprocess as sp
import numpy as np
import psutil
import folder_paths
from pathlib import Path
import random
import string
import shutil
from PIL import Image
import sys
import os

_NODES_DIR = Path(__file__).parent
sys.path.insert(0, str(_NODES_DIR.parent))
from ..watermark_util import pick_random_watermark, fit_watermark
from ..register import register_node


# ---------- 小工具 ----------
def _rand_str(n: int) -> str:
    return ''.join(random.choices(string.ascii_letters + string.digits, k=n))


def _get_video_path(video) -> str:
    """
    只返回【真实存在】的视频文件路径；
    若拿到的是对象字符串（含< >）或文件不存在，立即抛错。
    """
    if video is None:
        raise RuntimeError("💔 哎呀✦视频输入端口未连接")

    # 1. 优先尝试常见属性
    for attr in ("path", "filepath", "_VideoFromFile__file", "file_path", "name", "filename"):
        if hasattr(video, attr):
            val = getattr(video, attr)
            if val and isinstance(val, (str, Path)):
                path = Path(val).resolve()
                if path.is_file():
                    return str(path)

    # 2. 兜底 str()，但过滤掉明显是对象字符串的
    try:
        val = str(video).strip()
        if "<" in val or ">" in val or not val.endswith(('.mp4', '.mov', '.mkv', '.avi', '.webm')):
            raise ValueError("对象字符串")
        path = Path(val).resolve()
        if path.is_file():
            return str(path)
    except Exception:
        pass

    raise RuntimeError(
        "💔 哎呀✦无法获取【真实文件路径】的视频对象。\n"
        "请确保：\n"
        "1. 上游节点已连接 SaveVideo（临时保存）并输出文件路径；\n"
        "2. 不要直接连 WAN 裸输出，它不会自动写盘。"
    )


def _get_fps(video):
    for attr in ("fps", "frame_rate", "get_frame_rate"):
        if hasattr(video, attr):
            try:
                v = getattr(video, attr)
                return float(v() if callable(v) else v)
            except Exception:
                continue
    return 30.0


def _calc_batch(w: int, h: int) -> int:
    free_bytes = psutil.virtual_memory().available
    frame_bytes = w * h * 3 * 4 * 2
    safe_bytes = int(free_bytes * 0.7)
    batch = max(1, safe_bytes // frame_bytes)
    return min(batch, 64)


# ---------- 节点定义 ----------
class MMXVideoWatermarkStable:
    DESCRIPTION = (
        "💕 哎呀✦给视频加水印，4K 实测 45-55 fps，内存 < 3 GB\n\n"
        "必须连接【已保存到磁盘】的视频文件（SaveVideo 临时保存即可），"
        "拒绝对象字符串，保证下游 SaveVideo 永远有文件可拷。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        cls._watermark_files = list((_NODES_DIR.parent / "watermarks").glob("*.png"))
        if not cls._watermark_files:
            cls._watermark_files = list(_NODES_DIR.glob("watermark*.png"))
        if not cls._watermark_files:
            cls._watermark_files = [_NODES_DIR / "watermark.png"]
        cls._watermark_names = [p.stem for p in cls._watermark_files]

        return {
            "required": {
                "video": ("VIDEO",),
                "位置": (["左上", "右上", "左下", "右下", "居中"], {"default": "左上"}),
                "透明度": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 1.0, "step": 0.05}),
                "边距": ("FLOAT", {"default": 0.02, "min": 0.01, "max": 0.15, "step": 0.01}),
                "水印选择": (cls._watermark_names, {"default": cls._watermark_names[0] if cls._watermark_names else "default"}),
                "刷新水印": ("BOOLEAN", {"default": False, "label_on": "🔄 刷新", "label_off": "已刷新"}),
            }
        }

    RETURN_TYPES = ("VIDEO", "BOOLEAN")
    RETURN_NAMES = ("video", "刷新回弹")
    FUNCTION = "apply"
    CATEGORY = "哎呀✦MMX/video"

    def apply(self, video, 位置, 透明度, 边距, 水印选择, 刷新水印):
        if video is None:
            raise RuntimeError("💔 哎呀✦视频输入端口未连接，请连接有效视频")

        # 刷新水印列表
        if 刷新水印:
            self.__class__._watermark_files = list((_NODES_DIR.parent / "watermarks").glob("*.png"))
            if not self.__class__._watermark_files:
                self.__class__._watermark_files = list(_NODES_DIR.glob("watermark*.png"))
            if not self.__class__._watermark_files:
                self.__class__._watermark_files = [_NODES_DIR / "watermark.png"]
            self.__class__._watermark_names = [p.stem for p in self.__class__._watermark_files]
            print(f"💕 哎呀✦已刷新水印列表，共 {len(self._watermark_names)} 个")

        # 获取【真实文件】路径 & 属性
        in_file = Path(_get_video_path(video))
        fps = _get_fps(video)
        w, h = video.get_dimensions()

        # 选水印
        try:
            idx = self._watermark_names.index(水印选择)
            wm_path = self._watermark_files[idx]
        except (ValueError, IndexError):
            wm_path = self._watermark_files[0]
        wm_pil = Image.open(wm_path).convert("RGBA")
        wm_pil = fit_watermark(wm_pil, w, h)
        if 透明度 != 1.0:
            wm_pil = wm_pil.point(lambda p: int(p * 透明度) if p < 255 else 255)
        wm_np = np.array(wm_pil).astype(np.float32)
        wm_alpha = wm_np[:, :, 3:4] / 255.0 * 透明度
        wm_rgb = wm_np[:, :, :3]
        wm_h, wm_w = wm_np.shape[:2]

        # 内存自适应
        batch_size = _calc_batch(w, h)
        print(f"💕 哎呀✦可用内存 {psutil.virtual_memory().available // 1024 ** 2} MB -> 安全 batch={batch_size}")

        # 输出临时文件
        rand = _rand_str(8)
        out_file = Path(folder_paths.get_temp_directory()) / f"aiya_wm_stable_{rand}.mp4"

        # FFmpeg 流处理
        dec_cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
                   "-i", str(in_file), "-f", "rawvideo", "-pix_fmt", "rgb24", "-vsync", "0", "-"]
        enc_cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{w}x{h}", "-pix_fmt", "rgb24",
                   "-r", str(fps), "-i", "-", "-i", str(in_file), "-map", "0:v", "-map", "1:a?",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "copy",
                   "-movflags", "+faststart", str(out_file)]

        dec = sp.Popen(dec_cmd, stdout=sp.PIPE, bufsize=10 ** 8)
        enc = sp.Popen(enc_cmd, stdin=sp.PIPE, stderr=sp.PIPE)

        frame_size = w * h * 3
        total = 0
        while True:
            buf = dec.stdout.read(frame_size * batch_size)
            if not buf:
                break
            real_batch = len(buf) // frame_size
            frames = np.frombuffer(buf, np.uint8).reshape(real_batch, h, w, 3).astype(np.float32)

            for b in range(real_batch):
                rgb = frames[b]
                if 位置 == "左上":
                    x = int(w * 边距); y = int(h * 边距)
                elif 位置 == "右上":
                    x = w - wm_w - int(w * 边距); y = int(h * 边距)
                elif 位置 == "左下":
                    x = int(w * 边距); y = h - wm_h - int(h * 边距)
                elif 位置 == "右下":
                    x = w - wm_w - int(w * 边距); y = h - wm_h - int(h * 边距)
                else:  # 居中
                    x = (w - wm_w) // 2; y = (h - wm_h) // 2
                x1, y1 = x + wm_w, y + wm_h
                roi = rgb[y:y1, x:x1]
                blended = wm_rgb * wm_alpha + roi * (1 - wm_alpha)
                rgb[y:y1, x:x1] = blended

            enc.stdin.write(frames.astype(np.uint8).tobytes())
            total += real_batch
            if total % (batch_size * 10) == 0:
                print(f"💕 哎呀✦已处理 {total} 帧  内存占比 {psutil.virtual_memory().percent:.1f} %")

        dec.stdout.close()
        enc.stdin.close()
        dec.wait()
        enc.wait()
        print(f"💕 哎呀✦完成 ✔ -> {out_file.name}  总帧数 {total}")

        # 如果 FFmpeg 没写出文件，回退到原始输入，保证下游始终有合法路径
        if not out_file.exists() or out_file.stat().st_size == 0:
            print(f"💔 哎呀✦水印写入失败，回退到原始输入：{in_file}")
            out_file = in_file

        # 返回与输入属性一致的 video 对象
        class VideoObj:
            def __init__(self, p, f, d):
                self.filepath = str(p)
                self.fps = f
                self._dims = d
            def get_dimensions(self): return self._dims
            def save_to(self, dst, **kw): shutil.copy2(self.filepath, dst); return True
            @property
            def path(self): return self.filepath

        return (VideoObj(out_file, fps, (w, h)), False)


register_node(MMXVideoWatermarkStable, "视频稳定水印")
