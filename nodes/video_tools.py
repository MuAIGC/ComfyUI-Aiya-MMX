# ~/ComfyUI/custom_nodes/Aiya_mmx/nodes/video_tools.py
from __future__ import annotations
import os
import time
import shutil
import requests
import tempfile
from pathlib import Path
from datetime import datetime
import cv2
import folder_paths
from ..register import register_node
from ..date_variable import replace_date_vars


# --------------------------------------------------
#  1. 通用视频下载  DownloadVideo_mmx
# --------------------------------------------------
class DownloadVideo_mmx:
    DESCRIPTION = (
        "💕 哎呀✦通用视频下载节点（VIDEO 输出）\n\n"
        "输入：http/https 直链（.mp4/.mov/.avi 等）\n"
        "输出：橙色 VIDEO → 下游任意视频节点即插即用\n\n"
        "文件名：支持与你 saveJPG 完全相同的日期变量\n"
        "保存路径：官方 output 目录，自动防重名"
    )
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "download"
    CATEGORY = "哎呀✦MMX/video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "download_url": ("STRING", {"default": "", "multiline": False, "placeholder": "https://example.com/video.mp4"}),
                "filename_prefix": ("STRING", {"default": "Aiya/%Aiya:yyyyMMdd%/download"}),
                "timeout_seconds": ("INT", {"default": 300, "min": 30, "max": 1800, "step": 30}),
            }
        }

    def download(self, download_url: str, filename_prefix: str, timeout_seconds: int):
        if not download_url.strip():
            raise RuntimeError("❌ 下载链接为空")

        url = download_url.strip()
        prefix = replace_date_vars(filename_prefix.strip(), safe_path=True)
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            prefix, folder_paths.get_output_directory(), 1920, 1080)

        fname = f"{filename}_{counter:05}.mp4"
        video_path = Path(full_output_folder) / fname

        print(f"[DownloadVideo_mmx] 开始下载 → {url}")
        try:
            with requests.get(url, stream=True, timeout=timeout_seconds) as r:
                r.raise_for_status()
                with open(video_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        except Exception as e:
            raise RuntimeError(f"下载失败：{e}")

        print(f"[DownloadVideo_mmx] 已保存 → {video_path}")

        # 用 cv2 抽参数
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # 自写轻量容器，对齐 VHS 接口
        video = Video(str(video_path), fps, w, h)
        return (video,)


# --------------------------------------------------
#  2. 视频强制落盘 + 尺寸  VideoToPath_mmx
# --------------------------------------------------
class VideoToPath_mmx:
    DESCRIPTION = (
        "💕 哎呀✦把【任何视频对象】立即写盘→返回文件路径 + 实测宽高\n"
        "零属性依赖，插 WAN 后面即可继续跑后续节点"
    )
    RETURN_TYPES = ("VIDEO", "STRING", "INT", "INT")
    RETURN_NAMES = ("video", "file_path", "width", "height")
    FUNCTION = "convert"
    CATEGORY = "哎呀✦MMX/video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "delete_source": ("BOOLEAN", {"default": False, "label_on": "删除源文件", "label_off": "保留源文件"}),
            }
        }

    def convert(self, video, delete_source=False):
        if video is None:
            raise RuntimeError("💔 哎呀✦视频输入端口未连接")

        ts = int(time.time() * 1000)
        temp_dir = Path(folder_paths.get_temp_directory())
        temp_file = temp_dir / f"aiya_video_{ts}.mp4"

        # 强制写盘
        video.save_to(str(temp_file))
        if not temp_file.exists() or temp_file.stat().st_size == 0:
            raise RuntimeError("💔 哎呀✦视频强制落盘失败")

        # 实测宽高
        cap = cv2.VideoCapture(str(temp_file))
        if not cap.isOpened():
            raise RuntimeError("💔 哎呀✦无法打开落盘文件")
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        if width <= 0 or height <= 0:
            raise RuntimeError("💔 哎呀✦实测宽高为 0")

        # 轻量容器返回
        class FileVideo:
            def __init__(self, p, w, h, f):
                self.filepath = str(p)
                self._w, self._h, self._fps = w, h, f
            def get_dimensions(self): return (self._w, self._h)
            def save_to(self, dst, **kw):
                shutil.copy2(self.filepath, dst); return True
            @property
            def path(self): return self.filepath
            @property
            def fps(self): return self._fps

        if delete_source:
            try:
                src = Path(video.path)
                if src.is_file() and src != temp_file:
                    src.unlink(missing_ok=True)
            except Exception:
                pass

        return (FileVideo(temp_file, width, height, fps), str(temp_file), width, height)


# --------------------------------------------------
#  轻量 VIDEO 容器，对齐 VHS
# --------------------------------------------------
class Video:
    __slots__ = ("path", "fps", "width", "height")
    def __init__(self, path: str, fps: float, width: int, height: int):
        self.path, self.fps, self.width, self.height = path, fps, width, height
    def get_dimensions(self): return (self.width, self.height)
    def save_to(self, dst: str | Path, **kw):
        shutil.copy2(self.path, dst); return True
    def __repr__(self): return f"Video({self.path} {self.fps:.2f}fps {self.width}x{self.height})"

# --------------------------------------------------
#  3. 路径加载视频  LoadVideoFromPath_mmx
# --------------------------------------------------
CACHE_DIR_V = Path(folder_paths.get_output_directory()) / "Aiya/Aiya_path"


class LoadVideoFromPath_mmx:
    DESCRIPTION = "💕 哎呀✦从路径加载视频，空输入自动读缓存，逻辑同 img/txt"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "", "multiline": False}),
                "cache_name": ("STRING", {"default": "default", "multiline": False}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "load"
    CATEGORY = "哎呀✦MMX/video"

    def load(self, path, cache_name):
        path = path.strip()
        cache_name = cache_name.strip() or "default"
        cache_file = CACHE_DIR_V / f"{cache_name}.videopath"

        # 空输入 → 读缓存
        if not path:
            if cache_file.exists():
                path = cache_file.read_text(encoding="utf-8").strip()
            if not path:
                raise RuntimeError(f"LoadVideoFromPath_mmx: 缓存「{cache_name}」为空！")
        # 非空输入 → 写缓存
        else:
            path = replace_date_vars(path)
            CACHE_DIR_V.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(path, encoding="utf-8")

        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"LoadVideoFromPath_mmx: 文件不存在 → {path}")

        # 用 cv2 抽参数
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"LoadVideoFromPath_mmx: 无法打开视频 → {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # 轻量容器返回
        return (Video(str(path), fps, w, h),)


# --------------------------------------------------
#  统一注册
# --------------------------------------------------
register_node(DownloadVideo_mmx, "DownloadVideo_mmx")
register_node(VideoToPath_mmx,  "VideoToPath_mmx")
register_node(LoadVideoFromPath_mmx, "LoadVideoFromPath_mmx")
