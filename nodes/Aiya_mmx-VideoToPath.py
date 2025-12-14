# MMXTools/ComfyUI/custom_nodes/Aiya_mmx/nodes/Aiya-mmx-VideoToPath.py
"""
💕 哎呀✦视频强制落盘 + 尺寸测量节点
零属性依赖，WAN/任何对象→立即写盘→返回真实路径 + 实测宽高
"""
from __future__ import annotations
import shutil
import time
import tempfile
from pathlib import Path
import folder_paths
import cv2
from ..register import register_node


class AiyaMMXVideoToPath:
    DESCRIPTION = (
        "💕 哎呀✦把【任何视频对象】立即写盘→返回文件路径 + 实测宽高\n"
        "零属性依赖，插 WAN 后面即可继续跑后续节点"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "delete_source": ("BOOLEAN", {"default": False, "label_on": "删除源文件", "label_off": "保留源文件"}),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "INT", "INT")
    RETURN_NAMES = ("video", "file_path", "width", "height")
    FUNCTION = "convert"
    CATEGORY = "哎呀✦MMX/工具"

    def convert(self, video, delete_source=False):
        if video is None:
            raise RuntimeError("💔 哎呀✦视频输入端口未连接")

        # 1. 先生成临时文件路径
        ts = int(time.time() * 1000)
        temp_dir = Path(folder_paths.get_temp_directory())
        temp_file = temp_dir / f"aiya_video_{ts}.mp4"

        # 2. 强制写盘（零依赖，只用 save_to）
        video.save_to(str(temp_file))
        if not temp_file.exists() or temp_file.stat().st_size == 0:
            raise RuntimeError("💔 哎呀✦视频强制落盘失败，上游对象未正确实现 save_to")

        # 3. 用 cv2 实测宽高（不再依赖上游对象任何属性）
        cap = cv2.VideoCapture(str(temp_file))
        if not cap.isOpened():
            raise RuntimeError("💔 哎呀✦无法打开落盘文件，FFmpeg 可能写入失败")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        if width <= 0 or height <= 0:
            raise RuntimeError("💔 哎呀✦实测宽高为 0，上游对象可能未正确输出视频")

        # 4. 封装成标准文件对象返回
        class FileVideo:
            def __init__(self, p, w, h, f):
                self.filepath = str(p)
                self._w = w
                self._h = h
                self._fps = f
            def get_dimensions(self):
                return (self._w, self._h)
            def save_to(self, dst, **kw):
                shutil.copy2(self.filepath, dst)
                return True
            @property
            def path(self):
                return self.filepath
            @property
            def fps(self):
                return self._fps

        # 5. 可选：删除源文件
        if delete_source:
            try:
                src = Path(video.path)
                if src.is_file() and src != temp_file:
                    src.unlink(missing_ok=True)
            except Exception:
                pass

        return (FileVideo(temp_file, width, height, fps), str(temp_file), width, height)


register_node(AiyaMMXVideoToPath, "视频强制落盘+尺寸")
