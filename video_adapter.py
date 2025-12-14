"""
video_adapter.py
与 VHS-Video 100% 兼容的最小容器 —— 不管对方传多少参数都接得住
"""
from __future__ import annotations
import shutil
import os


class Video:
    __slots__ = ("_path", "_fps", "_width", "_height")

    def __init__(self, path: str, fps: float, width: int, height: int) -> None:
        self._path: str   = path
        self._fps: float  = fps
        self._width: int  = width
        self._height: int = height

    # ----------- 下游必须 -----------
    def get_video_path(self) -> str:
        return self._path

    def get_fps(self) -> float:
        return self._fps

    def get_dimensions(self) -> tuple[int, int]:
        return (self._width, self._height)

    # 终极兼容：吞噬任何关键字，仅做复制
    def save_to(self, full_path: str, *args, **kwargs) -> None:
        """
        不管下游传 format/codec/bit_rate/... 一律忽略，
        只做一件事：把已落盘文件复制到目标路径
        """
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        shutil.copyfile(self._path, full_path)

    # ----------- 调试用 -----------
    def __repr__(self) -> str:
        return f"Video(path={self._path!r}, fps={self._fps}, wh={self._width}x{self._height})"