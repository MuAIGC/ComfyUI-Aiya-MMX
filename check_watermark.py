# ~/ComfyUI/custom_nodes/Aiya_mmx/check_watermark.py
"""
💕 哎呀✦MMX 系列小工具：水印自检
"""
from __future__ import annotations
from pathlib import Path

# 路径常量
util_file = Path(__file__).parent / "nodes" / "watermark_util.py"
watermark_dir = util_file.parent.parent / "watermarks"

# 少女风自检日志
print("🌸 哎呀✦插件根目录 :", util_file.parent.parent)
print("🌸 哎呀✦水印目录   :", watermark_dir)
print("🌸 目录存在?        :", watermark_dir.is_dir())

png_list = list(watermark_dir.glob("*.[pP][nN][g]"))
print("🌸 下含 png?        :", png_list)

# 贴心小棉袄：目录不存在就自动建好
if not watermark_dir.is_dir():
    watermark_dir.mkdir(parents=True, exist_ok=True)
    print("✨ 哎呀✦已自动帮你建好水印文件夹啦~")

__all__ = ["watermark_dir", "png_list"]