# ~/ComfyUI/custom_nodes/Aiya_mmx/nodes/__init__.py
"""
💕 哎呀✦MMX 节点自动装载机
按文件名升序批量导入，避免手动维护
"""
from __future__ import annotations
import glob
import os

# 获取当前目录下所有 .py 文件（排除 __init__.py 自身）
for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))):
    name = os.path.basename(f)[:-3]  # 去掉 .py
    if name == "__init__":
        continue
    # 动态 import，执行模块顶层的 register_node(...)
    __import__(__package__ + "." + name, fromlist=[""])
    print(f"✅ 哎呀✦已装载节点模块：{name}")
