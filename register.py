# MMXTools/ComfyUI/custom_nodes/Aiya_mmx/register.py
"""
💕 哎呀✦MMX 节点登记处
"""
from typing import Dict, Type

NODE_CLASS_MAPPINGS: Dict[str, Type] = {}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {}


def register_node(node_class: Type, display_name: str) -> None:
    """
    在功能文件里调用，完成登记
    会自动给 display_name 加上「💕哎呀✦」前缀，避免重名
    """
    class_name = node_class.__name__

    # 自动加前缀，如果已经带了就跳过
    if not display_name.startswith("💕哎呀✦"):
        display_name = f"💕哎呀✦{display_name}"

    # 防重复
    if class_name in NODE_CLASS_MAPPINGS:
        print(f"⚠️  节点 {class_name} 已被注册，跳过")
        return

    NODE_CLASS_MAPPINGS[class_name] = node_class
    NODE_DISPLAY_NAME_MAPPINGS[class_name] = display_name
    print(f"✅ 已注册节点：{display_name} ({class_name})")