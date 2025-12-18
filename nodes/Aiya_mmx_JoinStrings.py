"""
Aiya_mmx_JoinStrings.py
💕 哎呀✦多字符串拼接节点
输入：任意数量的 STRING 口（拉线即增）
输出：橙色 STRING → 下游任意字符串节点即插即用
注册：JoinStrings_mmx
"""
from __future__ import annotations
from ..register import register_node
from ..date_variable import replace_date_vars


class JoinStrings_mmx:          # ← 类名同步
    DESCRIPTION = (
        "💕 哎呀✦多字符串拼接节点（STRING 输出）\n\n"
        "输入：任意数量的 STRING 口（拉线即增）\n"
        "输出：橙色 STRING → 下游任意字符串节点即插即用\n\n"
        "连接符：支持日期变量，可空"
    )
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "join"
    CATEGORY = "哎呀✦MMX/text"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "connector": ("STRING", {"default": " ", "multiline": False}),
            },
            "optional": {
                "string1": ("STRING", {"default": "", "multiline": True}),
                "string2": ("STRING", {"default": "", "multiline": True}),
                "string3": ("STRING", {"default": "", "multiline": True}),
            },
            "hidden": {
                "extra_strings": ("STRING", {"default": ""}),
            }
        }

    def join(self, connector: str, **kwargs) -> tuple[str]:
        connector = replace_date_vars(connector, safe_path=False)
        parts = [v.strip() for k, v in kwargs.items()
                 if k.startswith("string") and isinstance(v, str) and v.strip()]
        result = connector.join(parts)
        print(f"[JoinStrings_mmx] 拼接完成 → {repr(result)}")
        return (result,)


register_node(JoinStrings_mmx, "JoinStrings_mmx")