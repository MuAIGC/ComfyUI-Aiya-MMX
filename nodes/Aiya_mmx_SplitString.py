"""
Aiya_mmx_SplitString.py
💕 哎呀✦字符串分割节点
输入：1个STRING口
输出：5个STRING口，自适应输出（空位补""）
注册：SplitString_mmx
"""
from __future__ import annotations
from ..date_variable import replace_date_vars   # 相对上层目录，稳妥
from ..register import register_node            # 相对上层目录，稳妥


class SplitString_mmx:
    DESCRIPTION = (
        "💕 哎呀✦字符串分割节点（1→5 STRING）\n\n"
        "输入：任意字符串\n"
        "输出：5个STRING口，按换行或自定义分隔符切分，空位补\"\"\n\n"
        "分隔符：留空=换行分割"
    )
    RETURN_TYPES = tuple(["STRING"] * 5)
    RETURN_NAMES = tuple([f"string{i}" for i in range(1, 6)])
    FUNCTION = "split"
    CATEGORY = "哎呀✦MMX/text"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "separator": ("STRING", {"default": "", "multiline": False}),
            }
        }

    def split(self, text: str, separator: str) -> tuple[str, ...]:
        # 替换日期变量
        text = replace_date_vars(text, safe_path=False)
        separator = replace_date_vars(separator, safe_path=False)

        # 分割
        if separator == "":
            parts = text.splitlines()
        else:
            parts = text.split(separator)

        # 只留前 5 段，不足补空
        parts = parts[:5] + [""] * (5 - len(parts))
        result = tuple(p.strip() for p in parts)
        print(f"[SplitString_mmx] 分割完成 → {result}")
        return result


register_node(SplitString_mmx, "SplitString_mmx")