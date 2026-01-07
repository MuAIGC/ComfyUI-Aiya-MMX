# ~/ComfyUI/custom_nodes/ComfyUI-Aiya-MMX/nodes/color_picker.py
from __future__ import annotations
from ..register import register_node

# 预置颜色表：name -> HEX 或 (start, end)
PALETTE: dict[str, str | tuple[str, str]] = {
    "纯白": "#FFFFFF",
    "标准蓝": "#3C7BFF",
    "鲜红": "#FF0000",
    "浅蓝": "#94C4FF",
    "淡青": "#E0F7FF",
    "薄荷": "#D6F5D6",
    "淡粉": "#FFE6F0",
    "暖黄": "#FFF8E1",
    "薰衣草": "#E8E3FF",
    "蛋壳": "#FDF6E3",
    "冰灰": "#F2F5F7",
    "云朵": "#FAFAFA",
    "雾银": "#EBEFF2",
    "柔紫": "#F2E6FF",
    "奶茶": "#F8F0E5",
    "抹茶": "#E8F5E9",
    "天空": "#E3F2FD",
    "蜜桃": "#FFF0F5",
    "牛仔": "#5B9BFF",
    "湖水": "#4FC3F7",
    "薄荷绿": "#7CFFBF",
    "樱花": "#FFB7C5",
    "柠檬": "#FFFACD",
    "奶油": "#FFFDD0",
    "藕荷": "#D9C2D9",
    "藕粉": "#F5E6DE",
    "高级灰": "#B8BCC8",
    "石墨": "#708090",
    "渐变灰": ("#EBEBEB", "#C8C8C8"),
    "渐变米": ("#FFF8DC", "#FFE4B5"),
    "渐变蓝": ("#0070C0", "#6BB3FF"),
    "渐变薰衣草": ("#E8E3FF", "#C5B8FF"),
    "渐变薄荷": ("#D6F5D6", "#A8E6A8"),
    "渐变蜜桃": ("#FFF0F5", "#FFC5D9"),
    "渐变牛仔": ("#5B9BFF", "#8AB6FF"),
    "渐变柠檬": ("#FFFACD", "#FFF176"),
    "渐变藕荷": ("#D9C2D9", "#C0A0C0"),
    "渐变暖黄": ("#FFF8E1", "#FFECB3"),
    "渐变冰蓝": ("#E0F7FF", "#B3E5FC"),
    "渐变抹茶": ("#E8F5E9", "#C8E6C9"),
    "渐变天空": ("#E3F2FD", "#BBDEFB"),
    "渐变湖水": ("#4FC3F7", "#81D4FA"),
    "渐变高级灰": ("#B8BCC8", "#9AA0B8"),
    "渐变樱花": ("#FFB7C5", "#FF8FA3"),
    "渐变雾银": ("#EBEFF2", "#DDE2E6"),
    "渐变奶油": ("#FFFDD0", "#FFF8B8"),
}


class ColorPicker_mmx:
    DESCRIPTION = (
        "💕 哎呀✦颜色选择器（下拉+自定义）\n\n"
        "下拉：60+ 预置纯色/渐变 HEX\n"
        "自定义：任意 HEX/RGB 字符串\n\n"
        "输出：纯色→“颜色名#HEX”\n"
        "      渐变→“颜色名（#HEX向#HEX渐变）”"
    )
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("color_text",)
    FUNCTION = "pick"
    CATEGORY = "哎呀✦MMX/color"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (["自定义"] + list(PALETTE.keys()), {"default": "纯白"}),
                "custom_hex": ("STRING", {"default": "", "multiline": False}),
            }
        }

    def pick(self, preset: str, custom_hex: str) -> tuple[str, ...]:
        # 自定义优先
        if custom_hex.strip():
            out = custom_hex.strip().upper()
            # 简单合法性检查
            if not (out.startswith("#") and len(out) == 7):
                print(f"[ColorPicker_mmx] 警告：'{out}' 非标准 HEX，已回退 #FFFFFF")
                out = "#FFFFFF"
            result = f"自定义{out}"
        else:
            color_def = PALETTE.get(preset, "#FFFFFF")
            if isinstance(color_def, tuple):
                start, end = color_def
                result = f"{preset}（{start}向{end}渐变）"
            else:
                result = f"{preset}{color_def}"

        print(f"[ColorPicker_mmx] 输出 → {result}")
        return (result,)


# 注册节点
register_node(ColorPicker_mmx, "ColorPicker_mmx")