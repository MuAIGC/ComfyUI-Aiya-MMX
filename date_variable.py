# ~/ComfyUI/custom_nodes/Aiya_mmx/date_variable.py
"""
Aiya-MMX 时间变量小魔法
同时支持：%Aiya:yyyyMMdd%  和  %date:yyyyMMdd% （±N 天偏移）
"""
from __future__ import annotations
import datetime
import re
from typing import Dict

# 用户变量 → Python strftime 格式（大小写敏感）
VAR_MAP: Dict[str, str] = {
    "yyyy": "%Y", "MM": "%m", "dd": "%d",
    "HH": "%H", "hh": "%I", "mm": "%M", "ss": "%S",
    "yyyymmdd": "%Y%m%d", "yyyy-mm-dd": "%Y-%m-%d",
    "yyyyMMdd": "%Y%m%d",
    "yyyy-MM-dd-HH-mm-ss": "%Y-%m-%d-%H-%M-%S",
    "yyyyMMddHHmmss": "%Y%m%d%H%M%S",
    "HHmmss": "%H%M%S", "mmss": "%M%S",
    "yyyy/MM/dd": "%Y/%m/%d", "yyyy\\MM\\dd": r"%Y\%m\%d",
    "yyyy-MM-dd_HH-mm-ss": "%Y-%m-%d_%H-%M-%S",
    "yyyy_MM_dd_HH_mm_ss": "%Y_%m_%d_%H_%M_%S",
    "HH-mm-ss": "%H-%M-%S", "HH_mm_ss": "%H_%M_%S",
    "mm-ss": "%M-%S", "mm_ss": "%M_%S",
}

# 同时识别 %Aiya:fmt% 和 %date:fmt%
VAR_PATTERN = re.compile(
    r"(?:%Aiya:|%date:)("
    + "|".join(map(re.escape, VAR_MAP.keys()))
    + r")(?:([+-]\d+))?%"
)


def _safe_filename(text: str) -> str:
    """把路径不友好字符替换成安全符号"""
    return re.sub(r'[<>:"|?* ]', "_", text)


def replace_date_vars(text: str, safe_path: bool = True) -> str:
    """
    把 %Aiya:fmt% 或 %date:fmt% （±offset）替换成真实时间
    safe_path=True 时自动清理路径非法字符
    """
    now = datetime.datetime.now()
    while True:
        m = VAR_PATTERN.search(text)
        if not m:
            break
        user_fmt = m.group(1)
        offset = int(m.group(2) or 0)
        py_fmt = VAR_MAP[user_fmt]
        target_day = now + datetime.timedelta(days=offset)
        replacement = target_day.strftime(py_fmt)
        if safe_path:
            replacement = _safe_filename(replacement)
        text = text[:m.start()] + replacement + text[m.end():]
    return text


# 小自检
if __name__ == "__main__":
    demo = "旧_%date:yyyyMMdd%_新_%Aiya:yyyyMMdd-1%.png"
    print("Aiya-MMX DEMO 输入 :", demo)
    print("Aiya-MMX DEMO 输出 :", replace_date_vars(demo))