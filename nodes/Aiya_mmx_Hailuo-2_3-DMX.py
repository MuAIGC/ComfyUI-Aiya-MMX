"""
💕 哎呀✦MMX  MiniMax-Hailuo-2.3 视频生成节点
仅返回下载链接字符串，不下载、不封装
可选参数 + 中文说明 + 运镜指令完整提示
文件：Aiya_mmx_Hailuo-2_3-DMX.py
"""
from __future__ import annotations
import os
import time
import json
import requests
from pathlib import Path
from datetime import datetime
import folder_paths
from ..register import register_node

POLL_INTERVAL = 3
MAX_POLL    = 100

class AiyaHailuo23DMX:
    DESCRIPTION = (
        "💕 哎呀✦MiniMax-Hailuo-2.3 文生视频\n\n"
        "【可选参数】\n"
        "• 自动优化提示词：默认开启，可关闭\n"
        "• 快速预处理：默认关闭，可开启（缩短优化耗时）\n"
        "• 水印：默认关闭，可开启\n\n"
        "【运镜指令语法】\n"
        "在 prompt 中用 [指令] 格式插入，支持 15 种：\n"
        "左右移 [左移] [右移]  |  左右摇 [左摇] [右摇]\n"
        "推拉 [推进] [拉远]  |  升降 [上升] [下降]\n"
        "上下摇 [上摇] [下摇]  |  变焦 [变焦推近] [变焦拉远]\n"
        "其他 [晃动] [跟随] [固定]\n\n"
        "使用规则：\n"
        "1. 组合运镜：同一组 [] 内多个指令同时生效，如 [左摇,上升]（≤3 个）\n"
        "2. 顺序运镜：prompt 中前后出现的指令依次生效，如“[推进], 然后 [拉远]”\n"
        "3. 自然语言描述运镜也可，但标准指令更精准\n\n"
        "【尺寸】仅支持 768P / 1080P，其他值会报错"
    )

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("download_url",)
    FUNCTION = "generate"
    CATEGORY = "哎呀✦MMX/video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"forceInput": True}),
                "duration": (["6", "10"], {"default": "6"}),
                "resolution": (["768P", "1080P"], {"default": "768P"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                # 可选参数（中文下拉，与官方默认值一致）
                "自动优化提示词": (["开启", "关闭"], {"default": "开启"}),
                "快速预处理": (["关闭", "开启"], {"default": "关闭"}),
                "水印": (["关闭", "开启"], {"default": "关闭"}),
            }
        }

    def generate(self, api_key, prompt, duration, resolution, seed,
                 自动优化提示词, 快速预处理, 水印):
        if not api_key.strip() or not prompt.strip():
            raise RuntimeError("❌ API-Key 或 Prompt 为空")

        base_url = "https://www.dmxapi.cn"
        token    = api_key.strip()

        # 布尔映射（中文→官方布尔）
        prompt_optimizer = 自动优化提示词 == "开启"
        fast_pretreatment = 快速预处理 == "开启"
        aigc_watermark   = 水印 == "开启"

        # 1. 提交任务
        submit_url = f"{base_url}/v1/video_generation"
        payload = {
            "model": "MiniMax-Hailuo-2.3",
            "prompt": prompt.strip(),
            "duration": int(duration),
            "resolution": resolution,
            "prompt_optimizer": prompt_optimizer,
            "fast_pretreatment": fast_pretreatment,
            "aigc_watermark": aigc_watermark,
        }
        if seed != -1:
            payload["seed"] = int(seed)

        print(f"[Hailuo-2.3] 提交 POST → {submit_url}")
        resp = requests.post(submit_url,
                             json=payload,
                             headers={"Content-Type": "application/json",
                                      "Authorization": f"Bearer {token}"},
                             timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"提交失败 HTTP {resp.status_code}: {resp.text[:200]}")
        task_id = resp.json()["task_id"]
        print(f"[Hailuo-2.3] task_id = {task_id}")

        # 2. 轮询查询
        query_url = f"{base_url}/v1/query/video_generation"
        for cnt in range(1, MAX_POLL + 1):
            time.sleep(POLL_INTERVAL)
            q_resp = requests.get(query_url,
                                  params={"task_id": task_id},
                                  headers={"Authorization": f"Bearer {token}"},
                                  timeout=30)
            if q_resp.status_code != 200:
                print(f"[Hailuo-2.3] 查询异常 HTTP {q_resp.status_code}，继续重试…")
                continue
            raw = q_resp.json()
            status  = raw.get("status") or raw.get("state") or "unknown"
            file_id = raw.get("file_id")

            if status.lower() == "processing":
                print(f"[Hailuo-2.3] 处理中… {cnt}/{MAX_POLL}")
                continue
            if status.lower() == "success" and file_id:
                break
            if status.lower() == "failed":
                raise RuntimeError(f"任务失败: {raw}")
        else:
            raise RuntimeError("⏰ 轮询超时")

        # 3. 只拿下载链接，不下载
        retrieve_url = f"{base_url}/v1/files/retrieve"
        dl_resp = requests.get(retrieve_url,
                               params={"file_id": file_id, "task_id": task_id},
                               headers={"Authorization": f"Bearer {token}"},
                               timeout=30)
        if dl_resp.status_code != 200:
            raise RuntimeError(f"获取下载链接失败 HTTP {dl_resp.status_code}")
        download_url = dl_resp.json()["file"]["download_url"]
        print(f"[Hailuo-2.3] 下载链接已生成：{download_url}")
        return (download_url,)   # 仅返回字符串

register_node(AiyaHailuo23DMX, "Hailuo-2_3-DMX")