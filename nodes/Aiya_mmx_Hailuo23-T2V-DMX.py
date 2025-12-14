# ---------------------------------------------------------
#  Aiya_mmx_Hailuo-2_3-DMX.py
#  MiniMax-Hailuo-2.3 文生视频 · 同步下载（带重试）· 自写Video容器
#  新增：双输出 VIDEO + download_url
# ---------------------------------------------------------
from __future__ import annotations
import os
import time
import json
from pathlib import Path
import requests
from datetime import datetime
import folder_paths
from ..register import register_node

# ********  最小 VIDEO 容器（自写） ********
from ..video_adapter import Video   # 同目录上层
import cv2                          # 用于抽参数

POLL_INTERVAL = 3
MAX_POLL    = 100


# ---------------  带重试的下载函数 ---------------
def _download_file(url: str, dst: Path, max_retry: int = 3, timeout: int = 120):
    """3 次重试 + 异常隔离，网络偶发 DNS 失败不崩整个流程"""
    for attempt in range(1, max_retry + 1):
        try:
            print(f"[Download] 尝试第 {attempt}/{max_retry} 次：{url}")
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                down = 0
                with open(dst, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            down += len(chunk)
                print(f"[Download] 成功写入 {down} 字节 → {dst}")
                return  # 成功就跳出
        except Exception as e:
            print(f"[Download] 第 {attempt} 次失败：{e}")
            if attempt == max_retry:
                raise RuntimeError(f"下载失败（重试 {max_retry} 次）：{e}")
            time.sleep(2)  # 短暂冷却再试


# ---------------  节点本体 ---------------
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

    RETURN_TYPES = ("VIDEO", "STRING")          # 双输出
    RETURN_NAMES = ("video", "download_url")
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

        payload = {
            "model": "MiniMax-Hailuo-2.3",
            "prompt": prompt.strip(),
            "duration": int(duration),
            "resolution": resolution,
            "prompt_optimizer": 自动优化提示词 == "开启",
            "fast_pretreatment": 快速预处理 == "开启",
            "aigc_watermark": 水印 == "开启",
        }
        if seed != -1:
            payload["seed"] = int(seed)

        # 1. 提交任务
        submit_url = f"{base_url}/v1/video_generation"
        resp = requests.post(submit_url, json=payload,
                             headers={"Content-Type": "application/json",
                                      "Authorization": f"Bearer {token}"},
                             timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"提交失败 HTTP {resp.status_code}: {resp.text[:200]}")
        task_id = resp.json()["task_id"]
        print(f"[Hailuo-2.3] task_id = {task_id}")

        # 2. 轮询
        query_url = f"{base_url}/v1/query/video_generation"
        for cnt in range(1, MAX_POLL + 1):
            time.sleep(POLL_INTERVAL)
            q_resp = requests.get(query_url, params={"task_id": task_id},
                                  headers={"Authorization": f"Bearer {token}"}, timeout=30)
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

        # 3. 拿下载链接
        retrieve_url = f"{base_url}/v1/files/retrieve"
        dl_resp = requests.get(retrieve_url,
                               params={"file_id": file_id, "task_id": task_id},
                               headers={"Authorization": f"Bearer {token}"}, timeout=30)
        if dl_resp.status_code != 200:
            raise RuntimeError(f"获取下载链接失败 HTTP {dl_resp.status_code}")
        download_url = dl_resp.json()["file"]["download_url"]
        print(f"[Hailuo-2.3] 下载链接：{download_url}")

        # 4. 同步下载（带重试）到本地
        temp_dir = Path(folder_paths.get_temp_directory())
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file = temp_dir / f"hailuo23_{int(time.time()*1000)}.mp4"
        _download_file(download_url, temp_file)

        # 5. 用 cv2 抽参数 + 自写 Video 容器返回
        cap = cv2.VideoCapture(str(temp_file))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        video = Video(str(temp_file), fps, w, h)
        print(f"[Hailuo-2.3] VIDEO 对象已生成：{video}")
        # 6. 双输出：VIDEO + 下载链接字符串
        return (video, download_url)


register_node(AiyaHailuo23DMX, "Hailuo-2_3-T2V-DMX")
