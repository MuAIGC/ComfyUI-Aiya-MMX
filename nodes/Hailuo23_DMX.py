# Hailuo23_DMX.py
# 💕 哎呀✦MiniMax-Hailuo-2.3 全家桶（文生 / 图生 / 首尾帧）
from __future__ import annotations
import os
import time
import json
import base64
import io
import uuid
import cv2
import requests
from pathlib import Path
from datetime import datetime
from PIL import Image

import folder_paths
from ..register import register_node
from ..video_adapter import Video
from .MMX_nodes_image_save_jpg import ImageSaveJPG as _save_jpg

# --------------------------------------------------
# 通用常量
# --------------------------------------------------
BASE_URL = "https://www.dmxapi.cn"
POLL_INT = 2
MAX_POLL = 200


# --------------------------------------------------
# 通用工具
# --------------------------------------------------
def _download_file(url: str, dst: Path, max_retry: int = 3, timeout: int = 120):
    """带重试的下载"""
    for attempt in range(1, max_retry + 1):
        try:
            print(f"[Download] 第 {attempt}/{max_retry} 次：{url}")
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(dst, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            print(f"[Download] 成功 → {dst}")
            return
        except Exception as e:
            print(f"[Download] 第 {attempt} 次失败：{e}")
            if attempt == max_retry:
                raise RuntimeError(f"下载失败（重试 {max_retry} 次）：{e}")
            time.sleep(2)


def image_to_base64(path: str) -> str:
    """图片→base64，自动压缩到 <20 MB，边长≤7680"""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"指定图片不存在：{path}")
    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size
        if w * h > 7680 * 7680:
            img.thumbnail((7680, 7680), Image.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        if buffer.tell() > 19 * 1024 * 1024:
            buffer.seek(0)
            buffer.truncate()
            img.save(buffer, format="JPEG", quality=75)
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode()
        return f"data:image/jpeg;base64,{b64}"


def build_video_obj(video_path: Path) -> Video:
    """把本地 mp4 封装成 ComfyUI VIDEO 对象"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return Video(str(video_path), fps, w, h)


# --------------------------------------------------
# 1. 文生视频
# --------------------------------------------------
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

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "download_url")
    FUNCTION = "generate"
    CATEGORY = "哎呀✦MMX/video"

    CAMERA_SHOT_OPTIONS = [
        "无 / 我自己写",
        "[固定]", "[推进]", "[拉远]",
        "[左移]", "[右移]", "[左摇]", "[右摇]",
        "[上升]", "[下降]", "[上摇]", "[下摇]",
        "[变焦推近]", "[变焦拉远]", "[晃动]", "[跟随]",
        "[左摇,上升]", "[推进,右摇]", "[拉远,下降]",
    ]

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
                "camera_shot": (cls.CAMERA_SHOT_OPTIONS, {"default": "无 / 我自己写"}),
            }
        }

    def generate(self, api_key, prompt, duration, resolution, seed,
                 自动优化提示词, 快速预处理, 水印, camera_shot):
        if not api_key.strip() or not prompt.strip():
            raise RuntimeError("❌ API-Key 或 Prompt 为空")

        if camera_shot != "无 / 我自己写":
            prompt = f"{camera_shot} {prompt}"

        token = api_key.strip()
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

        # 提交
        submit_url = f"{BASE_URL}/v1/video_generation"
        resp = requests.post(submit_url, json=payload,
                             headers={"Content-Type": "application/json",
                                      "Authorization": f"Bearer {token}"},
                             timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"提交失败 HTTP {resp.status_code}: {resp.text[:200]}")
        task_id = resp.json()["task_id"]
        print(f"[Hailuo-2.3-T2V] task_id = {task_id}")

        # 轮询
        query_url = f"{BASE_URL}/v1/query/video_generation"
        for cnt in range(1, MAX_POLL + 1):
            time.sleep(POLL_INT)
            q_resp = requests.get(query_url, params={"task_id": task_id},
                                  headers={"Authorization": f"Bearer {token}"}, timeout=30)
            if q_resp.status_code != 200:
                print(f"[Hailuo-2.3-T2V] 查询异常 HTTP {q_resp.status_code}，继续重试…")
                continue
            raw = q_resp.json()
            status = raw.get("status") or raw.get("state") or "unknown"
            file_id = raw.get("file_id")
            if status.lower() == "processing":
                print(f"[Hailuo-2.3-T2V] 处理中… {cnt}/{MAX_POLL}")
                continue
            if status.lower() == "success" and file_id:
                break
            if status.lower() == "failed":
                raise RuntimeError(f"任务失败: {raw}")
        else:
            raise RuntimeError("⏰ 轮询超时")

        # 下载
        retrieve_url = f"{BASE_URL}/v1/files/retrieve"
        dl_resp = requests.get(retrieve_url,
                               params={"file_id": file_id, "task_id": task_id},
                               headers={"Authorization": f"Bearer {token}"}, timeout=30)
        if dl_resp.status_code != 200:
            raise RuntimeError(f"获取下载链接失败 HTTP {dl_resp.status_code}")
        download_url = dl_resp.json()["file"]["download_url"]

        temp_dir = Path(folder_paths.get_temp_directory())
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file = temp_dir / f"hailuo23_t2v_{int(time.time()*1000)}.mp4"
        _download_file(download_url, temp_file)

        video = build_video_obj(temp_file)
        print(f"[Hailuo-2.3-T2V] VIDEO 对象已生成：{video}")
        return (video, download_url)


# --------------------------------------------------
# 2. 图生视频
# --------------------------------------------------
CAMERA_MOVES = [
    "无", "[左移]", "[右移]", "[左摇]", "[右摇]",
    "[推进]", "[拉远]", "[上升]", "[下降]",
    "[上摇]", "[下摇]", "[变焦推近]", "[变焦拉远]", "[晃动]", "[跟随]", "[固定]"
]

SHOT_TEMPLATE = {
    "无": "",
    "人物特写": "a close-up shot of a person, ",
    "半身中景": "a medium shot of upper body, ",
    "全身远景": "a full-body long shot, ",
    "推镜特写": "a smooth push-in close-up shot, ",
    "拉镜远景": "a smooth pull-out long shot, ",
    "左移跟随": "camera pans left following subject, ",
    "右移跟随": "camera pans right following subject, ",
    "上升俯视": "camera rises to overhead view, ",
    "下降仰视": "camera descends to low-angle view, "
}


class Hailuo23Image2Video:
    DESCRIPTION = (
        "💕 哎呀✦MiniMax-Hailuo-2.3 图生视频（官方 15 种运镜 + 镜头模板）\n\n"
        "【必填】\n"
        "  api_key   : 平台分配的 sk-********************************\n"
        "  image     : 喂入的 ComfyUI IMAGE（自动转 JPG）\n"
        "  prompt    : 主体描述，支持 2000 字符，可混自然语言\n\n"
        "【选单】\n"
        "  shot_template : 常用镜头模板\n"
        "  camera_move   : 官方 15 种运镜指令\n"
        "  duration      : 6 s 或 10 s（1080P 只能选 6 s）\n"
        "  resolution    : 768P（默认）或 1080P（仅 6 s）\n"
        "  seed          : -1 为随机，≥0 固定种子\n\n"
        "【返回】\n"
        "  video        : ComfyUI VIDEO 对象，可直接接 VHS 预览/保存\n"
        "  download_url : 原始 mp4 公网直链，有效期 24 h"
    )
    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "download_url")
    FUNCTION = "generate"
    CATEGORY = "哎呀✦MMX/video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "image": ("IMAGE",),
                "shot_template": (list(SHOT_TEMPLATE.keys()), {"default": "无"}),
                "camera_move": (CAMERA_MOVES, {"default": "无"}),
                "prompt": ("STRING", {"default": "", "multiline": True,
                                      "placeholder": "在此写主体描述，如：一只白色小猫"}),
                "duration": (["6", "10"], {"default": "6"}),
                "resolution": (["768P", "1080P"], {"default": "768P"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            }
        }

    def generate(self, api_key, image, shot_template, camera_move, prompt, duration, resolution, seed):
        if not api_key.strip():
            raise RuntimeError("❌ API-Key 为空")
        token = api_key.strip()

        # 保存临时 JPG
        saver = _save_jpg()
        ret = saver.save_images(
            images=image,
            filename_prefix="temp/hailuo23_i2v",
            quality=95,
            optimize=True,
            progressive=False,
            save_prompt_as_txt=False,
            prompt=None,
            extra_pnginfo=None
        )
        jpg_path = ret["result"][1]

        # 拼 prompt
        shot = SHOT_TEMPLATE.get(shot_template, "")
        move = camera_move if camera_move != "无" else ""
        final_prompt = f"{shot}{move}{prompt.strip()}".strip()

        # 提交
        first_frame_image = image_to_base64(jpg_path)
        payload = {
            "model": "MiniMax-Hailuo-2.3",
            "prompt": final_prompt,
            "first_frame_image": first_frame_image,
            "duration": int(duration),
            "resolution": resolution,
            "prompt_optimizer": True,
            "fast_pretreatment": False,
            "aigc_watermark": False,
        }
        if seed != -1:
            payload["seed"] = int(seed)

        submit_url = f"{BASE_URL}/v1/video_generation"
        resp = requests.post(submit_url, json=payload,
                             headers={"Content-Type": "application/json",
                                      "Authorization": f"Bearer {token}"},
                             timeout=90)
        if resp.status_code != 200:
            raise RuntimeError(f"提交失败 HTTP {resp.status_code}: {resp.text[:300]}")
        task_id = resp.json()["task_id"]

        # 轮询
        query_url = f"{BASE_URL}/v1/query/video_generation"
        start_t = time.time()
        for cnt in range(1, MAX_POLL + 1):
            time.sleep(POLL_INT)
            q = requests.get(query_url, params={"task_id": task_id},
                             headers={"Authorization": f"Bearer {token}"}, timeout=30)
            if q.status_code != 200:
                print(f"[Hailuo-2.3-I2V] 查询异常 HTTP {q.status_code}，重试…")
                continue
            raw = q.json()
            status = raw.get("status") or raw.get("state") or "unknown"
            file_id = raw.get("file_id")
            if status.lower() == "processing":
                used = time.time() - start_t
                remain = (MAX_POLL - cnt) * POLL_INT
                print(f"\r[Hailuo-2.3-I2V] 处理中… {cnt}/{MAX_POLL} "
                      f"已用 {used:.1f}s 预估剩余 {remain:.1f}s", end="")
                continue
            if status.lower() == "success" and file_id:
                print("\r[Hailuo-2.3-I2V] 任务完成！           ")
                break
            if status.lower() == "failed":
                raise RuntimeError(f"任务失败: {raw}")
        else:
            raise RuntimeError("⏰ 轮询超时")

        # 下载
        retrieve_url = f"{BASE_URL}/v1/files/retrieve"
        dl_resp = requests.get(retrieve_url,
                               params={"file_id": file_id, "task_id": task_id},
                               headers={"Authorization": f"Bearer {token}"}, timeout=30)
        if dl_resp.status_code != 200:
            raise RuntimeError(f"获取下载链接失败 HTTP {dl_resp.status_code}")
        download_url = dl_resp.json()["file"]["download_url"]

        output_dir = Path(folder_paths.get_output_directory())
        output_dir.mkdir(exist_ok=True)
        video_path = output_dir / f"hailuo23_i2v_{uuid.uuid4().hex[:8]}.mp4"
        _download_file(download_url, video_path)

        video = build_video_obj(video_path)
        print(f"[Hailuo-2.3-I2V] VIDEO 对象已生成：{video}")
        return (video, download_url)


# --------------------------------------------------
# 3. 首尾帧生视频
# --------------------------------------------------
class Hailuo23FirstLast2Video:
    DESCRIPTION = (
        "💕 哎呀✦MiniMax-Hailuo-02 首尾帧生视频（官方 15 种运镜 + 镜头模板）\n\n"
        "【必填】\n"
        "  api_key   : 平台 sk-********************************\n"
        "  prompt    : 视频描述，最大 2000 字符，支持 [运镜] 语法\n"
        "  first_image : 首帧（决定输出分辨率）\n"
        "  last_image  : 尾帧（尺寸不一致时自动裁剪）\n\n"
        "【选单】\n"
        "  shot_template : 镜头模板\n"
        "  camera_move   : 15 种官方运镜指令\n"
        "  duration      : 6 s 或 10 s（1080P 仅 6 s）\n"
        "  resolution    : 768P（默认）或 1080P\n"
        "  seed          : -1 随机，≥0 固定\n\n"
        "【返回】\n"
        "  video        : ComfyUI VIDEO 对象\n"
        "  download_url : 公网直链，24 h 有效"
    )
    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "download_url")
    FUNCTION = "generate"
    CATEGORY = "哎呀✦MMX/video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "first_image": ("IMAGE",),
                "last_image": ("IMAGE",),
                "shot_template": (list(SHOT_TEMPLATE.keys()), {"default": "无"}),
                "camera_move": (CAMERA_MOVES, {"default": "无"}),
                "prompt": ("STRING", {"default": "", "multiline": True,
                                      "placeholder": "在此写主体描述，如：A little girl grow up."}),
                "duration": (["6", "10"], {"default": "6"}),
                "resolution": (["768P", "1080P"], {"default": "768P"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            }
        }

    def generate(self, api_key, first_image, last_image, shot_template,
                 camera_move, prompt, duration, resolution, seed):
        if not api_key.strip():
            raise RuntimeError("❌ API-Key 为空")
        token = api_key.strip()

        # 保存首/尾帧
        saver = _save_jpg()
        first_path = saver.save_images(
            images=first_image, filename_prefix="temp/hailuo23_fl2v_first",
            quality=95, optimize=True, progressive=False,
            save_prompt_as_txt=False, prompt=None, extra_pnginfo=None
        )["result"][1]
        last_path = saver.save_images(
            images=last_image, filename_prefix="temp/hailuo23_fl2v_last",
            quality=95, optimize=True, progressive=False,
            save_prompt_as_txt=False, prompt=None, extra_pnginfo=None
        )["result"][1]

        # 拼 prompt
        shot = SHOT_TEMPLATE.get(shot_template, "")
        move = camera_move if camera_move != "无" else ""
        final_prompt = f"{shot}{move}{prompt.strip()}".strip()

        # 提交首尾帧
        first_b64 = image_to_base64(first_path)
        last_b64 = image_to_base64(last_path)
        payload = {
            "model": "MiniMax-Hailuo-02",
            "prompt": final_prompt,
            "first_frame_image": first_b64,
            "last_frame_image": last_b64,
            "duration": int(duration),
            "resolution": resolution,
            "prompt_optimizer": True,
            "aigc_watermark": False,
        }
        if seed != -1:
            payload["seed"] = int(seed)

        submit_url = f"{BASE_URL}/v1/video_generation"
        resp = requests.post(submit_url, json=payload,
                             headers={"Content-Type": "application/json",
                                      "Authorization": f"Bearer {token}"},
                             timeout=90)
        if resp.status_code != 200:
            raise RuntimeError(f"提交失败 HTTP {resp.status_code}: {resp.text[:300]}")
        task_id = resp.json()["task_id"]

        # 轮询
        query_url = f"{BASE_URL}/v1/query/video_generation"
        start_t = time.time()
        for cnt in range(1, MAX_POLL + 1):
            time.sleep(POLL_INT)
            q = requests.get(query_url, params={"task_id": task_id},
                             headers={"Authorization": f"Bearer {token}"}, timeout=30)
            if q.status_code != 200:
                print(f"[Hailuo-02-FL2V] 查询异常 HTTP {q.status_code}，重试…")
                continue
            raw = q.json()
            status = raw.get("status") or raw.get("state") or "unknown"
            file_id = raw.get("file_id")
            if status.lower() == "processing":
                used = time.time() - start_t
                remain = (MAX_POLL - cnt) * POLL_INT
                print(f"\r[Hailuo-02-FL2V] 处理中… {cnt}/{MAX_POLL} "
                      f"已用 {used:.1f}s 预估剩余 {remain:.1f}s", end="")
                continue
            if status.lower() == "success" and file_id:
                print("\r[Hailuo-02-FL2V] 任务完成！           ")
                break
            if status.lower() == "failed":
                raise RuntimeError(f"任务失败: {raw}")
        else:
            raise RuntimeError("⏰ 轮询超时")

        # 下载
        retrieve_url = f"{BASE_URL}/v1/files/retrieve"
        dl_resp = requests.get(retrieve_url,
                               params={"file_id": file_id, "task_id": task_id},
                               headers={"Authorization": f"Bearer {token}"}, timeout=30)
        if dl_resp.status_code != 200:
            raise RuntimeError(f"获取下载链接失败 HTTP {dl_resp.status_code}")
        download_url = dl_resp.json()["file"]["download_url"]

        output_dir = Path(folder_paths.get_output_directory())
        output_dir.mkdir(exist_ok=True)
        video_path = output_dir / f"hailuo23_fl2v_{uuid.uuid4().hex[:8]}.mp4"
        _download_file(download_url, video_path)

        video = build_video_obj(video_path)
        print(f"[Hailuo-02-FL2V] VIDEO 对象已生成：{video}")
        return (video, download_url)


register_node(AiyaHailuo23DMX, "Hailuo23-文生视频-DMX")
register_node(Hailuo23Image2Video, "Hailuo23-图生视频-DMX")
register_node(Hailuo23FirstLast2Video, "Hailuo23-首尾帧生视频-DMX")
