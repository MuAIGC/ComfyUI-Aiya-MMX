# Aiya_mmx_Hailuo23_I2V-DMX.py
# 💕 哎呀✦MiniMax-Hailuo-2.3 图生视频（内置运镜选单 + 提示词模板）
from __future__ import annotations
import cv2
import time
import requests
import base64
import io
from pathlib import Path
from PIL import Image
import folder_paths
from ..register import register_node
from ..video_adapter import Video
from .MMX_nodes_image_save_jpg import ImageSaveJPG as _save_jpg

BASE_URL = "https://www.dmxapi.cn"
MODEL = "MiniMax-Hailuo-2.3"
POLL_INT = 2
MAX_POLL = 200

# ===== 官方 15 种运镜指令 =====
CAMERA_MOVES = [
    "无",                # 0
    "[左移]", "[右移]",
    "[左摇]", "[右摇]",
    "[推进]", "[拉远]",
    "[上升]", "[下降]",
    "[上摇]", "[下摇]",
    "[变焦推近]", "[变焦拉远]",
    "[晃动]", "[跟随]", "[固定]"
]

# ===== 常用镜头模板 =====
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


def jpg_path_to_base64(path: str) -> str:
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
            buffer.seek(0); buffer.truncate()
            img.save(buffer, format="JPEG", quality=75)
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode()
        return f"data:image/jpeg;base64,{b64}"


def _download_file(url: str, dst: Path, max_retry: int = 3, timeout: int = 120):
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


class Hailuo23Image2Video:
    DESCRIPTION = (
        "💕 哎呀✦MiniMax-Hailuo-2.3 图生视频（内置运镜选单）\n\n"
        "输入：api_key + IMAGE + 镜头模板 + 运镜指令 + 自定义 prompt\n"
        "内部自动拼接官方运镜语法，无需手写 [ ]"
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

        # ① 内部保存 JPG（临时目录）
        save_node = _save_jpg()
        ret = save_node.save_images(
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
        print(f"[Hailuo23-Img2Vid] 临时 JPG 路径：{jpg_path}")

        # ② 自动拼接官方运镜语法
        shot = SHOT_TEMPLATE.get(shot_template, "")
        move = camera_move if camera_move != "无" else ""
        final_prompt = f"{shot}{move}{prompt.strip()}".strip()
        print(f"[Hailuo23-Img2Vid] 最终 prompt：{final_prompt}")

        # ③ Base64 → 提交
        first_frame_image = jpg_path_to_base64(jpg_path)
        submit_url = f"{BASE_URL}/v1/video_generation"
        payload = {
            "model": MODEL,
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

        print(f"[Hailuo23-Img2Vid] 提交任务（Base64 方式）...")
        resp = requests.post(submit_url, json=payload,
                             headers={"Content-Type": "application/json",
                                      "Authorization": f"Bearer {token}"},
                             timeout=90)
        if resp.status_code != 200:
            raise RuntimeError(f"提交失败 HTTP {resp.status_code}: {resp.text[:300]}")
        task_id = resp.json()["task_id"]
        print(f"[Hailuo23-Img2Vid] task_id = {task_id}")

        # ④ 轮询、下载、封装 VIDEO
        query_url = f"{BASE_URL}/v1/query/video_generation"
        start_t = time.time()
        for cnt in range(1, MAX_POLL + 1):
            time.sleep(POLL_INT)
            q = requests.get(query_url, params={"task_id": task_id},
                             headers={"Authorization": f"Bearer {token}"}, timeout=30)
            if q.status_code != 200:
                print(f"[Hailuo23-Img2Vid] 查询异常 HTTP {q.status_code}，重试…")
                continue
            raw = q.json()
            status = raw.get("status") or raw.get("state") or "unknown"
            file_id = raw.get("file_id")
            if status.lower() == "processing":
                used = time.time() - start_t
                remain = (MAX_POLL - cnt) * POLL_INT
                print(f"\r[Hailuo23-Img2Vid] 处理中… {cnt}/{MAX_POLL} "
                      f"已用 {used:.1f}s 预估剩余 {remain:.1f}s", end="")
                continue
            if status.lower() == "success" and file_id:
                print(f"\r[Hailuo23-Img2Vid] 任务完成！           ")
                break
            if status.lower() == "failed":
                raise RuntimeError(f"任务失败: {raw}")
        else:
            raise RuntimeError("⏰ 轮询超时")

        retrieve_url = f"{BASE_URL}/v1/files/retrieve"
        dl_resp = requests.get(retrieve_url,
                               params={"file_id": file_id, "task_id": task_id},
                               headers={"Authorization": f"Bearer {token}"}, timeout=30)
        if dl_resp.status_code != 200:
            raise RuntimeError(f"获取下载链接失败 HTTP {dl_resp.status_code}")
        download_url = dl_resp.json()["file"]["download_url"]
        print(f"[Hailuo23-Img2Vid] 下载链接：{download_url}")

        import uuid
        output_dir = Path(folder_paths.get_output_directory())
        output_dir.mkdir(exist_ok=True)
        video_path = output_dir / f"hailuo23_i2v_{uuid.uuid4().hex[:8]}.mp4"
        _download_file(download_url, video_path)

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        video = Video(str(video_path), fps, w, h)
        print(f"[Hailuo23-Img2Vid] VIDEO 对象已生成：{video}")
        return (video, download_url)


register_node(Hailuo23Image2Video, "Hailuo23-图生视频-DMX")