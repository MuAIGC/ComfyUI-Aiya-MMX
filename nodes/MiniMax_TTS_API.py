# MiniMax_TTS_API.py
from __future__ import annotations
import os
import json
import requests
import torch
import io
from datetime import datetime
import folder_paths
from ..register import register_node
import soundfile as sf
import re

# ========== 官方 80 种主音色 ID（2025-12 更新） ==========
VOICE_PRESETS = [
    "male-qn-qingse",        # 01 青涩青年        中文
    "male-qn-jingying",      # 02 精英青年        中文
    "male-qn-badao",         # 03 霸道青年        中文
    "male-qn-daxuesheng",    # 04 青年大学生      中文
    "female-shaonv",         # 05 少女            中文
    "female-yujie",          # 06 御姐            中文
    "female-chengshu",       # 07 成熟女性        中文
    "female-tianmei",        # 08 甜美女性        中文
    "male-qn-qingse-jingpin", # 09 青涩青年-b      中文
    "male-qn-jingying-jingpin", #10 精英青年-b      中文
    "male-qn-badao-jingpin", # 11 霸道青年-b      中文
    "male-qn-daxuesheng-jingpin", #12 大学生-b      中文
    "female-shaonv-jingpin", # 13 少女-b          中文
    "female-yujie-jingpin",  # 14 御姐-b          中文
    "female-chengshu-jingpin", #15 成熟女-b        中文
    "female-tianmei-jingpin", #16 甜美女-b        中文
    "clever_boy",            # 17 聪明男童        中文
    "cute_boy",              # 18 可爱男童        中文
    "lovely_girl",           # 19 萌萌女童        中文
    "cartoon_pig",           # 20 卡通猪小琪      中文
    "bingjiao_didi",         # 21 病娇弟弟        中文
    "junlang_nanyou",        # 22 俊朗男友        中文
    "chunzhen_xuedi",        # 23 纯真学弟        中文
    "lengdan_xiongzhang",    # 24 冷淡学长        中文
    "badao_shaoye",          # 25 霸道少爷        中文
    "tianxin_xiaoling",      # 26 甜心小玲        中文
    "qiaopi_mengmei",        # 27 俏皮萌妹        中文
    "wumei_yujie",           # 28 妩媚御姐        中文
    "diadia_xuemei",         # 29 嗲嗲学妹        中文
    "danya_xuejie",          # 30 淡雅学姐        中文
    "Chinese (Mandarin)_Reliable_Executive",      # 31 沉稳高管        中文
    "Chinese (Mandarin)_News_Anchor",             # 32 新闻女声        中文
    "Chinese (Mandarin)_Mature_Woman",            # 33 傲娇御姐        中文
    "Chinese (Mandarin)_Unrestrained_Young_Man",  # 34 不羁青年        中文
    "Arrogant_Miss",                              # 35 嚣张小姐        中文
    "Robot_Armor",                                # 36 机械战甲        中文
    "Chinese (Mandarin)_Kind-hearted_Antie",      # 37 热心大婶        中文
    "Chinese (Mandarin)_HK_Flight_Attendant",     # 38 港普空姐        中文
    "Chinese (Mandarin)_Humorous_Elder",          # 39 搞笑大爷        中文
    "Chinese (Mandarin)_Gentleman",               # 40 温润男声        中文
    "Chinese (Mandarin)_Warm_Bestie",             # 41 温暖闺蜜        中文
    "Chinese (Mandarin)_Male_Announcer",          # 42 播报男声        中文
    "Chinese (Mandarin)_Sweet_Lady",              # 43 甜美女声        中文
    "Chinese (Mandarin)_Southern_Young_Man",      # 44 南方小哥        中文
    "Chinese (Mandarin)_Wise_Women",              # 45 阅历姐姐        中文
    "Chinese (Mandarin)_Gentle_Youth",            # 46 温润青年        中文
    "Chinese (Mandarin)_Warm_Girl",               # 47 温暖少女        中文
    "Chinese (Mandarin)_Kind-hearted_Elder",      # 48 花甲奶奶        中文
    "Chinese (Mandarin)_Cute_Spirit",             # 49 憨憨萌兽        中文
    "Chinese (Mandarin)_Radio_Host",              # 50 电台男主播      中文
    "Chinese (Mandarin)_Lyrical_Voice",           # 51 抒情男声        中文
    "Chinese (Mandarin)_Straightforward_Boy",     # 52 率真弟弟        中文
    "Chinese (Mandarin)_Sincere_Adult",           # 53 真诚青年        中文
    "Chinese (Mandarin)_Gentle_Senior",           # 54 温柔学姐        中文
    "Chinese (Mandarin)_Stubborn_Friend",         # 55 嘴硬竹马        中文
    "Chinese (Mandarin)_Crisp_Girl",              # 56 清脆少女        中文
    "Chinese (Mandarin)_Pure-hearted_Boy",        # 57 清澈邻家弟      中文
    "Chinese (Mandarin)_Soft_Girl",               # 58 软软女孩        中文
    "Cantonese_ProfessionalHost（F)",             # 59 粤普女主持      粤语
    "Cantonese_GentleLady",                       # 60 粤语温柔女      粤语
    "Cantonese_ProfessionalHost（M)",             # 61 粤普男主持      粤语
    "Cantonese_PlayfulMan",                       # 62 粤语活泼男      粤语
    "Cantonese_CuteGirl",                         # 63 粤语可爱女      粤语
    "Cantonese_KindWoman",                        # 64 粤语善良女      粤语
    "Santa_Claus",                                # 65 圣诞老人        英文
    "Grinch",                                     # 66 格林奇          英文
    "Rudolph",                                    # 67 鲁道夫          英文
    "Arnold",                                     # 68 阿诺德          英文
    "Charming_Santa",                             # 69 魅力圣诞老人    英文
    "Charming_Lady",                              # 70 魅力女士        英文
    "Sweet_Girl",                                 # 71 甜美女孩        英文
    "Cute_Elf",                                   # 72 可爱精灵        英文
    "Attractive_Girl",                            # 73 魅力女孩        英文
    "Serene_Woman",                               # 74 宁静女士        英文
    "English_Trustworthy_Man",                    # 75 可信男士        英文
    "English_Graceful_Lady",                      # 76 优雅女士        英文
    "English_Aussie_Bloke",                       # 77 澳洲男士        英文
    "English_Whispering_girl",                    # 78 耳语少女        英文
    "English_Diligent_Man",                       # 79 勤奋男士        英文
    "English_Gentle-voiced_man",                  # 80 温柔男声        英文
]


# ========== 节点1: 单音色TTS ==========
class MiniMaxTTS:
    DESCRIPTION = (
        "💕 Aiya MiniMax TTS（speech-2.6-hd）\n\n"
        "【功能】输入文本 → 输出标准 AUDIO 张量，节点自身零落盘，下游随意保存/预览\n"
        "【必填】API 密钥 & 合成文本；其余参数按需调节\n"
        "【音色】80 种官方主音色（中英粤全覆盖），其余 ID 已下架\n"
        "【模型】支持 speech-2.6-hd 等模型，可手动输入\n"
        "【参数】语速 0.5-2×、音高 ±12、音量 0-10、情绪 6 种、采样率 16k/24k/48k\n"
        "【输出】audio(1,1,N) 标准 dict + info 字符串（音色/模型/大小等）\n"
        "【连接】新增 voice_in 字符串口：\n"
        "  ① 有连线 → 优先使用上游音色（如独立音色选择器）\n"
        "  ② 无连线 → 回落到自身 voice_id 下拉框\n"
        "【异常】任何错误均返回合法空音频，下游不崩；看 info 端口提示\n\n"
        "========== 官方音色速查表（复制到文本节点查看） ==========\n"
        "01  male-qn-qingse                     青涩青年        中文\n"
        "02  male-qn-jingying                   精英青年        中文\n"
        "08  female-tianmei                     甜美女性        中文（默认）\n"
        "14  female-yujie-jingpin               御姐-b          中文\n"
        "31  Chinese (Mandarin)_Reliable_Executive      沉稳高管        中文\n"
        "41  Chinese (Mandarin)_Warm_Bestie             温暖闺蜜        中文\n"
        "59  Cantonese_GentleLady                       粤语温柔女      粤语\n"
        "65  Santa_Claus                                圣诞老人        英文\n"
        "80  English_Gentle-voiced_man                  温柔男声        英文"
    )

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("音频", "info")
    FUNCTION = "generate_speech"
    CATEGORY = "哎呀✦MMX/TTS"
    OUTPUT_NODE = True

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @staticmethod
    def extract_voice_id(display: str) -> str:
        return display

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "placeholder": "sk-***************************"
                }),
                "api_url": ("STRING", {
                    "default": "https://www.dmxapi.cn/v1/audio/speech",
                    "placeholder": "API请求地址"
                }),
                "model": ("STRING", {
                    "default": "speech-2.6-hd",
                    "placeholder": "模型名称，如：speech-2.6-hd"
                }),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello, this is a test. 你好，测试完毕。",
                    "placeholder": "Text to synthesize"
                }),
                "voice_id": (VOICE_PRESETS, {
                    "default": "female-tianmei-jingpin"
                }),
                "speed": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                "pitch": ("INT", {
                    "default": 0,
                    "min": -12,
                    "max": 12,
                    "step": 1,
                    "display": "slider"
                }),
                "volume": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "emotion": (["neutral", "happy", "sad", "angry", "fearful", "surprised"], {
                    "default": "neutral"
                }),
                "audio_format": (["mp3", "wav"], {
                    "default": "mp3"
                }),
                "sample_rate": ("INT", {
                    "default": 24000,
                    "min": 16000,
                    "max": 48000,
                    "step": 8000
                }),
            },
            "optional": {
                "voice_in": ("STRING", {
                    "default": "",
                    "placeholder": "外部音色ID（连线时优先）"
                }),
                "custom_voice_id": ("STRING", {
                    "default": "",
                    "placeholder": "Custom voice ID（备用）"
                }),
            }
        }

    @staticmethod
    def audio_bytes_to_tensor(data: bytes, ext: str, target_sr: int = 24000):
        wav, sr = sf.read(io.BytesIO(data))  # (N,) or (N, 2)
        if wav.ndim == 2:
            wav = wav.mean(-1)
        if sr != target_sr:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        tensor = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0)  # (1, 1, N)
        return tensor, target_sr

    def generate_speech(
        self,
        api_key,
        api_url,
        model,
        text,
        voice_id,
        speed,
        pitch,
        volume,
        emotion,
        audio_format,
        sample_rate,
        voice_in="",
        custom_voice_id="",
    ):
        # ===== 1. 基本校验 =====
        if not api_key.strip():
            return ({"waveform": torch.zeros(1, 1, 1), "sample_rate": 24000}, "❌ API Key 为空")
        if not text.strip():
            return ({"waveform": torch.zeros(1, 1, 1), "sample_rate": 24000}, "❌ 合成文本 为空")
        
        # ===== 2. 处理API URL =====
        final_api_url = api_url.strip()
        if not final_api_url:
            final_api_url = "https://www.dmxapi.cn/v1/audio/speech"
            
        # ===== 3. 处理模型名称 =====
        final_model = model.strip()
        if not final_model:
            final_model = "speech-2.6-hd"

        # ===== 4. 音色优先级：voice_in > custom_voice_id > voice_id 下拉框 =====
        if voice_in.strip():                      # ① 外部连线优先
            final_voice_id = self.extract_voice_id(voice_in)
        elif custom_voice_id.strip():             # ② 备用自定义
            final_voice_id = self.extract_voice_id(custom_voice_id)
        else:                                     # ③ 回落自身下拉框
            final_voice_id = self.extract_voice_id(voice_id)

        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": final_model,
            "input": text,
            "voice": final_voice_id,
            "output_format": "url",
            "speed": speed,
            "voice_setting": {
                "voice_id": final_voice_id,
                "speed": speed,
                "pitch": pitch,
                "emotion": emotion,
                "volume": volume,
            },
            "audio_setting": {
                "sample_rate": sample_rate,
                "format": audio_format,
            },
        }

        try:
            print(f"[MiniMax TTS] 正在生成语音...")
            print(f"  API地址: {final_api_url}")
            print(f"  模型: {final_model}")
            print(f"  文本长度: {len(text)} 字符")
            print(f"  音色ID: {final_voice_id}")

            response = requests.post(final_api_url, headers=headers, json=payload, timeout=120)
            print(f"[MiniMax TTS] HTTP {response.status_code}")

            if response.status_code != 200:
                err_info = f"❌ API 错误 {response.status_code}: {response.text[:300]}"
                print(err_info)
                return ({"waveform": torch.zeros(1, 1, 1), "sample_rate": 24000}, err_info)

            # ===== 取音频数据 =====
            audio_data = None
            audio_url = response.headers.get("Audio-Url") or response.headers.get("audio-url")
            if audio_url:
                print(f"[MiniMax TTS] 从响应头取得音频URL: {audio_url}")
                r = requests.get(audio_url, timeout=60)
                if r.status_code != 200:
                    err = f"❌ 下载音频失败: {r.status_code}"
                    return ({"waveform": torch.zeros(1, 1, 1), "sample_rate": 24000}, err)
                audio_data = r.content
            else:
                body = response.content
                ct = response.headers.get("Content-Type", "")
                if ct.startswith("audio/") or body.startswith((b"ID3", b"RIFF", b"\xFF\xFB", b"\xFF\xF3", b"\xFF\xE3")):
                    print("[MiniMax TTS] 检测到body为音频二进制，直接使用")
                    audio_data = body
                else:
                    try:
                        result = response.json()
                        url = result.get("audio", {}).get("url")
                        if url:
                            print(f"[MiniMax TTS] 从JSON取得音频URL: {url}")
                            r = requests.get(url, timeout=60)
                            if r.status_code != 200:
                                err = f"❌ 下载音频失败: {r.status_code}"
                                return ({"waveform": torch.zeros(1, 1, 1), "sample_rate": 24000}, err)
                            audio_data = r.content
                        else:
                            return ({"waveform": torch.zeros(1, 1, 1), "sample_rate": 24000}, "❌ 未找到音频URL")
                    except ValueError as e:
                        return ({"waveform": torch.zeros(1, 1, 1), "sample_rate": 24000}, f"❌ 返回体不是合法JSON: {e}")

            if audio_data is None:
                return ({"waveform": torch.zeros(1, 1, 1), "sample_rate": 24000}, "❌ 无法获取音频数据")

            # ===== 正常返回 =====
            waveform, sr = self.audio_bytes_to_tensor(audio_data, audio_format, sample_rate)
            audio_dict = {"waveform": waveform, "sample_rate": sr}
            info_str = (
                f"API: {final_api_url} | voice: {voice_id} | model: {final_model} | "
                f"speed: {speed} | pitch: {pitch} | emotion: {emotion} | "
                f"sample_rate: {sr} | format: {audio_format} | "
                f"size: {len(audio_data)} bytes"
            )
            print(f"[MiniMax TTS] ✅ 音频已就绪，数据长度: {len(audio_data)} bytes")
            return (audio_dict, info_str)

        except requests.exceptions.Timeout:
            return ({"waveform": torch.zeros(1, 1, 1), "sample_rate": 24000}, "❌ 请求超时 (120s)")
        except Exception as e:
            import traceback
            traceback.print_exc()
            err = f"❌ 错误: {str(e)}"
            print(err)
            return ({"waveform": torch.zeros(1, 1, 1), "sample_rate": 24000}, err)


# ========== 节点2: 多人对话TTS ==========
class MiniMaxTTSMultiChar:
    DESCRIPTION = (
        "💕 MiniMax 多人对话 TTS（speech-2.6-hd）\n\n"
        "【用法】\n"
        "1) script 端口每行格式：\n"
        "     角色|语速|音高|情绪:文本   （后三项可省略，默认 1.0/0/neutral）\n"
        "   例：\n"
        "     小明|1.2:今天我们去吃火锅吧！\n"
        "     小红|0.9|+2|happy:超开心！\n"
        "     小刚:我就用默认参数\n"
        "2) voice_map 端口写「角色=音色ID」映射，一行一条。\n"
        "3) 其余参数（采样率、格式等）全局默认；单独写的优先级>全局。\n"
        "4) 输出一条拼接好的长音频 + 每句 info（换行分隔）。\n"
        "5) 任意句子合成失败自动插入 0.1 s 静音，下游永不崩溃。\n"
    )

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("拼接音频", "info")
    FUNCTION = "generate_multichar_speech"
    CATEGORY = "哎呀✦MMX/TTS"
    OUTPUT_NODE = True

    def __init__(self):
        self.worker = MiniMaxTTS()

    # ---------------- 小工具 ----------------
    @staticmethod
    def _make_silence_tensor(sec: float, sr: int):
        """生成静音张量，确保返回float32类型"""
        n = int(sec * sr)
        return torch.zeros(1, 1, n, dtype=torch.float32)

    @staticmethod
    def _parse_script(script: str):
        """
        解析剧本
        每行格式：  角色|speed|pitch|emotion:文本
        返回 List[Dict{'role','speed','pitch','emotion','text'}]
        缺省值：speed=1.0  pitch=0  emotion='neutral'
        """
        lines = [ln.strip() for ln in script.splitlines() if ln.strip()]
        out = []
        for ln in lines:
            if ':' not in ln:
                continue
            head, txt = ln.split(':', 1)
            # 默认值
            role, speed, pitch, emotion = head.strip(), 1.0, 0, 'neutral'
            # 按 | 拆分最多 4 段
            parts = [p.strip() for p in head.split('|')]
            if len(parts) >= 1:
                role = parts[0]
            if len(parts) >= 2:
                try:
                    speed = float(parts[1])
                except ValueError:
                    speed = 1.0
            if len(parts) >= 3:
                try:
                    pitch = int(parts[2])
                except ValueError:
                    pitch = 0
            if len(parts) >= 4:
                emotion = parts[3] if parts[3] in {"neutral", "happy", "sad", "angry", "fearful", "surprised"} else "neutral"
            out.append({"role": role, "speed": speed, "pitch": pitch, "emotion": emotion, "text": txt.strip()})
        return out

    @staticmethod
    def _parse_voice_map(voice_map: str):
        mp = {}
        for ln in voice_map.splitlines():
            ln = ln.strip()
            if not ln or '=' not in ln:
                continue
            role, vid = ln.split('=', 1)
            mp[role.strip()] = vid.strip()
        return mp

    # ---------------- 输入端口 ----------------
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "", "placeholder": "sk-***************************"
                }),
                "api_url": ("STRING", {
                    "default": "https://www.dmxapi.cn/v1/audio/speech",
                    "placeholder": "API请求地址"
                }),
                "model": ("STRING", {
                    "default": "speech-2.6-hd",
                    "placeholder": "模型名称，如：speech-2.6-hd"
                }),
                "script": ("STRING", {
                    "multiline": True,
                    "default": "小明|1.2:今天我们去吃火锅吧！\n小红|0.9:超开心！\n小刚:我就用默认参数",
                    "placeholder": "角色|speed|pitch|emotion:文本  （后三项可省略）"
                }),
                "voice_map": ("STRING", {
                    "multiline": True,
                    "default": "小明=male-qn-qingse\n小红=female-tianmei\n小刚=male-qn-jingying",
                    "placeholder": "角色=音色ID  一行一条"
                }),
                "speed": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05, "display": "slider"
                }),
                "pitch": ("INT", {
                    "default": 0, "min": -12, "max": 12, "step": 1, "display": "slider"
                }),
                "volume": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "display": "slider"
                }),
                "emotion": (["neutral", "happy", "sad", "angry", "fearful", "surprised"], {"default": "neutral"}),
                "audio_format": (["mp3", "wav"], {"default": "mp3"}),
                "sample_rate": ("INT", {
                    "default": 24000, "min": 16000, "max": 48000, "step": 8000
                }),
            }
        }

    # ---------------- 主入口 ----------------
    def generate_multichar_speech(
        self,
        api_key,
        api_url,
        model,
        script,
        voice_map,
        speed,
        pitch,
        volume,
        emotion,
        audio_format,
        sample_rate,
    ):
        # 错误处理：API Key为空
        if not api_key.strip():
            silence = torch.zeros(1, 1, 1, dtype=torch.float32)
            return ({"waveform": silence, "sample_rate": 24000}, "❌ API Key 为空")
        
        # 处理API URL
        final_api_url = api_url.strip()
        if not final_api_url:
            final_api_url = "https://www.dmxapi.cn/v1/audio/speech"
            
        # 处理模型名称
        final_model = model.strip()
        if not final_model:
            final_model = "speech-2.6-hd"
        
        # 解析剧本和音色映射
        dialogue = self._parse_script(script)
        role2voice = self._parse_voice_map(voice_map)
        
        # 错误处理：剧本或映射为空
        if not dialogue:
            silence = torch.zeros(1, 1, 1, dtype=torch.float32)
            return ({"waveform": silence, "sample_rate": sample_rate}, "❌ 剧本解析为空")
        if not role2voice:
            silence = torch.zeros(1, 1, 1, dtype=torch.float32)
            return ({"waveform": silence, "sample_rate": sample_rate}, "❌ 音色映射为空")

        wav_list, info_list = [], []
        
        # 逐句处理对话
        for idx, item in enumerate(dialogue, 1):
            role, text = item["role"], item["text"]
            
            # 获取本句参数（优先用剧本里的，否则用全局默认）
            spd = item.get("speed", speed)
            ptc = item.get("pitch", pitch)
            emo = item.get("emotion", emotion)
            
            # 获取角色对应的音色ID
            voice_id = role2voice.get(role)
            if not voice_id:
                err = f"第{idx}句角色『{role}』未在 voice_map 中找到映射，已插入静音"
                info_list.append(err)
                wav_list.append(self._make_silence_tensor(0.1, sample_rate))
                continue

            # 调用单音色合成节点
            audio_dict, info = self.worker.generate_speech(
                api_key=api_key,
                api_url=final_api_url,
                model=final_model,
                text=text,
                voice_id=voice_id,
                speed=spd,
                pitch=ptc,
                volume=volume,
                emotion=emo,
                audio_format=audio_format,
                sample_rate=sample_rate,
            )
            
            # 处理合成结果
            if "❌" in info:
                # 合成失败，插入静音
                wav_list.append(self._make_silence_tensor(0.1, sample_rate))
                info_list.append(f"第{idx}句({role}) 失败: {info}")
            else:
                # 合成成功，确保音频是float32类型
                waveform = audio_dict["waveform"]
                if isinstance(waveform, torch.Tensor):
                    waveform = waveform.float()  # 强制转换为float32
                wav_list.append(waveform)
                info_list.append(f"#{idx}({role}|spd={spd}|ptc={ptc}|emo={emo}) {info}")

        # 拼接所有音频片段
        if wav_list:
            # 确保所有张量都是float32类型
            wav_list = [wav.float() if isinstance(wav, torch.Tensor) else wav for wav in wav_list]
            full_wave = torch.cat(wav_list, dim=-1)
        else:
            # 如果没有音频片段，返回静音
            full_wave = self._make_silence_tensor(1.0, sample_rate)
        
        # 最终确认数据类型为float32（ComfyUI标准）
        full_wave = full_wave.float()
        
        # 构造ComfyUI标准的音频输出字典
        final_audio = {
            "waveform": full_wave,      # shape: (1, 1, n_samples), dtype: float32
            "sample_rate": sample_rate   # 采样率
        }
        
        # 生成信息输出
        final_info = "\n".join(info_list)
        
        return (final_audio, final_info)


# ========== 节点3: 音色选择器 ==========
class MiniMaxVoicePicker:
    DESCRIPTION = (
        "💕 哎呀✦MiniMax 音色选择器\n\n"
        "【用途】单独输出一个 voice_id 字符串，可连接下游 TTS 节点\n"
        "【列表】80 种官方主音色（中英粤全覆盖），下拉框即拿即用\n"
        "【连接】将本节点输出的「voice_id」接入「MiniMax TTS」的 custom_voice_id 口即可生效\n"
        "【好处】① 复用音色 ② 一键切换 ③ 工作流更直观"
    )

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("voice_in",)
    FUNCTION = "pick_voice"
    CATEGORY = "哎呀✦MMX/TTS"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "音色选择": (VOICE_PRESETS, {
                    "default": "female-tianmei-jingpin",
                    "label": "官方主音色（80 种）"
                }),
            }
        }

    def pick_voice(self, 音色选择):
        # 下拉框值本身就是合法 ID，直接返回
        return (音色选择,)


# ========== 注册所有节点 ==========
register_node(MiniMaxTTS, "MiniMax TTS 文字转语音")
register_node(MiniMaxTTSMultiChar, "MiniMax TTS 多人对话")
register_node(MiniMaxVoicePicker, "MiniMax TTS音色选择器")
