# Aiya_mmx_minimax_tts_DMX.py 
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


class MiniMaxTTS_DMX:
    DESCRIPTION = (
        "💕 Aiya MiniMax TTS via DMXAPI（speech-2.6-hd）\n\n"
        "【功能】输入文本 → 输出标准 AUDIO 张量，节点自身零落盘，下游随意保存/预览\n"
        "【必填】API 密钥 & 合成文本；其余参数按需调节\n"
        "【音色】80 种官方主音色（中英粤全覆盖），其余 ID 已下架\n"
        "【模型】仅支持 speech-2.6-hd（国内 TTS TOP1，端到端 <250ms）\n"
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
    CATEGORY = "哎呀✦MMX/audio"
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
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello, this is a test. 你好，测试完毕。",
                    "placeholder": "Text to synthesize"
                }),
                "model": (["speech-2.6-hd"], {
                    "default": "speech-2.6-hd"
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
        text,
        model,
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

        # ===== 2. 音色优先级：voice_in > custom_voice_id > voice_id 下拉框 =====
        if voice_in.strip():                      # ① 外部连线优先
            final_voice_id = self.extract_voice_id(voice_in)
        elif custom_voice_id.strip():             # ② 备用自定义
            final_voice_id = self.extract_voice_id(custom_voice_id)
        else:                                     # ③ 回落自身下拉框
            final_voice_id = self.extract_voice_id(voice_id)

        api_url = "https://www.dmxapi.cn/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
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
            print(f"[MiniMax TTS DMX] 正在生成语音...")
            print(f"  文本长度: {len(text)} 字符")
            print(f"  模型: {model}")
            print(f"  音色ID: {final_voice_id}")

            response = requests.post(api_url, headers=headers, json=payload, timeout=120)
            print(f"[MiniMax TTS DMX] HTTP {response.status_code}")

            if response.status_code != 200:
                err_info = f"❌ API 错误 {response.status_code}: {response.text[:300]}"
                print(err_info)
                return ({"waveform": torch.zeros(1, 1, 1), "sample_rate": 24000}, err_info)

            # ===== 取音频数据 =====
            audio_data = None
            audio_url = response.headers.get("Audio-Url") or response.headers.get("audio-url")
            if audio_url:
                print(f"[MiniMax TTS DMX] 从响应头取得音频URL: {audio_url}")
                r = requests.get(audio_url, timeout=60)
                if r.status_code != 200:
                    err = f"❌ 下载音频失败: {r.status_code}"
                    return ({"waveform": torch.zeros(1, 1, 1), "sample_rate": 24000}, err)
                audio_data = r.content
            else:
                body = response.content
                ct = response.headers.get("Content-Type", "")
                if ct.startswith("audio/") or body.startswith((b"ID3", b"RIFF", b"\xFF\xFB", b"\xFF\xF3", b"\xFF\xE3")):
                    print("[MiniMax TTS DMX] 检测到body为音频二进制，直接使用")
                    audio_data = body
                else:
                    try:
                        result = response.json()
                        url = result.get("audio", {}).get("url")
                        if url:
                            print(f"[MiniMax TTS DMX] 从JSON取得音频URL: {url}")
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
                f"voice: {voice_id} | model: {model} | speed: {speed} | pitch: {pitch} | "
                f"emotion: {emotion} | sample_rate: {sr} | format: {audio_format} | "
                f"size: {len(audio_data)} bytes"
            )
            print(f"[MiniMax TTS DMX] ✅ 音频已就绪，数据长度: {len(audio_data)} bytes")
            return (audio_dict, info_str)

        except requests.exceptions.Timeout:
            return ({"waveform": torch.zeros(1, 1, 1), "sample_rate": 24000}, "❌ 请求超时 (120s)")
        except Exception as e:
            import traceback
            traceback.print_exc()
            err = f"❌ 错误: {str(e)}"
            print(err)
            return ({"waveform": torch.zeros(1, 1, 1), "sample_rate": 24000}, err)

register_node(MiniMaxTTS_DMX, "MiniMax TTS 文字转语音_DMX")
