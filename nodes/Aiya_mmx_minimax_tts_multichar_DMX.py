# Aiya_mmx_minimax_tts_multichar_DMX.py
from __future__ import annotations
import io
import os
import re
import json
import requests
import torch
import soundfile as sf
from datetime import datetime
import folder_paths
from ..register import register_node
# 复用单音色节点的 generate_speech 逻辑与音色列表
from .Aiya_mmx_minimax_tts_DMX import MiniMaxTTS_DMX, VOICE_PRESETS


class MiniMaxTTSMultiChar_DMX:
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
    CATEGORY = "哎呀✦MMX/DMXAPI"
    OUTPUT_NODE = True

    def __init__(self):
        self.worker = MiniMaxTTS_DMX()

    # ---------------- 小工具 ----------------
    @staticmethod
    def _make_silence_tensor(sec: float, sr: int):
        n = int(sec * sr)
        return torch.zeros(1, 1, n)

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
                "model": (["speech-2.6-hd"], {"default": "speech-2.6-hd"}),
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
        script,
        voice_map,
        model,
        speed,
        pitch,
        volume,
        emotion,
        audio_format,
        sample_rate,
    ):
        if not api_key.strip():
            return ({"waveform": torch.zeros(1, 1, 1), "sample_rate": 24000}, "❌ API Key 为空")
        dialogue = self._parse_script(script)
        role2voice = self._parse_voice_map(voice_map)
        if not dialogue:
            return ({"waveform": torch.zeros(1, 1, 1), "sample_rate": 24000}, "❌ 剧本解析为空")
        if not role2voice:
            return ({"waveform": torch.zeros(1, 1, 1), "sample_rate": 24000}, "❌ 音色映射为空")

        wav_list, info_list = [], []
        for idx, item in enumerate(dialogue, 1):
            role, text = item["role"], item["text"]
            # 本句参数优先用剧本里写的，没写再回落到全局
            spd = item.get("speed", speed)
            ptc = item.get("pitch", pitch)
            emo = item.get("emotion", emotion)
            voice_id = role2voice.get(role)
            if not voice_id:
                err = f"第{idx}句角色『{role}』未在 voice_map 中找到映射，已插入静音"
                info_list.append(err)
                wav_list.append(self._make_silence_tensor(0.1, sample_rate))
                continue

            # 调用单音色 worker
            audio_dict, info = self.worker.generate_speech(
                api_key=api_key,
                text=text,
                model=model,
                voice_id=voice_id,
                speed=spd,
                pitch=ptc,
                volume=volume,
                emotion=emo,
                audio_format=audio_format,
                sample_rate=sample_rate,
            )
            if "❌" in info:
                wav_list.append(self._make_silence_tensor(0.1, sample_rate))
                info_list.append(f"第{idx}句({role}) 失败: {info}")
            else:
                wav_list.append(audio_dict["waveform"])
                info_list.append(f"#{idx}({role}|spd={spd}|ptc={ptc}|emo={emo}) {info}")

        # 拼接
        full_wave = torch.cat(wav_list, dim=-1)
        final_audio = {"waveform": full_wave, "sample_rate": sample_rate}
        final_info = "\n".join(info_list)
        return (final_audio, final_info)


register_node(MiniMaxTTSMultiChar_DMX, "MiniMax TTS 多人对话_DMX")
