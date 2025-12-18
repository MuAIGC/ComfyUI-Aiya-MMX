<p align="center">
  <img src="https://github.com/MuAIGC/ComfyUI-Aiya-MMX/raw/main/ComfyUI-Aiya-MMX-B.jpg" width="90%"/>
</p>

<h1 align="center">💕 ComfyUI-Aiya-MMX 💕</h1>

<p align="center">
  <b>一键加水印 · 调尺寸 · 保存 · 抽卡</b><br>
  全部带 Emoji 的少女风节点包
</p>

<p align="center">
  <img src="https://img.shields.io/badge/ComfyUI-★★★★★-FF7EB9?style=flat-square"/>
  <img src="https://img.shields.io/badge/Python-3.8+-FF7EB9?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-FF7EB9?style=flat-square"/>
  <img src="https://img.shields.io/badge/Platform-Linux%7CWin%7CmacOS-FF7EB9?style=flat-square"/>
</p>

---

## 📦 节点一览 · 少女心速览
<!-- 📦 节点一览 · 少女心速览（2024-12-18 更新） -->
| 节点 | 功能 | 少女风亮点 |
|---|---|---|
| **图像稳定水印** 💕 | 单张/批量加透明水印 | 下拉选图+一键刷新 |
| **视频稳定水印** 🌸 | 4K@50 fps 流式水印 | 内存<3 GB、音轨无损 |
| **视频强制落盘+尺寸** ✨ | WAN/任何对象→真实路径 | 零属性依赖 |
| **一键保存 JPG** 🖼 | 保存+同步提示词 txt | 支持日期变量 |
| **简易提示词&分辨率** 🎀 | 比例一键锁、空潜输出 | 8 倍数对齐 |

### 🍌 Nano-Banana 全家桶
| 节点 | 功能 | 少女风亮点 |
|---|---|---|
| Nano-Banana Pro · NanoBanana_Pro_DMX | 文生图 / 图生图 / 14 图 | 最高分辨率自动选 |
| Nano-Banana Pro（原版） | 同上，兼容旧工作流 | legacy 友好 |

### 🎬 Hailuo23 视频三件套
| 节点 | 功能 | 少女风亮点 |
|---|---|---|
| Hailuo23-文生视频-DMX | 提示词直接出视频 | 首尾帧自动补全 |
| Hailuo23-图生视频-DMX | 参考图→视频 | 动作一致性 UP |
| Hailuo23-首尾帧生视频-DMX | 给定首帧+尾帧 | 中间过渡 AI 脑补 |

### 🎙 MiniMax 语音小剧场
| 节点 | 功能 | 少女风亮点 |
|---|---|---|
| MiniMax TTS 文字转语音_DMX | 单角色朗读 | 支持 30+ 音色 |
| MiniMax TTS 多人对话_DMX | 多角色剧本 | 一句话自动分角色 |
| MiniMax TTS 音色选择器_DMX | 可视化选声 | 实时试听，少女音秒选 |

### 🌈 创意加餐
| 节点 | 功能 | 少女风亮点 |
|---|---|---|
| SeeDream45_DMX | 二次元 / 写实双修 | 4.5 代新底模 |
| 文图生视频提示词_DMX | 自动优化视频 Prompt | 中英双语一键润色 |
| DownloadVideo | 网络视频直链下载 | 支持 4K 去水印 |

---

## 🚀 3 步快速上手

1. **克隆到 ComfyUI**
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/MuAIGC/ComfyUI-Aiya-MMX.git
   ```

2. **放入水印**
   把透明 PNG 扔进  
   `ComfyUI/custom_nodes/ComfyUI-Aiya-MMX/watermarks/`  
   节点里点「🔄 刷新」立刻识别！

3. **拖线即用**
   - 图像/视频水印：连「位置/透明度/边距」→ 一键出图  
   - 日期归档：文件名写 `%Aiya:yyyyMMdd%` 自动替换  
   - 抽卡：改 seed → 新图立即生成

---

## 🎨 进阶少女玩法

| 技巧 | 输入示例 | 输出 |
| --- | --- | --- |
| **日期归档** | `输出/%Aiya:yyyyMMdd%_img` | `输出/20251207_img_00001.png` |
| **多图合成** | 14 张参考图 + 提示词「图1 加图2 风格」 | 自动映射端口→最高分辨率 |
| **视频流** | WAN → 强制落盘 → 水印 → SaveVideo | 4K 50 fps 实时叠加 |

---

## ⚙️ 环境与依赖

- **零额外依赖**——仅使用 ComfyUI 自带库 + 标准库  
- **GPU/CPU 均可**——自动内存自适应  
- **全平台** Linux / Windows / macOS

---

## 🌟 更新与反馈

<p align="center">
  <a href="https://github.com/MuAIGC/ComfyUI-Aiya-MMX/issues">📮 Issue / PR</a> 
  · 
  <a href="https://github.com/MuAIGC/ComfyUI-Aiya-MMX/wiki">📖 Wiki</a>
</p>

<p align="center">
  <img src="https://github.com/MuAIGC/ComfyUI-Aiya-MMX/raw/main/MMXlab-600x600.jpg" width="250"/>
  <br>
  <b>微信号 MMXaigc</b><br>
  <small>说明来由验证更快哦</small>
</p>

---

<p align="center">
  <b>Made with 💕 by MuAIGC</b>
</p>
```

直接复制→粘贴到 `README.md` → Commit，仓库首页立刻粉嫩！
