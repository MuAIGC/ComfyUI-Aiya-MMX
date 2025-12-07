给你一份「少女风」配色 + 徽章 + 居中排版的 README，直接复制即可生效。

```markdown
<p align="center">
  <img src="https://github.com/MuAIGC/ComfyUI-Aiya-MMX/raw/main/MMXlab-600x600.jpg" width="280"/>
</p>

<h1 align="center">
  <span style="color:#ff7eb9;">💕</span> ComfyUI-Aiya-MMX <span style="color:#ff7eb9;">💕</span>
</h1>

<p align="center">
  <b>一键加水印 · 调尺寸 · 保存 · 抽卡</b><br>
  全部带 Emoji 的少女风节点包
</p>

<p align="center">
  <img src="https://img.shields.io/badge/ComfyUI-★★★★★-ff7eb9?style=flat-square"/>
  <img src="https://img.shields.io/badge/Python-3.8+-ff7eb9?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-ff7eb9?style=flat-square"/>
  <img src="https://img.shields.io/badge/Platform-Linux%20%7C%20Win%20%7C%20macOS-ff7eb9?style=flat-square"/>
</p>
```

---

## 📦 节点一览 · 少女心速览

| 节点 | 功能 | 少女风亮点 |
| --- | --- | --- |
| **图像稳定水印** 💕 | 单张/批量加透明水印 | 下拉选图+一键刷新 |
| **视频稳定水印** 🌸 | 4K@50 fps 流式水印 | 内存<3 GB、音轨无损 |
| **视频强制落盘+尺寸** ✨ | WAN/任何对象→真实路径 | 零属性依赖 |
| **Nano-Banana Pro** 🍌 | 文生图/图生图/14图 | 最高分辨率自动选 |
| **一键保存 JPG** 🖼 | 保存+同步提示词 txt | 支持日期变量 |
| **简易提示词&分辨率** 🎀 | 比例一键锁、空潜输出 | 8 倍数对齐 |

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

## 📂 目录结构

```
ComfyUI-Aiya-MMX/
├─ nodes/               ← 全部节点源码
├─ watermarks/          ← 你的透明 PNG 放这里
├─ date_variable.py     ← 日期占位符核心
├─ watermark_util.py    ← 水印工具库
├─ register.py          ← 统一注册器
├─ utils.py             ← tensor/PIL 互转
├─ pyproject.toml       ← PyPI 发布配置
└─ README.md            ← 本文件（少女风版）
```

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
