# 💕 ComfyUI-Aiya-MMX 使用指南

> 一键加水印、调尺寸、保存、抽卡——全部带 Emoji 的少女风节点包！

---

## 📦 包含节点一览

| 节点名称 | 功能 | 特色 |
|---|---|---|
| **图像稳定水印** 💕 | 单张/批量加透明水印 | 下拉选图+一键刷新 |
| **视频稳定水印** 🌸 | 4K@50 fps 流式水印 | 内存<3 GB、音轨无损 |
| **视频强制落盘+尺寸** ✨ | WAN/任何对象→真实路径 | 零属性依赖 |
| **Nano-Banana Pro** 🍌 | 文生图/图生图/14 图 | 最高分辨率自动选 |
| **一键保存 JPG** 🖼 | 保存+同步提示词 txt | 支持日期变量 |
| **简易提示词&分辨率** 🎀 | 比例一键锁、空潜输出 | 8 倍数对齐 |

---

## 🚀 3 步快速上手

1. 克隆到 ComfyUI  
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/MuAIGC/ComfyUI-Aiya-MMX.git
   ```

2. 放入水印  
   把透明 PNG 扔进  
   `ComfyUI/custom_nodes/ComfyUI-Aiya-MMX/watermarks/`  
   节点里点「🔄 刷新」立刻识别。

3. 拖线即用  
   - 图像/视频水印：连「位置/透明度/边距」→ 一键出图  
   - 日期变量：文件名写 `%Aiya:yyyyMMdd%` 自动替换  
   - 抽卡：改 seed → 新图立即生成

---

## 🔧 进阶玩法

| 技巧 | 输入示例 | 输出 |
|---|---|---|
| **日期归档** | `输出/%Aiya:yyyyMMdd%_img` | `输出/20251207_img_00001.png` |
| **多图合成** | 14 张参考图 + 提示词「图1 加图2 风格」 | 自动映射端口→最高分辨率 |
| **视频流** | WAN → 强制落盘 → 水印 → SaveVideo | 4K 50 fps 实时叠加 |

---

## ⚙️ 依赖与环境

- **零额外依赖**——仅使用 ComfyUI 自带库 + 标准库  
- **GPU/CPU 均可**——水印节点自动内存自适应  
- **Linux / Windows / macOS** 全平台支持

---

## 📂 目录结构

```
ComfyUI-Aiya-MMX/
├─ nodes/               ← 全部节点代码
├─ watermarks/          ← 你的透明 PNG 放这里
├─ date_variable.py     ← 日期占位符核心
├─ watermark_util.py    ← 水印工具库
├─ register.py          ← 统一注册器
├─ utils.py             ← 自给自足 tensor/PIL 互转
├─ pyproject.toml       ← PyPI 发布配置
└─ README.md            ← 本文件
```

---

## 🌟 更新与反馈
- **Issue / PR**：欢迎到 [GitHub 仓库](https://github.com/MuAIGC/ComfyUI-Aiya-MMX) 提需求  
- **微信群**：
<div align="center">
  <img src="https://github.com/MuAIGC/ComfyUI-Aiya-MMX/raw/main/MMXlab-600x600.jpg" alt="MMX Lab" width="300"/>
<p>微信号 MMXaigc（说明来由验证会更快一些）</p>
</div>
---
