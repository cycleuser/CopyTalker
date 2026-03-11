# CopyTalker 安装和使用指南总结

## 📋 目录

1. [快速开始](#快速开始)
2. [详细安装指南](#详细安装指南)
3. [TTS 引擎依赖](#tts-引擎依赖)
4. [故障排查](#故障排查)
5. [使用示例](#使用示例)

---

## 快速开始

### macOS 用户

```bash
# 方法 1: 使用自动安装脚本（推荐）
curl -fsSL https://raw.githubusercontent.com/cycleuser/CopyTalker/main/install.sh -o install.sh
chmod +x install.sh
./install.sh

# 方法 2: 手动安装
brew install ffmpeg portaudio libsndfile
pip install copytalker[full,cjk]
copytalker --gui
```

### Linux 用户 (Ubuntu/Debian)

```bash
# 方法 1: 使用自动安装脚本（推荐）
curl -fsSL https://raw.githubusercontent.com/cycleuser/CopyTalker/main/install.sh -o install.sh
chmod +x install.sh
./install.sh

# 方法 2: 手动安装
sudo apt update
sudo apt install -y ffmpeg portaudio19-dev libsndfile1 libmecab-dev mecab mecab-ipadic-utf8
pip install copytalker[full,cjk]
copytalker --gui
```

### Windows 用户

```powershell
# 方法 1: 使用 PowerShell 安装脚本（推荐）
Invoke-WebRequest -Uri https://raw.githubusercontent.com/cycleuser/CopyTalker/main/install.ps1 -OutFile install.ps1
.\install.ps1

# 方法 2: 手动安装
# 1. 从 https://ffmpeg.org/download.html 下载 FFmpeg
# 2. 添加到 PATH
# 3. pip install copytalker[full,cjk]
# 4. copytalker --gui
```

---

## 详细安装指南

### 完整文档

- **中文**: [`README_CN.md`](README_CN.md)
- **英文**: [`README.md`](README.md)
- **详细安装指南**: [`INSTALL.md`](INSTALL.md)

### 安装脚本

| 系统 | 脚本 | 说明 |
|------|------|------|
| macOS/Linux | [`install.sh`](install.sh) | 自动安装所有依赖 |
| Windows | [`install.ps1`](install.ps1) | PowerShell 自动安装 |

### TTS 检查工具

运行以下脚本检查 TTS 引擎状态：

```bash
python check_tts.py
```

该脚本会：
- 检测已安装的 TTS 引擎
- 显示缺失的依赖
- 提供安装建议

---

## TTS 引擎依赖

### 1. Kokoro TTS（推荐⭐）

**特点：** 高质量神经网络，离线使用，支持情感

**安装：**
```bash
pip install kokoro cn2an pypinyin jieba        # 中文
pip install fugashi jaconv unidic-lite         # 日文
# 或一次性安装
pip install copytalker[full,cjk]
```

### 2. Edge TTS（云端）

**特点：** 微软 Azure 云端服务，音质好，需要网络

**安装：**
```bash
pip install edge-tts>=6.1.0
edge-tts --list-voices  # 测试
```

### 3. pyttsx3（离线备用）

**特点：** 完全离线，系统级 TTS，音质一般

**安装：**
```bash
# macOS
pip install pyttsx3

# Linux
sudo apt install espeak-ng libespeak1
pip install pyttsx3
```

---

## 故障排查

### 常见错误及解决方案

#### 错误 1: "No TTS engine available"

**解决：**
```bash
pip install copytalker[full]
# 或至少安装一个
pip install kokoro      # 推荐
pip install edge-tts    # 云端
pip install pyttsx3     # 离线
```

#### 错误 2: "Edge TTS is not available"

**解决：**
```bash
# 检查安装
pip show edge-tts

# 重新安装
pip uninstall edge-tts
pip install edge-tts>=6.1.0

# 测试
edge-tts --list-voices

# 检查网络
ping azure.microsoft.com
```

#### 错误 3: Kokoro 无法生成中文/日文

**解决：**
```bash
# 安装 CJK 支持
pip install copytalker[cjk]

# 或单独安装
pip install cn2an pypinyin jieba        # 中文
pip install fugashi jaconv unidic-lite  # 日文
```

#### 错误 4: PyAudio 安装失败（macOS/Linux）

**解决：**
```bash
# macOS
brew install portaudio
pip install pyaudio

# Linux
sudo apt install portaudio19-dev
pip install pyaudio

# 或使用默认的 sounddevice（推荐）
pip install sounddevice
```

### 诊断工具

使用 `check_tts.py` 进行完整诊断：

```bash
python check_tts.py
```

输出示例：
```
============================================================
CopyTalker TTS Engine Availability Checker
============================================================

Audio Backend:
✓ sounddevice

TTS Engines:
✓ Kokoro TTS
  Kokoro TTS is available
✗ Edge TTS
  Package not installed: No module named 'edge_tts'
✓ pyttsx3
  pyttsx3 is available (offline system TTS)

============================================================
✓ 2 TTS engines available
============================================================
```

---

## 使用示例

### 命令行界面

```bash
# 实时翻译（英语到中文）
copytalker translate --target zh

# 自动检测源语言
copytalker translate --source auto --target ja

# 指定 TTS 语音
copytalker translate --target zh --voice zf_xiaobei

# 使用特定 TTS 引擎
copytalker translate --target en --tts-engine edge-tts

# 列出可用语音
copytalker list-voices --language zh

# 下载模型
copytalker download-models --all
```

### 图形界面

```bash
# 启动 GUI
copytalker --gui
```

GUI 功能：
- 🎛️ 完整的设置面板
- 📝 实时转录和翻译显示
- 🎤 语音克隆功能
- 📥 模型下载管理（支持按语言选择）
- 🎚️ 设备选择（CPU/GPU）

### Python API

```python
from copytalker import AppConfig, TranslationPipeline

# 配置
config = AppConfig()
config.stt.language = "auto"
config.translation.target_lang = "zh"
config.tts.engine = "kokoro"
config.tts.voice = "zf_xiaobei"

# 创建流水线
pipeline = TranslationPipeline(config)

# 注册回调
def on_translation(event):
   print(f"翻译结果：{event.data.translated_text}")

pipeline.register_callback("translation", on_translation)

# 开始
pipeline.start()

# ... 运行直到停止

# 停止
pipeline.stop()
```

---

## 文件清单

| 文件 | 说明 |
|------|------|
| [`README_CN.md`](README_CN.md) | 中文文档（含详细 TTS 依赖说明） |
| [`README.md`](README.md) | 英文文档（含详细 TTS 依赖说明） |
| [`INSTALL.md`](INSTALL.md) | 完整安装指南 |
| [`install.sh`](install.sh) | macOS/Linux自动安装脚本 |
| [`install.ps1`](install.ps1) | Windows 自动安装脚本 |
| [`check_tts.py`](check_tts.py) | TTS 引擎检查工具 |

---

## 获取帮助

- **GitHub Issues**: https://github.com/cycleuser/CopyTalker/issues
- **文档**: https://github.com/cycleuser/CopyTalker#readme
- **讨论**: https://github.com/cycleuser/CopyTalker/discussions

---

## 系统要求

| 组件 | 要求 |
|------|------|
| Python | 3.9+ |
| 操作系统 | macOS 10.14+, Linux (Ubuntu 18.04+), Windows 10+ |
| 内存 | 最低 4GB，推荐 8GB+ |
| 磁盘空间 | 基础安装 2GB，完整安装 10GB+ |
| 网络 | 下载模型时需要（可离线使用） |

---

祝您使用愉快！🎙️
