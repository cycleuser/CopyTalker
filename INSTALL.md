# CopyTalker 完整安装指南

本文档提供 macOS、Linux 和 Windows 系统的详细安装步骤。

## 快速开始

### macOS (推荐方式)

```bash
# 1. 安装 Homebrew（如果没有）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. 安装系统依赖
brew install ffmpeg portaudio libsndfile

# 3. 创建虚拟环境（可选但推荐）
python3 -m venv venv
source venv/bin/activate

# 4. 安装 CopyTalker 完整版
pip install copytalker[full,cjk]

# 5. 验证安装
copytalker --help

# 6. 启动 GUI
copytalker --gui
```

### Linux (Ubuntu/Debian)

```bash
# 1. 更新包列表
sudo apt update

# 2. 安装系统依赖
sudo apt install -y ffmpeg portaudio19-dev libsndfile1 python3-dev \
                    libmecab-dev mecab mecab-ipadic-utf8 \
                    espeak-ng libespeak1

# 3. 创建虚拟环境（可选但推荐）
python3 -m venv venv
source venv/bin/activate

# 4. 安装 CopyTalker 完整版
pip install copytalker[full,cjk]

# 5. 验证安装
copytalker --help

# 6. 启动 GUI
copytalker --gui
```

### Windows

```bash
# 1. 安装 FFmpeg
# 从 https://ffmpeg.org/download.html 下载
# 解压并添加到系统 PATH

# 2. 安装 CopyTalker（不需要额外系统依赖）
pip install copytalker[full,cjk]

# 3. 验证安装
copytalker --help

# 4. 启动 GUI
copytalker --gui
```

## TTS 引擎依赖详解

CopyTalker 支持多种 TTS 引擎，每种有不同的依赖要求。

### 1. Kokoro TTS（推荐⭐）

**特点：** 高质量神经网络TTS，离线使用，支持情感

**依赖：**
- 基础包：`kokoro>=0.1.0`（包含在 `full` 中）
- CJK 支持：`cn2an`, `pypinyin`, `jieba`（中文）
- CJK 支持：`fugashi`, `jaconv`, `unidic-lite`（日文）

**安装命令：**
```bash
# macOS & Linux
pip install copytalker[full,cjk]

# 或单独安装
pip install kokoro cn2an pypinyin jieba fugashi jaconv unidic-lite
```

**测试：**
```python
from copytalker import AppConfig, TranslationPipeline

config = AppConfig()
config.tts.engine = "kokoro"
config.tts.voice = "zf_xiaobei"  # 中文女声
print("Kokoro TTS 已就绪")
```

### 2. Edge TTS（云端）

**特点：** 微软 Azure 云端服务，音质好，需要网络

**依赖：**
- `edge-tts>=6.1.0`

**安装命令：**
```bash
pip install edge-tts>=6.1.0
```

**测试：**
```bash
# 列出可用声音
edge-tts --list-voices

# 测试生成语音
edge-tts --voice zh-CN-XiaoxiaoNeural --text "你好世界" --write-media test.mp3
```

**故障排查：**
如果 GUI 显示 "Edge TTS is not available"：
1. 检查网络连接
2. 确认已安装：`pip show edge-tts`
3. 测试 CLI：`edge-tts --list-voices`
4. 检查是否能访问 Azure：`ping azure.microsoft.com`

### 3. pyttsx3（离线备用）

**特点：** 完全离线，系统级 TTS，音质一般

**依赖：**
- macOS: 无需额外依赖（使用系统 NSSpeechSynthesizer）
- Linux: `espeak-ng`
- Windows: 预编译二进制

**安装命令：**
```bash
# macOS
pip install pyttsx3

# Linux
sudo apt install espeak-ng libespeak1
pip install pyttsx3

# Windows
pip install pyttsx3
```

**测试：**
```python
import pyttsx3
engine = pyttsx3.init()
engine.say("Hello World")
engine.runAndWait()
```

### 4. IndexTTS（语音克隆）

**特点：** 支持情感语音克隆，需要参考音频

**依赖：**
- `pynini>=2.1.5`
- `soundfile>=0.12.0`

**安装命令：**
```bash
pip install pynini soundfile
```

**注意：** Pinyin 在某些系统上可能需要从源码编译。

### 5. Fish-Speech（高级语音克隆）

**特点：** 50+ 种情感标签，高质量语音克隆

**依赖：**
- `soundfile>=0.12.0`（本地推理）
- `fish-audio-sdk>=0.1.0`（云 API）

**安装命令：**
```bash
# 本地推理
pip install soundfile

# 或使用云 API
pip install fish-audio-sdk httpx
```

## 常见错误及解决方案

### 错误 1: "No TTS engine available"

**原因：** 没有安装任何可用的 TTS 引擎

**解决：**
```bash
# 安装所有 TTS 引擎
pip install copytalker[full]

# 或至少安装一个
pip install kokoro  # 推荐
# 或
pip install edge-tts
# 或
pip install pyttsx3
```

### 错误 2: "Edge TTS is not available"

**原因：** Edge TTS 未安装或网络问题

**解决：**
```bash
# 重新安装
pip uninstall edge-tts
pip install edge-tts>=6.1.0

# 测试
edge-tts --list-voices

# 如果失败，检查网络
ping 8.8.8.8
```

### 错误 3: Kokoro 无法生成中文/日文

**原因：** 缺少 CJK 语言处理依赖

**解决：**
```bash
# 安装中文支持
pip install cn2an pypinyin jieba

# 安装日文支持
pip install fugashi jaconv mojimoji unidic-lite

# 或一次性安装
pip install copytalker[cjk]
```

### 错误 4: PyAudio 安装失败（macOS/Linux）

**原因：** 系统 PortAudio 开发库缺失

**解决：**
```bash
# macOS
brew install portaudio
pip install pyaudio

# Ubuntu/Debian
sudo apt install portaudio19-dev
pip install pyaudio

# 或者继续使用默认的 sounddevice（推荐）
pip install sounddevice
```

### 错误 5: mecab-python3 安装失败（Linux）

**原因：** 缺少 MeCab 系统库

**解决：**
```bash
# Ubuntu/Debian
sudo apt install libmecab-dev mecab mecab-ipadic-utf8
pip install mecab-python3

# Fedora
sudo dnf install mecab mecab-devel mecab-ipadic
pip install mecab-python3
```

## 最小化安装

如果只需要基本功能：

```bash
# 仅安装核心功能（不含 TTS）
pip install copytalker

# 手动选择安装一个 TTS
pip install pyttsx3  # 最简单，离线
# 或
pip install edge-tts  # 需要网络
```

## 完整功能安装

安装所有功能：

```bash
# macOS
brew install ffmpeg portaudio libsndfile
pip install copytalker[full,cjk,indextts,fish-speech]

# Linux
sudo apt install ffmpeg portaudio19-dev libsndfile1 libmecab-dev espeak-ng
pip install copytalker[full,cjk,indextts,fish-speech]

# Windows
pip install copytalker[full,cjk,indextts,fish-speech]
```

## 验证安装

运行以下命令验证：

```bash
# 查看帮助
copytalker --help

# 查看版本
copytalker --version

# 列出支持的语言
copytalker list-languages

# 列出中文语音
copytalker list-voices --language zh

# 启动 GUI
copytalker --gui
```

## Python API 测试

```python
from copytalker import AppConfig, TranslationPipeline

# 配置
config = AppConfig()
config.stt.language = "auto"
config.translation.target_lang = "zh"
config.tts.engine = "kokoro"  # 或 "edge-tts", "pyttsx3"
config.tts.voice = "zf_xiaobei"

# 测试 TTS
if config.tts.engine == "kokoro":
    print("✓ Kokoro TTS 可用")
elif config.tts.engine == "edge-tts":
    print("✓ Edge TTS 可用")
elif config.tts.engine == "pyttsx3":
    print("✓ pyttsx3 可用")

print(f"当前 TTS 引擎：{config.tts.engine}")
print(f"当前语音：{config.tts.voice}")
```

## 性能优化建议

1. **使用 GPU 加速**（NVIDIA）：
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Apple Silicon Mac**（M1/M2/M3）：
   ```bash
   # 使用 MPS 加速
   pip install torch torchvision torchaudio
   ```

3. **内存优化**：
   - 使用较小的 Whisper 模型（tiny/base）
   - 使用 NLLB-200-distilled-600M 而非大模型

## 卸载与清理

```bash
# 卸载 CopyTalker
pip uninstall copytalker

# 清除缓存模型
rm -rf ~/.cache/copytalker

# 清除配置文件
rm -rf ~/.config/copytalker
```

## 获取帮助

- GitHub Issues: https://github.com/cycleuser/CopyTalker/issues
- 文档：https://github.com/cycleuser/CopyTalker#readme
- 社区讨论：https://github.com/cycleuser/CopyTalker/discussions
