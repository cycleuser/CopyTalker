# CopyTalker

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/copytalker.svg)](https://badge.fury.io/py/copytalker)

**CopyTalker** is a cross-modal data conversion driven asynchronous multi-voice translation system. It enables real-time speech-to-speech translation with support for multiple languages and voices, utilizing state-of-the-art machine learning models for speech recognition, translation, and synthesis.

## Features

- **Real-time Speech Translation**: Instantly translate spoken language to another language with voice output
- **Multi-language Support**: Supports translation between 9 languages including English, Chinese, Japanese, Korean, French, German, Spanish, Russian, and Arabic
- **Multiple TTS Engines**: Kokoro (high-quality neural TTS), Edge TTS (cloud-based), pyttsx3 (offline)
- **Cross-modal Conversion**: Seamless conversion from speech to text to translated speech
- **Asynchronous Processing**: Efficient parallel processing with minimal latency
- **Simple GUI**: Easy-to-use Tkinter graphical interface
- **Offline Capabilities**: Download models for offline usage

## Supported Languages

| Code | Language |
|------|----------|
| en | English |
| zh | Chinese (Simplified) |
| ja | Japanese |
| ko | Korean |
| fr | French |
| de | German |
| es | Spanish |
| ru | Russian |
| ar | Arabic |

## Installation

### From PyPI (Recommended)

```bash
pip install copytalker
```

### With All Features

```bash
# Full installation with all TTS engines
pip install copytalker[full]

# With CJK language support
pip install copytalker[full,cjk]

# Development installation
pip install copytalker[dev]
```

### From Source

```bash
git clone https://github.com/EasyCam/CopyTalker.git
cd CopyTalker
pip install -e .[full]
```

### System Dependencies

CopyTalker requires FFmpeg and PortAudio for audio processing:

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg portaudio19-dev
```

**macOS:**
```bash
brew install ffmpeg portaudio
```

**Windows:**
Download FFmpeg from https://ffmpeg.org/download.html and add to PATH.

## Quick Start

### Command Line Interface

```bash
# Start real-time translation (English to Chinese)
copytalker translate --target zh

# With auto-detection of source language
copytalker translate --source auto --target ja

# Specify TTS voice
copytalker translate --target zh --voice zf_xiaobei

# Use specific TTS engine
copytalker translate --target en --tts-engine edge-tts

# List available voices
copytalker list-voices --language zh

# List supported languages
copytalker list-languages
```

### GUI Mode

```bash
# Launch graphical interface
copytalker --gui

# Or use dedicated command
copytalker-gui
```

#### Screenshots

**Main Interface**

![Main Interface](images/0-interface.png)

The main window provides access to all settings, real-time transcription and translation displays, and control buttons including Start Translation, Stop, and Download Models.

**Source Language Selection**

![Source Language Selection](images/1-select-source.png)

Select the source language or choose Auto-detect to let Whisper identify the spoken language automatically.

**Target Language Selection**

![Target Language Selection](images/2-select-target.png)

Choose the target language for translation output.

**Voice Selection**

![Voice Selection](images/3-select-vioce.png)

Pick a TTS voice for the target language. Voices change dynamically based on the selected target language and TTS engine.

**TTS Engine Selection**

![TTS Engine Selection](images/4-select-tts.png)

Choose between Kokoro (high-quality neural), Edge TTS (cloud-based), pyttsx3 (offline), or auto (automatic best choice).

**Translation Model Selection**

![Translation Model Selection](images/5-select-translator.png)

Select between Helsinki-NLP (language-pair specific) or NLLB (multilingual, supports all language pairs including ja-zh).

**Translation Device Selection**

![Translation Device Selection](images/6-select-trans-device.png)

Assign the translation model to CPU or CUDA GPU to balance resources.

**TTS Device Selection**

![TTS Device Selection](images/7-select-tts-device.png)

Assign the TTS engine to CPU or CUDA GPU independently from the translation model to avoid GPU resource contention.

### Python API

```python
from copytalker import AppConfig, TranslationPipeline

# Configure
config = AppConfig()
config.stt.language = "auto"  # Auto-detect source language
config.translation.target_lang = "zh"  # Translate to Chinese
config.tts.engine = "kokoro"  # Use Kokoro TTS
config.tts.voice = "zf_xiaobei"  # Chinese female voice

# Create and start pipeline
pipeline = TranslationPipeline(config)

# Register callbacks for events
def on_transcription(event):
    print(f"Heard: {event.data.text}")

def on_translation(event):
    print(f"Translated: {event.data.translated_text}")

pipeline.register_callback("transcription", on_transcription)
pipeline.register_callback("translation", on_translation)

# Start translation
pipeline.start()

# ... (pipeline runs until stopped)

# Stop
pipeline.stop()
```

### Using Context Manager

```python
from copytalker import AppConfig, TranslationPipeline

config = AppConfig()
config.translation.target_lang = "ja"

with TranslationPipeline(config) as pipeline:
    # Pipeline is running
    input("Press Enter to stop...")
# Pipeline automatically stopped
```

## Model Management

### Pre-download Models

```bash
# Download Whisper model
copytalker download-models --whisper small

# Download Kokoro TTS model
copytalker download-models --kokoro

# Download all recommended models
copytalker download-models --all
```

### Cache Management

```bash
# Show cache info
copytalker cache --info

# Clear all cached models
copytalker cache --clear

# Clear specific model type
copytalker cache --clear whisper
```

## Configuration

CopyTalker can be configured via:

1. **Command-line arguments**
2. **Environment variables**
3. **Configuration file** (`~/.config/copytalker/config.yaml`)

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `COPYTALKER_CACHE_DIR` | Model cache directory | `~/.cache/copytalker` |
| `COPYTALKER_DEVICE` | Compute device (cpu/cuda/auto) | `auto` |
| `COPYTALKER_CONFIG` | Config file path | `~/.config/copytalker/config.yaml` |

### Configuration File Example

```yaml
audio:
  sample_rate: 16000
  vad_aggressiveness: 3

stt:
  model_size: small
  device: auto

translation:
  target_lang: zh

tts:
  engine: kokoro
  voice: zf_xiaobei
  speed: 1.0

debug: false
```

## Architecture

CopyTalker follows a modular pipeline architecture:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Audio Capture  │────▶│  Speech-to-Text │────▶│   Translation   │────▶│  Text-to-Speech │
│    (VAD)        │     │   (Whisper)     │     │ (Helsinki/NLLB) │     │    (Kokoro)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

1. **Audio Capture**: Records audio with Voice Activity Detection (WebRTC VAD)
2. **Speech Recognition**: Transcribes using Faster-Whisper
3. **Translation**: Translates using Helsinki-NLP or NLLB models
4. **Text-to-Speech**: Synthesizes using Kokoro, Edge TTS, or pyttsx3

## Development

### Setup Development Environment

```bash
git clone https://github.com/EasyCam/CopyTalker.git
cd CopyTalker
pip install -e .[dev]
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=copytalker

# Run only unit tests
pytest tests/unit/

# Run fast tests only (skip slow)
pytest -m "not slow"
```

### Code Quality

```bash
# Format code
black src/copytalker tests
isort src/copytalker tests

# Lint
ruff check src/copytalker

# Type checking
mypy src/copytalker
```

## Requirements

- Python 3.9 or higher
- FFmpeg
- PortAudio (for PyAudio)
- Audio input/output capabilities

See [pyproject.toml](pyproject.toml) for detailed Python package dependencies.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) for speech recognition
- [Helsinki-NLP](https://huggingface.co/Helsinki-NLP) for translation models
- [Facebook NLLB](https://ai.meta.com/research/no-language-left-behind/) for multilingual translation
- [Kokoro TTS](https://github.com/hexgrad/kokoro) for high-quality neural TTS
- Various TTS libraries for voice synthesis
