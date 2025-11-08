import threading
import queue
import time
import numpy as np
import pyaudio
import webrtcvad
from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import argparse
import logging
from collections import deque
import re
import json
import os
import subprocess
import tempfile
import glob
import io
import wave

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 定义支持的语言列表 (用于用户交互和内部映射) ---
# 格式: [(语言代码, 语言名称), ...]
SUPPORTED_LANGUAGES = [
    ('en', 'English'),
    ('zh', 'Chinese (Simplified)'),
    ('es', 'Spanish'),
    ('fr', 'French'),
    ('de', 'German'),
    ('ja', 'Japanese'),
    ('ko', 'Korean'),
    ('ru', 'Russian'),
    ('ar', 'Arabic'),
]

# 用于内部映射的字典，将标准语言代码映射到模型使用的语言代码
SUPPORTED_TRANSLATION_LANG_MAP = {
    'zh-cn': 'zh', 'zh-tw': 'zh', 'cmn': 'zh',
    'spa': 'es',
    'fra': 'fr',
    'deu': 'de',
    'jpn': 'ja',
    'kor': 'ko',
    'rus': 'ru',
    'ara': 'ar',
    'ja': 'jpn'
}
for code, _ in SUPPORTED_LANGUAGES:
    SUPPORTED_TRANSLATION_LANG_MAP.setdefault(code, code)

# NLLB模型使用的语言代码映射
NLLB_LANG_CODE_MAP = {
    'en': 'eng_Latn',
    'zh': 'zho_Hans',
    'es': 'spa_Latn',
    'fr': 'fra_Latn',
    'de': 'deu_Latn',
    'ja': 'jpn_Jpan',
    'ko': 'kor_Hang',
    'ru': 'rus_Cyrl',
    'ar': 'arb_Arab',
}

# Piper TTS 基础路径
PIPER_VOICES_PATH = "/home/fred/Documents/GitHub/piper-voices"

# 语言代码到piper-voices文件夹名的映射
PIPER_LANG_CODE_MAP = {
    'en': 'en',
    'zh': 'zh',
    'es': 'es',
    'fr': 'fr',
    'de': 'de',
    'ja': 'ja',
    'ko': 'ko',
    'ru': 'ru',
    'ar': 'ar',
}

# espeak-ng 语言映射（更新了中文支持）
ESPEAK_LANG_MAP = {
    'en': 'en',
    'zh': 'zh',  # 使用 zh 而不是 cmn
    'es': 'es',
    'fr': 'fr',
    'de': 'de',
    'ja': 'ja',
    'ko': 'ko',
    'ru': 'ru',
    'ar': 'ar',
}

# 需要使用Piper的语言列表（因为espeak效果不好）
LANGUAGES_REQUIRING_PIPER = ['zh', 'ja', 'ko', 'ar']

AUTO_DETECT_CODE = "auto"

class FastTTS:
    """更快的本地TTS实现，支持多种引擎"""
    
    def __init__(self, engine="piper"):
        self.engine = engine
        self.piper_process = None
        self.espeak_available = self._check_espeak()
        
    def _check_espeak(self):
        """检查espeak-ng是否可用"""
        try:
            subprocess.run(['espeak-ng', '--version'], capture_output=True, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False
    
    def synthesize(self, text, lang_code, output_file=None):
        """合成语音"""
        if self.engine == "espeak" and self.espeak_available and lang_code not in LANGUAGES_REQUIRING_PIPER:
            return self._synthesize_espeak(text, lang_code, output_file)
        elif self.engine == "piper":
            return self._synthesize_piper(text, lang_code, output_file)
        else:
            raise ValueError(f"Unsupported TTS engine: {self.engine}")
    
    def _synthesize_espeak(self, text, lang_code, output_file=None):
        """使用espeak-ng合成语音（最快）"""
        espeak_lang = ESPEAK_LANG_MAP.get(lang_code, 'en')
        
        # 对于中文，使用特殊参数
        if lang_code == 'zh':
            cmd = [
                'espeak-ng',
                '-v', 'zh',  # 使用 zh
                '-s', '160',  # 稍微慢一点的语速，让中文更清晰
                '-p', '50',   # 音调
                '-w', output_file if output_file else '-',
                text
            ]
        else:
            cmd = [
                'espeak-ng',
                '-v', espeak_lang,
                '-s', '150',  # 语速
                '-w', output_file if output_file else '-',
                text
            ]
        
        if output_file:
            subprocess.run(cmd, check=True)
            return output_file
        else:
            # 直接返回音频数据
            result = subprocess.run(cmd, capture_output=True, check=True)
            return result.stdout
    
    def _synthesize_piper(self, text, lang_code, output_file=None):
        """使用Piper合成语音（优化版）"""
        # 查找最快的模型（通常是small或medium）
        model_path = self._find_fastest_model(lang_code)
        if not model_path:
            raise ValueError(f"No Piper model found for language {lang_code}")
        
        # 对于中文，调整参数
        if lang_code == 'zh':
            length_scale = '0.9'  # 中文稍微慢一点
        else:
            length_scale = '0.85'
        
        cmd = [
            'piper',
            '--model', model_path,
            '--output_file', output_file if output_file else '-',
            '--text', text,
            '--length_scale', length_scale,  # 加速语音
            '--noise_scale', '0.667',
            '--noise_w', '0.8'
        ]
        
        if output_file:
            subprocess.run(cmd, check=True)
            return output_file
        else:
            # 流式输出到stdout
            result = subprocess.run(cmd, capture_output=True, check=True)
            return result.stdout
    
    def _find_fastest_model(self, lang_code):
        """查找最快的模型（优先small/medium）"""
        piper_lang = PIPER_LANG_CODE_MAP.get(lang_code)
        if not piper_lang:
            return None
        
        lang_dir = os.path.join(PIPER_VOICES_PATH, piper_lang)
        if not os.path.exists(lang_dir):
            return None
        
        # 优先查找small或medium模型
        for size in ['small', 'medium', 'low']:
            pattern = os.path.join(lang_dir, f"*{size}*.onnx")
            models = glob.glob(pattern)
            if models:
                return models[0]
        
        # 如果没有找到，返回第一个模型
        models = glob.glob(os.path.join(lang_dir, "*.onnx"))
        return models[0] if models else None
    
    def synthesize_stream(self, text, lang_code, callback):
        """流式合成语音，通过回调函数返回音频块"""
        if self.engine == "espeak" and self.espeak_available and lang_code not in LANGUAGES_REQUIRING_PIPER:
            self._synthesize_stream_espeak(text, lang_code, callback)
        else:
            # Piper不支持真正的流式，但可以分段处理
            sentences = re.split(r'[.!?]+', text)
            for sentence in sentences:
                if sentence.strip():
                    audio_data = self.synthesize(sentence.strip(), lang_code)
                    if audio_data:
                        callback(audio_data)
    
    def _synthesize_stream_espeak(self, text, lang_code, callback):
        """espeak流式合成"""
        espeak_lang = ESPEAK_LANG_MAP.get(lang_code, 'en')
        
        if lang_code == 'zh':
            cmd = [
                'espeak-ng',
                '-v', 'zh',
                '-s', '160',
                '-p', '50',
                '--stdout',
                text
            ]
        else:
            cmd = [
                'espeak-ng',
                '-v', espeak_lang,
                '-s', '150',
                '--stdout',
                text
            ]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 读取音频数据块
        while True:
            chunk = process.stdout.read(1024)
            if not chunk:
                break
            callback(chunk)
        
        process.wait()

def scan_piper_voices():
    """递归扫描piper-voices文件夹，获取所有可用的TTS模型"""
    if not os.path.exists(PIPER_VOICES_PATH):
        logger.error(f"Piper voices directory not found at {PIPER_VOICES_PATH}")
        return {}
    
    # 构建语言到模型的映射
    lang_models = {}
    
    # 递归遍历所有子目录
    for root, dirs, files in os.walk(PIPER_VOICES_PATH):
        # 查找当前目录下的所有.onnx模型文件
        model_files = [f for f in files if f.endswith('.onnx')]
        
        if model_files:
            # 获取相对路径，确定语言代码
            rel_path = os.path.relpath(root, PIPER_VOICES_PATH)
            path_parts = rel_path.split(os.sep)
            
            # 第一级目录通常是语言代码
            if path_parts and path_parts[0] in PIPER_LANG_CODE_MAP.values():
                lang_code = path_parts[0]
                
                # 确保语言代码在映射中
                if lang_code not in lang_models:
                    lang_models[lang_code] = []
                
                # 添加模型
                for model_file in model_files:
                    model_name = os.path.splitext(model_file)[0]
                    full_model_path = os.path.join(root, model_name)
                    
                    # 使用相对路径作为模型标识，以便后续查找
                    rel_model_path = os.path.relpath(full_model_path, PIPER_VOICES_PATH)
                    lang_models[lang_code].append(rel_model_path)
    
    return lang_models

def check_piper_tts():
    """检查 Piper TTS 是否安装"""
    try:
        result = subprocess.run(['piper', '--help'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def get_piper_model_path(model_identifier):
    """根据模型标识符获取piper模型的完整路径"""
    # model_identifier可能是相对路径或模型名
    model_path = os.path.join(PIPER_VOICES_PATH, model_identifier)
    
    # 如果是相对路径，直接返回
    if os.path.exists(model_path + '.onnx'):
        return os.path.dirname(model_path)
    
    # 如果是模型名，尝试查找
    for root, dirs, files in os.walk(PIPER_VOICES_PATH):
        for file in files:
            if file == model_identifier + '.onnx':
                return root
    
    return None

def get_user_language_choice():
    print("\n--- Select Input (Source) Language ---")
    print("1. Automatic Detection")
    for i, (code, name) in enumerate(SUPPORTED_LANGUAGES, 2):
        print(f"{i}. {name} ({code})")

    source_lang_code = AUTO_DETECT_CODE
    while True:
        try:
            choice_str = input("\nPlease enter the number of your input language: ").strip()
            choice_idx = int(choice_str)
            if choice_idx == 1:
                source_lang_code = AUTO_DETECT_CODE
                print("Input language set to: Automatic Detection\n")
                break
            elif 2 <= choice_idx <= len(SUPPORTED_LANGUAGES) + 1:
                selected_code, selected_name = SUPPORTED_LANGUAGES[choice_idx - 2]
                source_lang_code = selected_code
                print(f"Input language set to: {selected_name} ({selected_code})\n")
                break
            else:
                print(f"Invalid number. Please enter a number between 1 and {len(SUPPORTED_LANGUAGES) + 1}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    print("\n--- Select Target Language ---")
    for i, (code, name) in enumerate(SUPPORTED_LANGUAGES, 1):
        print(f"{i}. {name} ({code})")

    target_lang_code = None
    while True:
        try:
            choice_str = input("\nPlease enter the number of your target language: ").strip()
            choice_idx = int(choice_str)
            if 1 <= choice_idx <= len(SUPPORTED_LANGUAGES):
                target_lang_code, target_name = SUPPORTED_LANGUAGES[choice_idx - 1]
                print(f"Target language set to: {target_name} ({target_lang_code})\n")
                break
            else:
                print(f"Invalid number. Please enter a number between 1 and {len(SUPPORTED_LANGUAGES)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    return source_lang_code, target_lang_code

def get_user_tts_choice(target_lang_code, available_models):
    """让用户选择TTS引擎和模型"""
    # 对于某些语言，强制使用Piper
    if target_lang_code in LANGUAGES_REQUIRING_PIPER:
        print(f"\nNote: {target_lang_code} requires Piper TTS for better quality.")
        engine_choice = "piper"
    else:
        print(f"\n--- Select TTS Engine for {target_lang_code} ---")
        print("1. espeak-ng (Fastest, lower quality)")
        print("2. Piper TTS (Better quality, slower)")
        
        engine_choice = None
        while True:
            try:
                choice_str = input("\nPlease select TTS engine (1-2): ").strip()
                choice_idx = int(choice_str)
                if choice_idx == 1:
                    engine_choice = "espeak"
                    print("Selected TTS engine: espeak-ng (fastest)\n")
                    break
                elif choice_idx == 2:
                    engine_choice = "piper"
                    print("Selected TTS engine: Piper TTS\n")
                    break
                else:
                    print("Invalid choice. Please enter 1 or 2.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    if engine_choice == "piper":
        # 选择Piper模型
        piper_lang_code = PIPER_LANG_CODE_MAP.get(target_lang_code)
        if not piper_lang_code:
            print(f"No Piper TTS models found for language {target_lang_code}")
            return None, None
        
        models = available_models.get(piper_lang_code, [])
        if not models:
            print(f"No TTS models found for language {target_lang_code}")
            return None, None
        
        print(f"\n--- Select Piper TTS Model for {target_lang_code} ---")
        for i, model_name in enumerate(models, 1):
            # 显示模型名（去掉路径）
            display_name = os.path.basename(model_name)
            print(f"{i}. {display_name}")
        
        while True:
            try:
                choice_str = input(f"\nPlease enter the number of your TTS model choice (1-{len(models)}): ").strip()
                choice_idx = int(choice_str)
                if 1 <= choice_idx <= len(models):
                    selected_model = models[choice_idx - 1]
                    display_name = os.path.basename(selected_model)
                    print(f"Selected TTS model: {display_name}\n")
                    return engine_choice, selected_model
                else:
                    print(f"Invalid number. Please enter a number between 1 and {len(models)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    else:
        return engine_choice, None

def load_language_model_dict():
    file_path = "language_model_dict.json"
    if not os.path.exists(file_path):
        logger.error(f"Language model dictionary file '{file_path}' not found.")
        raise FileNotFoundError(f"{file_path} not found.")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def select_model_for_pair(source_lang, target_lang, model_dict):
    key = f"{source_lang}->{target_lang}"
    fallback_key = "多语言->多语言"

    candidates = model_dict.get(key, [])
    if not candidates and fallback_key in model_dict:
        candidates = model_dict[fallback_key]

    if not candidates:
        raise ValueError(f"No models available for translation from {source_lang} to {target_lang}")

    if len(candidates) == 1:
        selected_model = candidates[0]
        print(f"Automatically selected model: {selected_model}")
        return selected_model

    print(f"\nAvailable models for {source_lang} -> {target_lang}:")
    for i, model in enumerate(candidates, 1):
        print(f"{i}. {model}")

    while True:
        try:
            choice = input(f"\nSelect a model (1-{len(candidates)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(candidates):
                selected_model = candidates[idx]
                print(f"Selected model: {selected_model}")
                return selected_model
            else:
                print(f"Please enter a number between 1 and {len(candidates)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

class SpeechProcessor:
    def __init__(self, stt_model_size="small", device="auto", source_lang=AUTO_DETECT_CODE, target_lang="en", selected_translation_model=None, tts_engine="espeak", selected_tts_model=None):
        self.audio_buffer = deque(maxlen=10)
        self.vad = webrtcvad.Vad(3)
        self.sample_rate = 16000
        self.frame_duration = 30
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)
        self.channels = 1

        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()
        self.translation_queue = queue.Queue()

        self.source_lang = source_lang
        self.target_lang = target_lang
        self.selected_translation_model = selected_translation_model
        self.tts_engine = tts_engine
        self.selected_tts_model = selected_tts_model
        
        logger.info(f"Target translation language set to: {self.target_lang}")
        logger.info(f"Source language for STT set to: {self.source_lang if self.source_lang != AUTO_DETECT_CODE else 'Automatic Detection'}")
        logger.info(f"Selected TTS engine: {self.tts_engine}")
        if self.selected_tts_model:
            logger.info(f"Selected TTS model: {os.path.basename(self.selected_tts_model)}")

        logger.info(f"Loading Whisper STT model ({stt_model_size}) on {device}...")
        compute_type = "float16" if device == "cuda" else ("int8" if device == "cpu" else "float16")
        self.stt_model = WhisperModel(stt_model_size, device=device, compute_type=compute_type)

        self.supported_translation_langs = SUPPORTED_TRANSLATION_LANG_MAP
        self.nllb_lang_codes = NLLB_LANG_CODE_MAP

        self.translation_models = {}
        self.translation_tokenizers = {}
        
        logger.info("Initializing PyAudio for TTS playback...")
        self.pa = pyaudio.PyAudio()
        
        self.tts_playing_lock = threading.Lock()
        
        # 初始化快速TTS
        self.fast_tts = FastTTS(engine=tts_engine)
        
        if tts_engine == "piper" and selected_tts_model:
            self.tts_model_dir = get_piper_model_path(selected_tts_model)
            if not self.tts_model_dir:
                logger.error(f"Failed to find TTS model {selected_tts_model}")
                raise RuntimeError(f"Failed to find TTS model {selected_tts_model}")

        self._start_threads()

    def _start_threads(self):
        self.stop_threads = False

        self.audio_thread = threading.Thread(target=self._audio_capture, name="AudioCaptureThread")
        self.audio_thread.daemon = True

        self.stt_thread = threading.Thread(target=self._speech_to_text, name="STTThread")
        self.stt_thread.daemon = True

        self.translation_thread = threading.Thread(target=self._translate_text, name="TranslationThread")
        self.translation_thread.daemon = True

        self.tts_thread = threading.Thread(target=self._text_to_speech, name="TTSThread")
        self.tts_thread.daemon = True

        self.audio_thread.start()
        self.stt_thread.start()
        self.translation_thread.start()
        self.tts_thread.start()
        logger.info("All processing threads started.")

    def _audio_capture(self):
        p = pyaudio.PyAudio()

        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.frame_size
            )
        except Exception as e:
            logger.error(f"Failed to open PyAudio stream: {e}")
            self.stop_threads = True
            return

        logger.info("Started listening for audio input...")

        voice_buffer = []
        is_speaking = False
        last_voice_time = time.time()

        while not self.stop_threads:
            with self.tts_playing_lock:
                try:
                    frame = stream.read(self.frame_size, exception_on_overflow=False)
                    audio_data = np.frombuffer(frame, dtype=np.int16)

                    is_voice = self.vad.is_speech(frame, self.sample_rate)

                    current_time = time.time()
                    if is_voice:
                        voice_buffer.append(audio_data)
                        is_speaking = True
                        last_voice_time = current_time
                    else:
                        silence_duration = current_time - last_voice_time
                        if is_speaking and silence_duration > 0.8:
                            if len(voice_buffer) > 5:
                                audio_segment = np.concatenate(voice_buffer)
                                audio_segment = audio_segment.astype(np.float32) / 32768.0
                                self.audio_queue.put(audio_segment.copy())
                                logger.debug(f"Queued audio segment of length {len(audio_segment)}")

                            voice_buffer = []
                            is_speaking = False
                        elif is_speaking:
                            voice_buffer.append(audio_data)

                    self.audio_buffer.append(audio_data)

                except Exception as e:
                    if not self.stop_threads:
                         logger.error(f"Audio capture error: {e}")
                    break

        try:
            stream.stop_stream()
            stream.close()
        except:
            pass
        try:
            p.terminate()
        except:
            pass
        logger.info("Audio capture thread stopped.")

    def _speech_to_text(self):
        logger.info("STT thread started.")
        while not self.stop_threads:
            try:
                audio = self.audio_queue.get(timeout=1.0)

                logger.info("Performing speech-to-text...")
                start_time = time.time()

                whisper_lang = None if self.source_lang == AUTO_DETECT_CODE else self.source_lang

                segments, info = self.stt_model.transcribe(
                    audio,
                    beam_size=5,
                    language=whisper_lang,
                    condition_on_previous_text=False
                )

                text = "".join(segment.text for segment in segments).strip()

                detected_lang_full = info.language if info else None
                detected_lang_confidence = info.language_probability if info else 0
                logger.debug(f"Whisper detected language: {detected_lang_full} (confidence: {detected_lang_confidence:.2f})")

                final_detected_lang = detected_lang_full if self.source_lang == AUTO_DETECT_CODE else self.source_lang
                normalized_lang = self.supported_translation_langs.get(final_detected_lang.lower(), 'en')

                process_time = time.time() - start_time
                if text:
                    logger.info(f"Speech recognized: '{text}' (Detected/Used Lang: {normalized_lang}, Time: {process_time:.2f}s)")
                    self.text_queue.put({
                        "text": text,
                        "source_lang": normalized_lang
                    })
                else:
                     logger.debug(f"No speech recognized in segment (Time: {process_time:.2f}s)")

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Speech-to-text error: {e}", exc_info=True)

    def _load_specific_translation_model(self, model_name):
        model_key = model_name

        if model_key not in self.translation_models:
            logger.info(f"Loading specific translation model: {model_name}...")

            if model_name.startswith("Helsinki-NLP/opus-mt-"):
                try:
                    tokenizer = MarianTokenizer.from_pretrained(model_name)
                    model = MarianMTModel.from_pretrained(model_name)
                except Exception as e:
                    logger.error(f"Failed to load Helsinki-NLP model {model_name}: {e}", exc_info=True)
                    raise
            elif model_name.startswith("facebook/nllb-"):
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                except Exception as e:
                    logger.error(f"Failed to load NLLB model {model_name}: {e}", exc_info=True)
                    raise
            else:
                raise ValueError(f"Unsupported model type: {model_name}")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            logger.info(f"Translation model {model_name} loaded successfully on {device}.")

            self.translation_models[model_key] = model
            self.translation_tokenizers[model_key] = tokenizer
            return tokenizer, model

        return self.translation_tokenizers[model_key], self.translation_models[model_key]

    def _translate_text(self):
        logger.info("Translation thread started.")
        while not self.stop_threads:
            try:
                data = self.text_queue.get(timeout=1.0)
                original_text = data["text"].strip()
                source_lang = data["source_lang"]

                if source_lang not in self.supported_translation_langs.values():
                    logger.warning(f"Source language '{source_lang}' not in supported list. Defaulting to 'en'.")
                    source_lang = 'en'

                if source_lang == self.target_lang:
                    logger.info(f"No translation needed: Source '{source_lang}' == Target '{self.target_lang}'. Text: '{original_text}'")
                    self.translation_queue.put({
                        "original": original_text,
                        "translated": original_text,
                        "source_lang": source_lang,
                        "target_lang": self.target_lang
                    })
                    continue

                logger.info(f"Translating text: '{original_text}' ({source_lang} -> {self.target_lang})")
                start_time = time.time()

                try:
                    tokenizer, model = self._load_specific_translation_model(self.selected_translation_model)
                except Exception as e:
                    logger.error(f"Skipping translation due to model loading failure: {e}")
                    continue

                if self.selected_translation_model.startswith("Helsinki-NLP/opus-mt-"):
                    inputs = tokenizer(original_text, return_tensors="pt", padding=True, truncation=True, max_length=400)
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=400)
                    
                    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                elif self.selected_translation_model.startswith("facebook/nllb-"):
                    src_lang_code = self.nllb_lang_codes.get(source_lang, 'eng_Latn')
                    tgt_lang_code = self.nllb_lang_codes.get(self.target_lang, 'eng_Latn')
                    
                    tokenizer.src_lang = src_lang_code
                    inputs = tokenizer(original_text, return_tensors="pt", padding=True, truncation=True, max_length=400)
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang_code), max_new_tokens=400)
                    
                    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                else:
                    logger.error(f"Unknown model type: {self.selected_translation_model}")
                    continue

                translated_text = re.sub(r'\s+', ' ', translated_text).strip()

                process_time = time.time() - start_time
                logger.info(f"Translation complete: '{translated_text}' (Time: {process_time:.2f}s)")

                self.translation_queue.put({
                    "original": original_text,
                    "translated": translated_text,
                    "source_lang": source_lang,
                    "target_lang": self.target_lang
                })

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Translation error: {e}", exc_info=True)

    def _text_to_speech(self):
        logger.info("TTS thread started.")
        while not self.stop_threads:
            try:
                data = self.translation_queue.get(timeout=1.0)
                translated_text = data["translated"].strip()

                if not translated_text:
                    logger.debug("Skipping TTS for empty translated text.")
                    continue

                logger.info(f"Synthesizing speech for: '{translated_text}'")
                start_time = time.time()

                try:
                    with self.tts_playing_lock:
                        # 对于需要Piper的语言，强制使用Piper
                        if self.target_lang in LANGUAGES_REQUIRING_PIPER or self.tts_engine == "piper":
                            self._play_piper(translated_text)
                        else:
                            # 使用espeak直接播放
                            self._play_espeak(translated_text, self.target_lang)
                            
                except Exception as e:
                    logger.error(f"Error during TTS synthesis or playback: {e}")
                    continue

                process_time = time.time() - start_time
                logger.info(f"Speech synthesis and playback completed (Time: {process_time:.2f}s)")

            except queue.Empty:
                continue
            except Exception as e:
                 logger.error(f"Text-to-speech error: {e}", exc_info=True)
    
    def _play_espeak(self, text, lang_code):
        """直接播放espeak合成的语音"""
        espeak_lang = ESPEAK_LANG_MAP.get(lang_code, 'en')
        
        # 对于中文，使用特殊参数
        if lang_code == 'zh':
            cmd1 = ['espeak-ng', '-v', 'zh', '-s', '160', '-p', '50', '--stdout', text]
        else:
            cmd1 = ['espeak-ng', '-v', espeak_lang, '-s', '150', '--stdout', text]
        
        # 使用aplay直接播放（Linux）
        cmd2 = ['aplay', '-q']
        
        p1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE)
        p2 = subprocess.Popen(cmd2, stdin=p1.stdout)
        p1.stdout.close()
        p2.wait()
    
    def _play_piper(self, text):
        """播放Piper合成的语音"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
        
        # 获取完整的模型路径
        model_full_path = os.path.join(PIPER_VOICES_PATH, self.selected_tts_model)
        model_path = model_full_path + '.onnx'
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return
        
        # 对于中文，调整参数
        if self.target_lang == 'zh':
            length_scale = '0.9'
        else:
            length_scale = '0.85'
        
        # 运行 Piper TTS
        subprocess.run([
            'piper',
            '--model', model_path,
            '--output_file', temp_path,
            '--text', text,
            '--length_scale', length_scale  # 加速
        ], check=True)
        
        # 播放生成的音频
        self._play_audio(temp_path)
        
        # 删除临时文件
        os.unlink(temp_path)

    def _play_audio(self, audio_file_path):
        """播放音频文件"""
        try:
            # 使用 ffprobe 获取音频信息
            cmd = ['ffprobe', '-v', 'error', '-show_entries', 'stream=sample_rate', '-of', 'default=noprint_wrappers=1:nokey=1', audio_file_path]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            sample_rate = int(result.stdout.strip()) if result.stdout.strip() else 22050
            
            # 使用 ffmpeg 读取音频数据
            cmd = ['ffmpeg', '-i', audio_file_path, '-f', 's16le', '-acodec', 'pcm_s16le', '-ac', '1', '-ar', str(sample_rate), '-']
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if result.returncode != 0:
                logger.error(f"Error reading audio file: {result.stderr.decode()}")
                return
                
            audio_data = np.frombuffer(result.stdout, dtype=np.int16)
            
            # 播放音频
            stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                output=True
            )
            
            stream.write(audio_data.tobytes())
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            logger.error(f"Error during audio playback: {e}")

    def shutdown(self):
        logger.info("Shutting down the speech translation system...")
        self.stop_threads = True

        self.audio_thread.join(timeout=2.0)
        self.stt_thread.join(timeout=2.0)
        self.translation_thread.join(timeout=2.0)
        self.tts_thread.join(timeout=2.0)

        if self.pa:
            self.pa.terminate()

        logger.info("System shut down.")

def main():
    parser = argparse.ArgumentParser(description='Real-time multilingual speech translation system.')
    parser.add_argument('--stt_model', type=str, default='small',
                       choices=['tiny', 'base', 'small', 'medium'],
                       help='Whisper ASR model size')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'cuda', 'auto'],
                       help='Inference device for Whisper')
    parser.add_argument('--tts_engine', type=str, default='espeak',
                       choices=['espeak', 'piper'],
                       help='TTS engine to use (espeak is fastest)')

    args = parser.parse_args()

    # 检查 TTS 引擎
    if args.tts_engine == "piper":
        if not check_piper_tts():
            logger.error("Piper TTS is not installed. Please install it first.")
            logger.error("You can install it by following the instructions at: https://github.com/rhasspy/piper")
            return
        
        # 扫描可用的Piper TTS模型
        logger.info(f"Scanning for available Piper TTS models in {PIPER_VOICES_PATH}...")
        available_models = scan_piper_voices()
        if not available_models:
            logger.error("No Piper TTS models found. Please check the piper-voices directory.")
            return
        
        logger.info(f"Found models for languages: {list(available_models.keys())}")
        for lang, models in available_models.items():
            logger.info(f"  {lang}: {len(models)} models")
    else:
        # 检查espeak-ng
        try:
            subprocess.run(['espeak-ng', '--version'], capture_output=True, check=True)
            logger.info("Using espeak-ng for TTS (fastest option)")
        except (FileNotFoundError, subprocess.CalledProcessError):
            logger.error("espeak-ng is not installed. Please install it with:")
            logger.error("  Ubuntu/Debian: sudo apt-get install espeak-ng")
            logger.error("  CentOS/RHEL: sudo yum install espeak-ng")
            logger.error("  Or use --tts_engine piper")
            return

    source_lang_code, target_lang_code = get_user_language_choice()
    
    if args.tts_engine == "piper" or target_lang_code in LANGUAGES_REQUIRING_PIPER:
        # 对于需要Piper的语言，强制检查Piper
        if not check_piper_tts():
            logger.error(f"{target_lang_code} requires Piper TTS but it's not installed.")
            return
        
        if 'available_models' not in locals():
            logger.info(f"Scanning for available Piper TTS models in {PIPER_VOICES_PATH}...")
            available_models = scan_piper_voices()
            if not available_models:
                logger.error("No Piper TTS models found. Please check the piper-voices directory.")
                return
        
        tts_engine, selected_tts_model = get_user_tts_choice(target_lang_code, available_models)
    else:
        tts_engine = "espeak"
        selected_tts_model = None

    model_dict = load_language_model_dict()
    selected_model = select_model_for_pair(source_lang_code, target_lang_code, model_dict)

    processor = None
    try:
        processor = SpeechProcessor(
            stt_model_size=args.stt_model,
            device=args.device,
            source_lang=source_lang_code,
            target_lang=target_lang_code,
            selected_translation_model=selected_model,
            tts_engine=tts_engine,
            selected_tts_model=selected_tts_model
        )

        logger.info(f"System initialized. Listening for speech in '{source_lang_code if source_lang_code != AUTO_DETECT_CODE else 'auto'}', translating into: {target_lang_code}")
        logger.info("Press Ctrl+C to stop the program.")

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal.")
    except Exception as e:
         logger.critical(f"Critical error in main execution: {e}", exc_info=True)
    finally:
        if processor:
            processor.shutdown()

if __name__ == "__main__":
    main()