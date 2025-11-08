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
import asyncio
import edge_tts
import tempfile
import subprocess

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

# Edge TTS 语言代码映射
EDGE_TTS_LANG_MAP = {
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

AUTO_DETECT_CODE = "auto"

async def get_voices_for_language(lang_code):
    """获取指定语言的所有可用语音"""
    voices = await edge_tts.list_voices()
    filtered_voices = []
    for voice in voices:
        if voice['Locale'].startswith(EDGE_TTS_LANG_MAP.get(lang_code, lang_code)):
            filtered_voices.append(voice)
    return filtered_voices

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

def get_user_voice_choice(target_lang_code):
    """让用户选择TTS语音"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    voices = loop.run_until_complete(get_voices_for_language(target_lang_code))
    loop.close()
    
    if not voices:
        print(f"No voices found for language {target_lang_code}")
        return None
    
    print(f"\n--- Select TTS Voice for {target_lang_code} ---")
    for i, voice in enumerate(voices, 1):
        # 安全地获取语音信息，处理可能的键缺失
        name = voice.get('Name', 'Unknown')
        friendly_name = voice.get('FriendlyName', voice.get('LocalName', 'Unknown'))
        gender = voice.get('Gender', 'Unknown')
        locale = voice.get('Locale', 'Unknown')
        print(f"{i}. {name} ({friendly_name}) - {gender} - {locale}")
    
    while True:
        try:
            choice_str = input(f"\nPlease enter the number of your voice choice (1-{len(voices)}): ").strip()
            choice_idx = int(choice_str)
            if 1 <= choice_idx <= len(voices):
                selected_voice = voices[choice_idx - 1]
                print(f"Selected voice: {selected_voice.get('FriendlyName', selected_voice.get('LocalName', 'Unknown'))}\n")
                return selected_voice['Name']
            else:
                print(f"Invalid number. Please enter a number between 1 and {len(voices)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

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
    def __init__(self, stt_model_size="small", device="auto", source_lang=AUTO_DETECT_CODE, target_lang="en", selected_translation_model=None, selected_voice=None):
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
        self.selected_voice = selected_voice
        logger.info(f"Target translation language set to: {self.target_lang}")
        logger.info(f"Source language for STT set to: {self.source_lang if self.source_lang != AUTO_DETECT_CODE else 'Automatic Detection'}")
        logger.info(f"Selected TTS voice: {self.selected_voice}")

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

    async def _generate_tts_audio(self, text, voice):
        """使用 edge-tts 生成音频"""
        communicate = edge_tts.Communicate(text, voice)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            await communicate.save(temp_file.name)
            return temp_file.name

    def _play_audio(self, audio_file_path):
        """播放音频文件"""
        try:
            cmd = ['ffprobe', '-v', 'error', '-show_entries', 'stream=sample_rate', '-of', 'default=noprint_wrappers=1:nokey=1', audio_file_path]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            sample_rate = int(result.stdout.strip()) if result.stdout.strip() else 24000
            
            cmd = ['ffmpeg', '-i', audio_file_path, '-f', 's16le', '-acodec', 'pcm_s16le', '-ac', '1', '-ar', str(sample_rate), '-']
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if result.returncode != 0:
                logger.error(f"Error reading audio file: {result.stderr.decode()}")
                return
                
            audio_data = np.frombuffer(result.stdout, dtype=np.int16)
            
            stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                output=True
            )
            
            stream.write(audio_data.tobytes())
            stream.stop_stream()
            stream.close()
            
            os.unlink(audio_file_path)
            
        except Exception as e:
            logger.error(f"Error during audio playback: {e}")
            try:
                os.unlink(audio_file_path)
            except:
                pass

    def _text_to_speech(self):
        logger.info("TTS thread started.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while not self.stop_threads:
            try:
                data = self.translation_queue.get(timeout=1.0)
                translated_text = data["translated"].strip()
                target_lang = data["target_lang"]

                if not translated_text:
                    logger.debug("Skipping TTS for empty translated text.")
                    continue

                logger.info(f"Synthesizing speech for: '{translated_text}' ({target_lang})")
                start_time = time.time()

                voice = self.selected_voice
                
                try:
                    audio_file_path = loop.run_until_complete(self._generate_tts_audio(translated_text, voice))
                    
                    self.tts_playing_lock.acquire()
                    try:
                        self._play_audio(audio_file_path)
                    finally:
                        self.tts_playing_lock.release()

                except Exception as e:
                    logger.error(f"Error during TTS synthesis or playback: {e}")
                    continue

                process_time = time.time() - start_time
                logger.info(f"Speech synthesis and playback completed (Time: {process_time:.2f}s)")

            except queue.Empty:
                continue
            except Exception as e:
                 logger.error(f"Text-to-speech error: {e}", exc_info=True)
        
        loop.close()

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

    args = parser.parse_args()

    source_lang_code, target_lang_code = get_user_language_choice()
    selected_voice = get_user_voice_choice(target_lang_code)

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
            selected_voice=selected_voice
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