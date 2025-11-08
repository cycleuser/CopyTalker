import threading
import queue
import time
import numpy as np
import pyaudio
import webrtcvad
from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer
import torch
import pyttsx3  # Move import to the top for clarity
import langdetect # Import directly
import argparse
import logging
from collections import deque
import re # For language code normalization

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s') # Add thread name for better tracing
logger = logging.getLogger(__name__)

class SpeechProcessor:
    def __init__(self, stt_model_size="small", device="auto", target_lang="en"):
        """
        初始化语音处理系统
        :param stt_model_size: Whisper模型大小 (tiny, base, small, medium)
        :param device: 推理设备 (cpu, cuda, auto)
        :param target_lang: 目标翻译语言代码 (例如: en, zh, ja, ko, fr, es, de)
        """
        self.audio_buffer = deque(maxlen=10)  # 存储最近10个音频片段
        self.vad = webrtcvad.Vad(3)  # VAD aggressiveness 0-3
        self.sample_rate = 16000
        self.frame_duration = 30  # ms
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)
        self.channels = 1
        
        # 初始化流式音频队列
        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()
        self.translation_queue = queue.Queue()
        
        # 加载Whisper模型
        logger.info(f"Loading Whisper STT model ({stt_model_size}) on {device}...")
        self.stt_model = WhisperModel(stt_model_size, device=device, compute_type="float16" if device != "cpu" else "int8") # Adjust compute type based on device

        # --- Improved Language Handling ---
        # Define supported languages for translation models explicitly
        # Map common detected codes or desired codes to the ones used by Marian models
        self.supported_translation_langs = {
            'zh': 'zh', 'zh-cn': 'zh', 'zh-tw': 'zh', 'cmn': 'zh', # Normalize Chinese variants
            'en': 'en',
            'es': 'es', 'spa': 'es',
            'fr': 'fr', 'fra': 'fr',
            'de': 'de', 'deu': 'de',
            'ja': 'ja', 'jpn': 'ja',
            'ko': 'ko', 'kor': 'ko',
            'ru': 'ru', 'rus': 'ru',
            'ar': 'ar', 'ara': 'ar'
            # Add more mappings as needed for your use cases
        }
        self.target_lang = target_lang
        logger.info(f"Target translation language set to: {self.target_lang}")

        # 翻译模型缓存
        self.translation_models = {}
        self.translation_tokenizers = {}
        
        # 初始化TTS引擎
        logger.info("Initializing TTS engine (pyttsx3)...")
        self.tts_engine = pyttsx3.init()
        # Set properties *before* you start speaking
        self.tts_engine.setProperty('rate', 180)  # Speed percent (can go over 100)
        self.tts_engine.setProperty('volume', 0.9) # Volume 0-1
        # Attempt to set English voice early (best effort)
        self._set_tts_voice(self.target_lang)

        # 启动处理线程
        self._start_threads()
        
    def _start_threads(self):
        """启动处理线程"""
        self.stop_threads = False
        
        # 音频采集线程
        self.audio_thread = threading.Thread(target=self._audio_capture, name="AudioCaptureThread")
        self.audio_thread.daemon = True
        
        # 语音识别线程
        self.stt_thread = threading.Thread(target=self._speech_to_text, name="STTThread")
        self.stt_thread.daemon = True
        
        # 翻译线程
        self.translation_thread = threading.Thread(target=self._translate_text, name="TranslationThread")
        self.translation_thread.daemon = True
        
        # 语音合成线程
        self.tts_thread = threading.Thread(target=self._text_to_speech, name="TTSThread")
        self.tts_thread.daemon = True
        
        # 启动所有线程
        self.audio_thread.start()
        self.stt_thread.start()
        self.translation_thread.start()
        self.tts_thread.start()
        logger.info("All processing threads started.")
        
    def _audio_capture(self):
        """捕获音频输入并进行VAD处理"""
        p = pyaudio.PyAudio()
        
        try:
            # 打开音频流
            stream = p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.frame_size
            )
        except Exception as e:
            logger.error(f"Failed to open PyAudio stream: {e}")
            self.stop_threads = True # Signal other threads to stop if audio capture fails
            return
            
        
        logger.info("Started listening for audio input...")
        
        voice_buffer = []
        is_speaking = False
        last_voice_time = time.time()
        
        while not self.stop_threads:
            try:
                # 读取音频帧
                frame = stream.read(self.frame_size, exception_on_overflow=False)
                audio_data = np.frombuffer(frame, dtype=np.int16)
                
                # VAD检测
                is_voice = self.vad.is_speech(frame, self.sample_rate)
                
                current_time = time.time()
                if is_voice:
                    voice_buffer.append(audio_data)
                    is_speaking = True
                    last_voice_time = current_time
                else:
                    # 检查是否结束说话 (silence duration threshold)
                    silence_duration = current_time - last_voice_time
                    if is_speaking and silence_duration > 0.8: # End of speech segment
                        if len(voice_buffer) > 5:  # Ensure enough voice data collected
                            audio_segment = np.concatenate(voice_buffer)
                            # Normalize audio to float32 range [-1, 1]
                            audio_segment = audio_segment.astype(np.float32) / 32768.0
                            # Add to processing queue
                            self.audio_queue.put(audio_segment.copy())
                            logger.debug(f"Queued audio segment of length {len(audio_segment)}")
                        
                        # Reset buffer and state for next utterance
                        voice_buffer = []
                        is_speaking = False
                    elif is_speaking:
                         # Append short silence within speech
                        voice_buffer.append(audio_data)
                
                # 保存最近的音频用于持续监听 (optional, not currently used elsewhere)
                self.audio_buffer.append(audio_data)
                    
            except Exception as e:
                if not self.stop_threads: # Only log if not intentionally stopping
                     logger.error(f"Audio capture error: {e}")
                break
        
        # 清理 PyAudio resources
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
        """语音转文字"""
        logger.info("STT thread started.")
        while not self.stop_threads:
            try:
                # 等待音频数据
                audio = self.audio_queue.get(timeout=1.0) # Increased timeout slightly
                
                logger.info("Performing speech-to-text...")
                start_time = time.time()
                
                # 使用Whisper进行语音识别 (Force language detection or specify if known)
                # Let Whisper auto-detect language for maximum flexibility
                segments, info = self.stt_model.transcribe(
                    audio, 
                    beam_size=5,
                    language=None, # Auto-detect language
                    condition_on_previous_text=False # Good practice for streaming
                )
                
                # 获取识别结果
                text = "".join(segment.text for segment in segments).strip()
                
                # --- Improved Language Detection ---
                detected_lang_full = info.language if info else None
                detected_lang_confidence = info.language_probability if info else 0
                logger.debug(f"Whisper detected language: {detected_lang_full} (confidence: {detected_lang_confidence:.2f})")

                # Fallback or refine language detection using langdetect if Whisper's confidence is low or result seems off
                # Or always double-check complex scenarios. Here, we primarily trust Whisper.
                final_detected_lang = detected_lang_full

                # Normalize the detected language code for our translation mapping
                normalized_lang = self.supported_translation_langs.get(final_detected_lang.lower(), 'en') # Default to 'en' if unknown

                process_time = time.time() - start_time
                if text:
                    logger.info(f"Speech recognized: '{text}' (Detected Lang: {normalized_lang}, Time: {process_time:.2f}s)")
                    # 将识别结果放入队列 (包含原文和标准化后的语言代码)
                    self.text_queue.put({
                        "text": text,
                        "source_lang": normalized_lang # Use the normalized code
                    })
                else:
                     logger.debug(f"No speech recognized in segment (Time: {process_time:.2f}s)")

            except queue.Empty:
                continue # No audio ready, loop again
            except Exception as e:
                logger.error(f"Speech-to-text error: {e}", exc_info=True) # Log full traceback
    
    def _load_translation_model(self, source_lang, target_lang):
        """加载翻译模型，如果未缓存则加载到内存"""
        # Normalize source and target langs for model lookup
        norm_src = self.supported_translation_langs.get(source_lang.lower(), source_lang.lower())
        norm_tgt = self.supported_translation_langs.get(target_lang.lower(), target_lang.lower())

        model_key = f"{norm_src}-{norm_tgt}"
        
        if model_key not in self.translation_models:
            logger.info(f"Loading translation model: {model_key}...")
            
            try:
                # 使用Helsinki-NLP的Marian模型
                model_name = f"Helsinki-NLP/opus-mt-{norm_src}-{norm_tgt}"
                logger.debug(f"Attempting to load tokenizer from: {model_name}")
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                logger.debug(f"Attempting to load model from: {model_name}")
                model = MarianMTModel.from_pretrained(model_name)
                
                # 使用GPU如果可用 (and not explicitly cpu)
                device = "cuda" if torch.cuda.is_available() and str(model.device) != 'cpu' else "cpu"
                model = model.to(device)
                logger.info(f"Translation model {model_key} loaded successfully on {device}.")
                
                self.translation_models[model_key] = model
                self.translation_tokenizers[model_key] = tokenizer
                
                
            except Exception as e:
                 # Provide more specific error context
                 logger.error(f"Failed to load translation model {model_key} ({model_name}): {e}", exc_info=True)
                 # Do not attempt fallback here as it complicates logic; let calling function handle failure
                 raise e # Re-raise to signal failure to caller
        
        return self.translation_tokenizers[model_key], self.translation_models[model_key]

    def _translate_text(self):
        """翻译文本"""
        logger.info("Translation thread started.")
        while not self.stop_threads:
            try:
                # 等待文本数据
                data = self.text_queue.get(timeout=1.0) # Increased timeout
                original_text = data["text"].strip()
                source_lang = data["source_lang"]

                # Validate source language against our mapping
                if source_lang not in self.supported_translation_langs.values():
                     logger.warning(f"Source language '{source_lang}' not in supported list for translation. Defaulting to 'en'.")
                     source_lang = 'en'

                # If source language and target language are the same, no translation needed
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
                
                # 获取或加载翻译模型
                try:
                    tokenizer, model = self._load_translation_model(source_lang, self.target_lang)
                except Exception as e:
                    logger.error(f"Skipping translation due to model loading failure: {e}")
                    continue # Skip this translation task
                
                # Prepare input - Handle potential truncation warnings by specifying max_length if needed
                # Marian models usually have a limit (e.g., 512 tokens)
                inputs = tokenizer(original_text, return_tensors="pt", padding=True, truncation=True, max_length=400) # Add truncation

                # Ensure tensors are on the correct device
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                # Model inference
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=400) # Limit output length
                
                # Decode output
                translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Basic post-processing: remove extra whitespace that might occur
                translated_text = re.sub(r'\s+', ' ', translated_text).strip()
                
                process_time = time.time() - start_time
                logger.info(f"Translation complete: '{translated_text}' (Time: {process_time:.2f}s)")
                
                # 将翻译结果放入队列
                self.translation_queue.put({
                    "original": original_text,
                    "translated": translated_text,
                    "source_lang": source_lang,
                    "target_lang": self.target_lang
                })
                
            except queue.Empty:
                continue # No text ready, loop again
            except Exception as e:
                logger.error(f"Translation error: {e}", exc_info=True)

    def _set_tts_voice(self, target_lang_code):
        """Attempt to configure the TTS engine for the target language."""
        try:
             voices = self.tts_engine.getProperty('voices')
             target_lower = target_lang_code.lower()
             logger.debug(f"Available TTS voices: {[v.name for v in voices]}")
             for voice in voices:
                  # Check various attributes for language match
                  # This heuristic can vary by OS and TTS engine backend
                  if (target_lower in getattr(voice, 'languages', [''])[0].lower() or
                      target_lower in voice.name.lower() or
                      target_lang_code.upper() in voice.name): # Sometimes codes like 'en_US' are in the name
                       logger.info(f"Setting TTS voice to: {voice.name}")
                       self.tts_engine.setProperty('voice', voice.id)
                       return
             logger.warning(f"TTS voice for language '{target_lang_code}' not found. Using default voice.")
        except Exception as e:
             logger.warning(f"Could not set TTS voice for '{target_lang_code}': {e}")


    def _text_to_speech(self):
        """文本转语音输出"""
        logger.info("TTS thread started.")
        while not self.stop_threads:
            try:
                # 等待翻译结果
                data = self.translation_queue.get(timeout=1.0) # Increased timeout
                translated_text = data["translated"].strip()
                target_lang = data["target_lang"] # Should be self.target_lang

                if not translated_text:
                    logger.debug("Skipping TTS for empty translated text.")
                    continue

                logger.info(f"Synthesizing speech for: '{translated_text}' ({target_lang})")
                start_time = time.time()

                # --- Improved TTS Voice Setting ---
                # While setting once at init helps, some engines might need it per utterance
                # depending on implementation details. It's safer to ensure it's set.
                # However, repeated calls might be redundant or slow. Uncomment below if issues arise.
                # self._set_tts_voice(target_lang)

                # --- Synthesize and Play ---
                # pyttsx3 is synchronous, runAndWait() blocks until finished.
                self.tts_engine.say(translated_text)
                self.tts_engine.runAndWait() # Blocks here until speech finishes

                process_time = time.time() - start_time
                logger.info(f"Speech synthesis completed (Time: {process_time:.2f}s)")

            except queue.Empty:
                continue # No translation ready, loop again
            except Exception as e:
                 # Catch exceptions during TTS to prevent thread crash
                 logger.error(f"Text-to-speech error: {e}", exc_info=True)


    def shutdown(self):
        """关闭系统"""
        logger.info("Shutting down the speech translation system...")
        self.stop_threads = True
        
        # Optionally wait briefly for threads to finish naturally
        # join() with timeout is good practice
        self.audio_thread.join(timeout=2.0)
        self.stt_thread.join(timeout=2.0)
        self.translation_thread.join(timeout=2.0)
        self.tts_thread.join(timeout=2.0)
        
        logger.info("System shut down.")

def main():
    parser = argparse.ArgumentParser(description='Real-time multilingual speech translation system.')
    parser.add_argument('--stt_model', type=str, default='small', 
                       choices=['tiny', 'base', 'small', 'medium'],
                       help='Whisper ASR model size')
    parser.add_argument('--target_lang', type=str, default='en',
                       help='Target language code for translation and TTS (e.g., en, zh, ja, ko, fr, es, de)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'cuda', 'auto'],
                       help='Inference device for Whisper')
    
    args = parser.parse_args()
    
    # Validate target language early
    if args.target_lang not in ['en', 'zh', 'es', 'fr', 'de', 'ja', 'ko', 'ru', 'ar']:
         logger.warning(f"Target language '{args.target_lang}' might not be fully supported by all components.")

    # 初始化系统 - Pass target_lang to constructor
    processor = None
    try:
        processor = SpeechProcessor(stt_model_size=args.stt_model, device=args.device, target_lang=args.target_lang)
        
        # Note: target_lang is now handled in the constructor
        # processor.set_target_language(args.target_lang) # Removed as it's done in __init__
        
        logger.info(f"System initialized. Listening for speech to translate into: {args.target_lang}")
        logger.info("Press Ctrl+C to stop the program.")
        
        # Keep the main thread alive
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