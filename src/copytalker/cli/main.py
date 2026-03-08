"""
Command-line interface for CopyTalker.

Uses lazy imports to minimize startup time and allow basic commands
to work without all dependencies installed.
"""

import argparse
import logging
import sys
from typing import Optional, List, TYPE_CHECKING

from copytalker import __version__
from copytalker.core.constants import (
    SUPPORTED_LANGUAGES,
    get_available_voices,
    get_language_name,
    AUTO_DETECT_CODE,
)

if TYPE_CHECKING:
    from copytalker.core.config import AppConfig
    from copytalker.core.pipeline import TranslationPipeline
    from copytalker.utils.model_cache import ModelCache

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="copytalker",
        description="CopyTalker - Real-time multilingual speech-to-speech translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  copytalker translate --target zh
  copytalker translate --source en --target zh --voice af_heart
  copytalker list-voices --language zh
  copytalker download-models --whisper small
  copytalker --gui

For more information, visit: https://github.com/cycleuser/CopyTalker
        """,
    )
    
    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"CopyTalker {__version__}",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON",
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress non-essential output",
    )
    
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch GUI interface",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Translate command
    translate_parser = subparsers.add_parser(
        "translate",
        help="Start real-time translation",
    )
    translate_parser.add_argument(
        "-s", "--source",
        type=str,
        default=AUTO_DETECT_CODE,
        help=f"Source language ({AUTO_DETECT_CODE} for auto-detect)",
    )
    translate_parser.add_argument(
        "-t", "--target",
        type=str,
        required=True,
        help="Target language (e.g., en, zh, ja, ko)",
    )
    translate_parser.add_argument(
        "-v", "--voice",
        type=str,
        help="TTS voice name",
    )
    translate_parser.add_argument(
        "--tts-engine",
        type=str,
        choices=["kokoro", "edge-tts", "pyttsx3", "indextts", "fish-speech", "auto"],
        default="auto",
        help="TTS engine to use",
    )
    translate_parser.add_argument(
        "--whisper-model",
        type=str,
        choices=["tiny", "base", "small", "medium", "large"],
        default="small",
        help="Whisper model size",
    )
    translate_parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Compute device",
    )
    translate_parser.add_argument(
        "--reference-audio",
        type=str,
        help="Reference audio file for voice cloning (IndexTTS/Fish-Speech)",
    )
    translate_parser.add_argument(
        "--emotion",
        type=str,
        help="Emotion for TTS (IndexTTS: happy/sad/angry/etc, Fish-Speech: emotion tag)",
    )
    
    # List voices command
    list_voices_parser = subparsers.add_parser(
        "list-voices",
        help="List available TTS voices",
    )
    list_voices_parser.add_argument(
        "-l", "--language",
        type=str,
        help="Filter by language",
    )
    list_voices_parser.add_argument(
        "--engine",
        type=str,
        choices=["kokoro", "edge-tts", "indextts", "fish-speech"],
        default="kokoro",
        help="TTS engine",
    )
    
    # List languages command
    subparsers.add_parser(
        "list-languages",
        help="List supported languages",
    )
    
    # List emotions command
    list_emotions_parser = subparsers.add_parser(
        "list-emotions",
        help="List available emotion tags for TTS engines",
    )
    list_emotions_parser.add_argument(
        "--engine",
        type=str,
        choices=["indextts", "fish-speech"],
        default="fish-speech",
        help="TTS engine to query emotions for",
    )
    
    # TTS synthesize command
    synth_parser = subparsers.add_parser(
        "synthesize",
        help="Synthesize text to speech audio file",
    )
    synth_parser.add_argument(
        "text",
        type=str,
        help="Text to synthesize",
    )
    synth_parser.add_argument(
        "-o", "--output",
        type=str,
        default="output.wav",
        help="Output WAV file path",
    )
    synth_parser.add_argument(
        "-l", "--language",
        type=str,
        default="en",
        help="Target language code",
    )
    synth_parser.add_argument(
        "--engine",
        type=str,
        choices=["kokoro", "edge-tts", "pyttsx3", "indextts", "fish-speech", "auto"],
        default="auto",
        help="TTS engine to use",
    )
    synth_parser.add_argument(
        "-v", "--voice",
        type=str,
        help="Voice name or reference audio path",
    )
    synth_parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed multiplier (0.5-2.0)",
    )
    synth_parser.add_argument(
        "--reference-audio",
        type=str,
        help="Reference audio for voice cloning (IndexTTS/Fish-Speech)",
    )
    synth_parser.add_argument(
        "--emotion",
        type=str,
        help="Emotion tag for synthesis",
    )
    synth_parser.add_argument(
        "--emotion-audio",
        type=str,
        help="Emotion reference audio (IndexTTS v2)",
    )
    synth_parser.add_argument(
        "--target-duration",
        type=float,
        help="Target audio duration in seconds (IndexTTS v2)",
    )
    
    # Clone voice command
    clone_parser = subparsers.add_parser(
        "clone-voice",
        help="Clone a voice from reference audio and synthesize text",
    )
    clone_parser.add_argument(
        "text",
        type=str,
        help="Text to speak in the cloned voice",
    )
    clone_parser.add_argument(
        "-r", "--reference-audio",
        type=str,
        required=True,
        help="Path to reference audio file (5-30 seconds)",
    )
    clone_parser.add_argument(
        "-o", "--output",
        type=str,
        default="cloned_output.wav",
        help="Output WAV file path",
    )
    clone_parser.add_argument(
        "--engine",
        type=str,
        choices=["indextts", "fish-speech"],
        default="indextts",
        help="Voice cloning engine",
    )
    clone_parser.add_argument(
        "-l", "--language",
        type=str,
        default="en",
        help="Target language code",
    )
    clone_parser.add_argument(
        "--emotion",
        type=str,
        help="Optional emotion to apply",
    )
    
    # Download models command
    download_parser = subparsers.add_parser(
        "download-models",
        help="Pre-download models",
    )
    download_parser.add_argument(
        "--whisper",
        type=str,
        choices=["tiny", "base", "small", "medium", "large"],
        help="Download Whisper model",
    )
    download_parser.add_argument(
        "--translation",
        type=str,
        help="Download translation model (e.g., Helsinki-NLP/opus-mt-en-zh)",
    )
    download_parser.add_argument(
        "--kokoro",
        action="store_true",
        help="Download Kokoro TTS model",
    )
    download_parser.add_argument(
        "--indextts",
        action="store_true",
        help="Download IndexTTS v2 model",
    )
    download_parser.add_argument(
        "--fish-speech",
        action="store_true",
        help="Download Fish-Speech model",
    )
    download_parser.add_argument(
        "--all",
        action="store_true",
        help="Download all recommended models",
    )
    
    # Cache info command
    cache_parser = subparsers.add_parser(
        "cache",
        help="Manage model cache",
    )
    cache_parser.add_argument(
        "--info",
        action="store_true",
        help="Show cache information",
    )
    cache_parser.add_argument(
        "--clear",
        type=str,
        nargs="?",
        const="all",
        choices=["all", "whisper", "translation", "tts"],
        help="Clear cache (optionally specify type)",
    )
    
    return parser


def cmd_translate(args: argparse.Namespace) -> int:
    """Run the translation pipeline."""
    import time
    from copytalker.core.config import AppConfig
    from copytalker.core.pipeline import TranslationPipeline
    
    config = AppConfig()
    
    # Apply command-line arguments
    config.stt.model_size = args.whisper_model
    config.stt.language = args.source
    config.translation.source_lang = args.source
    config.translation.target_lang = args.target
    config.tts.engine = args.tts_engine
    config.tts.language = args.target
    
    if args.voice:
        config.tts.voice = args.voice
    
    # IndexTTS/Fish-Speech specific settings
    if hasattr(args, 'reference_audio') and args.reference_audio:
        config.tts.indextts_reference_audio = args.reference_audio
        config.tts.fish_speech_reference_audio = args.reference_audio
        # Also use as voice for the engine
        if not args.voice:
            config.tts.voice = args.reference_audio
    
    if hasattr(args, 'emotion') and args.emotion:
        config.tts.indextts_emotion = args.emotion
        config.tts.fish_speech_emotion = args.emotion
    
    if args.device != "auto":
        config.stt.device = args.device
        config.translation.device = args.device
        config.tts.device = args.device
    
    source_name = "Auto-detect" if args.source == AUTO_DETECT_CODE else get_language_name(args.source)
    target_name = get_language_name(args.target)
    
    print(f"\nCopyTalker - Real-time Translation")
    print(f"{'=' * 40}")
    print(f"Source: {source_name}")
    print(f"Target: {target_name}")
    print(f"TTS Engine: {args.tts_engine}")
    print(f"Whisper Model: {args.whisper_model}")
    print(f"{'=' * 40}\n")
    
    def on_transcription(event):
        print(f"[Heard] {event.data.text} ({event.data.language})")
    
    def on_translation(event):
        print(f"[Translated] {event.data.translated_text}")
    
    def on_error(event):
        print(f"[Error] {event.data}", file=sys.stderr)
    
    try:
        pipeline = TranslationPipeline(config)
        pipeline.register_callback("transcription", on_transcription)
        pipeline.register_callback("translation", on_translation)
        pipeline.register_callback("error", on_error)
        
        print("Starting translation pipeline...")
        print("Press Ctrl+C to stop.\n")
        
        pipeline.start()
        
        # Keep running until interrupted
        while pipeline.is_running:
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    finally:
        if 'pipeline' in locals():
            pipeline.stop()
    
    print("Goodbye!")
    return 0


def cmd_list_voices(args: argparse.Namespace) -> int:
    """List available TTS voices."""
    engine = args.engine
    
    if args.language:
        languages = [args.language]
    else:
        languages = [code for code, _ in SUPPORTED_LANGUAGES]
    
    print(f"\nAvailable voices ({engine}):")
    print(f"{'=' * 40}")
    
    for lang in languages:
        voices = get_available_voices(lang, engine)
        if voices:
            print(f"\n{get_language_name(lang)} ({lang}):")
            for voice in voices:
                print(f"  - {voice}")
    
    return 0


def cmd_list_languages(args: argparse.Namespace) -> int:
    """List supported languages."""
    print("\nSupported Languages:")
    print(f"{'=' * 40}")
    
    for code, name in SUPPORTED_LANGUAGES:
        print(f"  {code:5} - {name}")
    
    return 0


def cmd_list_emotions(args: argparse.Namespace) -> int:
    """List available emotion tags for TTS engines."""
    engine = args.engine
    
    print(f"\nAvailable emotions ({engine}):")
    print(f"{'=' * 40}")
    
    if engine == "indextts":
        from copytalker.core.constants import INDEXTTS_EMOTIONS
        print("\nIndexTTS v2 emotions (8 basic emotions):")
        for emotion in INDEXTTS_EMOTIONS:
            print(f"  - {emotion}")
        print("\nControl methods:")
        print("  1. --emotion <name>            Emotion by name")
        print("  2. --emotion-audio <path>      Emotion from reference audio")
        print("  3. Text-based inference         Automatic from text content")
    elif engine == "fish-speech":
        from copytalker.core.constants import FISH_SPEECH_EMOTION_TAGS
        print("\nFish-Speech emotion/expression tags (50+):")
        for i, tag in enumerate(FISH_SPEECH_EMOTION_TAGS):
            print(f"  - {tag}", end="")
            if (i + 1) % 4 == 0:
                print()
            else:
                print("\t", end="")
        print()
        print("\nUsage: Embed tags in text, e.g. '(happy) Hello world!'")
    
    return 0


def cmd_synthesize(args: argparse.Namespace) -> int:
    """Synthesize text to speech."""
    from copytalker.api import tts_synthesize
    
    result = tts_synthesize(
        text=args.text,
        language=args.language,
        engine=args.engine,
        voice=args.voice,
        speed=args.speed,
        output_path=args.output,
        emotion=getattr(args, 'emotion', None),
        emotion_audio=getattr(args, 'emotion_audio', None),
        target_duration=getattr(args, 'target_duration', None),
        reference_audio=getattr(args, 'reference_audio', None),
    )
    
    if result.success:
        print(f"\nSynthesis complete!")
        print(f"  Output: {result.data['output_path']}")
        print(f"  Duration: {result.data['duration_seconds']:.2f}s")
        print(f"  Sample rate: {result.data['sample_rate']} Hz")
    else:
        print(f"\nSynthesis failed: {result.error}", file=sys.stderr)
        return 1
    
    return 0


def cmd_clone_voice(args: argparse.Namespace) -> int:
    """Clone a voice and synthesize text."""
    from copytalker.api import clone_voice
    
    result = clone_voice(
        text=args.text,
        reference_audio=args.reference_audio,
        engine=args.engine,
        language=args.language,
        output_path=args.output,
        emotion=getattr(args, 'emotion', None),
    )
    
    if result.success:
        print(f"\nVoice cloning complete!")
        print(f"  Output: {result.data['output_path']}")
        print(f"  Duration: {result.data['duration_seconds']:.2f}s")
        print(f"  Engine: {args.engine}")
    else:
        print(f"\nVoice cloning failed: {result.error}", file=sys.stderr)
        return 1
    
    return 0


def cmd_download_models(args: argparse.Namespace) -> int:
    """Download models."""
    from copytalker.utils.model_cache import ModelCache
    
    cache = ModelCache()
    
    if args.all:
        print("Downloading all recommended models...")
        args.whisper = "small"
        args.kokoro = True
        args.indextts = True
        args.fish_speech = True
    
    if args.whisper:
        print(f"\nDownloading Whisper {args.whisper} model...")
        try:
            cache.download_whisper_model(args.whisper)
            print(f"Whisper {args.whisper} model ready!")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    if args.translation:
        print(f"\nDownloading translation model: {args.translation}")
        try:
            cache.download_translation_model(args.translation)
            print(f"Translation model ready!")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    if args.kokoro:
        print("\nDownloading Kokoro TTS model...")
        try:
            cache.download_kokoro_model()
            print("Kokoro TTS model ready!")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    if hasattr(args, 'indextts') and args.indextts:
        print("\nDownloading IndexTTS v2 model...")
        try:
            cache.download_indextts_model(version="v2")
            print("IndexTTS v2 model ready!")
        except Exception as e:
            print(f"Error downloading IndexTTS: {e}", file=sys.stderr)
            return 1
    
    if hasattr(args, 'fish_speech') and args.fish_speech:
        print("\nDownloading Fish-Speech model...")
        try:
            cache.download_fish_speech_model()
            print("Fish-Speech model ready!")
        except Exception as e:
            print(f"Error downloading Fish-Speech: {e}", file=sys.stderr)
            return 1
    
    if not any([args.whisper, args.translation, args.kokoro,
                getattr(args, 'indextts', False),
                getattr(args, 'fish_speech', False),
                args.all]):
        print("No models specified. Use --help for options.")
        return 1
    
    print("\nDone!")
    return 0


def cmd_cache(args: argparse.Namespace) -> int:
    """Manage model cache."""
    from copytalker.utils.model_cache import ModelCache, format_size
    
    cache = ModelCache()
    
    if args.clear:
        if args.clear == "all":
            print("Clearing all cached models...")
            cache.clear_cache()
        else:
            print(f"Clearing {args.clear} models...")
            cache.clear_cache(args.clear)
        print("Cache cleared!")
        return 0
    
    # Default: show info
    print("\nModel Cache Information:")
    print(f"{'=' * 40}")
    print(f"Cache directory: {cache.cache_dir}")
    print(f"Total size: {format_size(cache.get_cache_size())}")
    
    cached = cache.get_cached_models()
    for model_type, models in cached.items():
        if models:
            print(f"\n{model_type.title()} models:")
            for model in models:
                print(f"  - {model}")
    
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # Setup logging (lazy import)
    from copytalker.core.config import AppConfig, setup_logging
    setup_logging(AppConfig(debug=args.debug))
    
    # Handle --gui flag
    if args.gui:
        try:
            from copytalker.gui.main_window import main as gui_main
            return gui_main()
        except ImportError as e:
            print(f"GUI not available: {e}", file=sys.stderr)
            return 1
    
    # Handle commands
    if args.command == "translate":
        return cmd_translate(args)
    elif args.command == "list-voices":
        return cmd_list_voices(args)
    elif args.command == "list-languages":
        return cmd_list_languages(args)
    elif args.command == "list-emotions":
        return cmd_list_emotions(args)
    elif args.command == "synthesize":
        return cmd_synthesize(args)
    elif args.command == "clone-voice":
        return cmd_clone_voice(args)
    elif args.command == "download-models":
        return cmd_download_models(args)
    elif args.command == "cache":
        return cmd_cache(args)
    else:
        # No command - show interactive menu or help
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
