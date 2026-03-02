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

For more information, visit: https://github.com/EasyCam/CopyTalker
        """,
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"CopyTalker {__version__}",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
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
        choices=["kokoro", "edge-tts", "pyttsx3", "auto"],
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
        choices=["kokoro", "edge-tts"],
        default="kokoro",
        help="TTS engine",
    )
    
    # List languages command
    subparsers.add_parser(
        "list-languages",
        help="List supported languages",
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


def cmd_download_models(args: argparse.Namespace) -> int:
    """Download models."""
    from copytalker.utils.model_cache import ModelCache
    
    cache = ModelCache()
    
    if args.all:
        print("Downloading all recommended models...")
        args.whisper = "small"
        args.kokoro = True
    
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
    
    if not any([args.whisper, args.translation, args.kokoro, args.all]):
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
