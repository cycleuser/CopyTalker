"""Model download controller extracted from main_window.py."""

from __future__ import annotations

import logging
import queue
import threading

logger = logging.getLogger(__name__)


class ModelDownloadController:
    """Manages model downloads for Whisper, TTS, and translation models."""

    def __init__(self, event_queue: queue.Queue) -> None:
        self._event_queue = event_queue

    def download(self, what: str) -> None:
        """Download a model in a background thread.

        Args:
            what: One of "indextts", "fish_speech", "kokoro", "whisper",
                  "translation", or "all".
        """
        self._event_queue.put(("dl_progress", f"Downloading {what}..."))
        threading.Thread(
            target=self._download_thread,
            args=(what,),
            daemon=True,
        ).start()

    def download_translation_for_langs(self, langs: list[str]) -> None:
        """Download translation models for specific languages."""
        self._event_queue.put(("dl_progress", f"Preparing to download {len(langs)} language(s)..."))
        threading.Thread(
            target=self._download_langs_thread,
            args=(langs,),
            daemon=True,
        ).start()

    def refresh_cache_info(self) -> str:
        """Return cache info string."""
        try:
            from copytalker.utils.model_cache import ModelCache, format_size

            cache = ModelCache()
            size = format_size(cache.get_cache_size())
            cached = cache.get_cached_models()
            lines = [f"Cache dir: {cache.cache_dir}", f"Total size: {size}"]
            for cat, items in cached.items():
                if items:
                    lines.append(f"  {cat}: {', '.join(items)}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _download_thread(self, what: str) -> None:
        try:
            from copytalker.utils.model_cache import ModelCache

            cache = ModelCache()
            failed: list[str] = []
            targets = (
                [what]
                if what != "all"
                else ["indextts", "fish_speech", "kokoro", "whisper", "translation"]
            )

            for target in targets:
                self._event_queue.put(("dl_progress", f"Downloading {target}..."))
                try:
                    if target == "indextts":
                        cache.download_indextts_model()
                    elif target == "fish_speech":
                        cache.download_fish_speech_model()
                    elif target == "kokoro":
                        cache.download_kokoro_model()
                    elif target == "whisper":
                        cache.download_whisper_model("small")
                    elif target == "translation":
                        from copytalker.core.constants import DEFAULT_TRANSLATION_MODELS

                        for key, models in DEFAULT_TRANSLATION_MODELS.items():
                            if key == "multilingual":
                                continue
                            for m in models:
                                self._event_queue.put(("dl_progress", f"Downloading {m}..."))
                                try:
                                    cache.download_translation_model(m)
                                except Exception as e2:
                                    failed.append(f"{m}: {e2}")
                        self._event_queue.put(("dl_progress", "Downloading NLLB..."))
                        try:
                            cache.download_translation_model("facebook/nllb-200-distilled-600M")
                        except Exception as e2:
                            failed.append(f"nllb: {e2}")
                except Exception as e:
                    logger.error(f"Download {target} failed: {e}")
                    failed.append(f"{target}: {e}")

            if failed:
                msg = (
                    "Some downloads failed:\n"
                    + "\n".join(failed)
                    + "\n\nTip: If network issues, set env HF_ENDPOINT=https://hf-mirror.com"
                )
                self._event_queue.put(("error", msg))
            else:
                self._event_queue.put(("status", "Download complete!"))
                self._event_queue.put(("dl_progress", "All downloads complete!"))
        except Exception as e:
            self._event_queue.put(("error", f"Download error: {e}"))
        finally:
            self._event_queue.put(("download_done", None))

    def _download_langs_thread(self, langs: list[str]) -> None:
        try:
            from copytalker.core.constants import DEFAULT_TRANSLATION_MODELS
            from copytalker.utils.model_cache import ModelCache

            cache = ModelCache()
            failed: list[str] = []

            for lang in langs:
                directions = [f"en->{lang}", f"{lang}->en"]
                for direction in directions:
                    models = DEFAULT_TRANSLATION_MODELS.get(direction, [])
                    for model_name in models:
                        try:
                            self._event_queue.put(("dl_progress", f"Downloading {model_name}..."))
                            cache.download_translation_model(model_name)
                        except Exception as e:
                            logger.error(f"Failed to download {model_name}: {e}")
                            failed.append(f"{model_name}: {e}")

                try:
                    self._event_queue.put(("dl_progress", f"Downloading NLLB for {lang}..."))
                    cache.download_translation_model("facebook/nllb-200-distilled-600M")
                except Exception as e:
                    logger.error(f"Failed to download NLLB: {e}")
                    failed.append(f"nllb: {e}")

            if failed:
                msg = "Some downloads failed:\n" + "\n".join(failed)
                self._event_queue.put(("status", f"Completed with errors: {len(failed)}"))
                self._event_queue.put(("dl_progress", msg))
            else:
                self._event_queue.put(("status", "Translation models downloaded successfully"))
                self._event_queue.put(("dl_progress", "Download complete!"))
        except Exception as e:
            self._event_queue.put(("error", f"Download error: {e}"))
        finally:
            self._event_queue.put(("download_done", None))
