"""
Model cache management and downloading utilities.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Callable

from copytalker.core.config import get_default_cache_dir
from copytalker.core.constants import MODEL_SIZES
from copytalker.core.exceptions import ModelDownloadError

logger = logging.getLogger(__name__)


class ModelCache:
    """
    Manages model caching and downloading.
    
    Uses HuggingFace Hub for model downloads with progress tracking.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize model cache.
        
        Args:
            cache_dir: Custom cache directory (uses default if None)
        """
        self.cache_dir = cache_dir or get_default_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def whisper_dir(self) -> Path:
        """Whisper model cache directory."""
        return self.cache_dir / "whisper"
    
    @property
    def translation_dir(self) -> Path:
        """Translation model cache directory."""
        return self.cache_dir / "translation"
    
    @property
    def tts_dir(self) -> Path:
        """TTS model cache directory."""
        return self.cache_dir / "tts"
    
    def ensure_dirs(self) -> None:
        """Create all cache subdirectories."""
        for d in [self.whisper_dir, self.translation_dir, self.tts_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def download_whisper_model(
        self,
        model_size: str = "small",
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Path:
        """
        Download Whisper model if not cached.
        
        Args:
            model_size: Model size ('tiny', 'base', 'small', 'medium', 'large')
            progress_callback: Optional callback for progress updates (0.0-1.0)
            
        Returns:
            Path to model directory
        """
        logger.info(f"Checking Whisper model: {model_size}")
        
        # faster-whisper handles its own caching via huggingface_hub
        # We just need to trigger the download
        try:
            from faster_whisper import WhisperModel
            
            # This will download if not cached
            logger.info(f"Downloading Whisper {model_size} model...")
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            del model
            
            logger.info(f"Whisper {model_size} model ready")
            return self.whisper_dir
            
        except Exception as e:
            logger.error(f"Failed to download Whisper model: {e}")
            raise ModelDownloadError(f"Whisper download failed: {e}") from e
    
    def download_translation_model(
        self,
        model_name: str,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Path:
        """
        Download translation model if not cached.
        
        Args:
            model_name: HuggingFace model ID
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to model directory
        """
        logger.info(f"Checking translation model: {model_name}")
        
        try:
            from huggingface_hub import snapshot_download
            
            model_path = snapshot_download(
                repo_id=model_name,
                cache_dir=str(self.translation_dir),
            )
            
            logger.info(f"Translation model {model_name} ready")
            return Path(model_path)
            
        except Exception as e:
            logger.error(f"Failed to download translation model: {e}")
            raise ModelDownloadError(f"Translation model download failed: {e}") from e
    
    def download_kokoro_model(
        self,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Path:
        """
        Download Kokoro TTS model if not cached.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to model directory
        """
        logger.info("Checking Kokoro TTS model")
        
        kokoro_dir = self.tts_dir / "kokoro"
        
        try:
            from huggingface_hub import snapshot_download
            
            model_path = snapshot_download(
                repo_id="hexgrad/Kokoro-82M",
                cache_dir=str(kokoro_dir),
            )
            
            logger.info("Kokoro TTS model ready")
            return Path(model_path)
            
        except Exception as e:
            logger.error(f"Failed to download Kokoro model: {e}")
            raise ModelDownloadError(f"Kokoro download failed: {e}") from e
    
    def get_cached_models(self) -> Dict[str, List[str]]:
        """
        Get list of cached models.
        
        Returns:
            Dictionary with model types as keys and lists of model names
        """
        cached = {
            "whisper": [],
            "translation": [],
            "tts": [],
        }
        
        # Check whisper models
        if self.whisper_dir.exists():
            for item in self.whisper_dir.iterdir():
                if item.is_dir():
                    cached["whisper"].append(item.name)
        
        # Check translation models
        if self.translation_dir.exists():
            for item in self.translation_dir.iterdir():
                if item.is_dir():
                    cached["translation"].append(item.name)
        
        # Check TTS models
        if self.tts_dir.exists():
            for item in self.tts_dir.iterdir():
                if item.is_dir():
                    cached["tts"].append(item.name)
        
        return cached
    
    def get_cache_size(self) -> int:
        """
        Get total cache size in bytes.
        
        Returns:
            Total size of all cached models in bytes
        """
        total = 0
        for path in self.cache_dir.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
        return total
    
    def clear_cache(self, model_type: Optional[str] = None) -> None:
        """
        Clear cached models.
        
        Args:
            model_type: Type to clear ('whisper', 'translation', 'tts', or None for all)
        """
        if model_type is None:
            dirs = [self.whisper_dir, self.translation_dir, self.tts_dir]
        elif model_type == "whisper":
            dirs = [self.whisper_dir]
        elif model_type == "translation":
            dirs = [self.translation_dir]
        elif model_type == "tts":
            dirs = [self.tts_dir]
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        for d in dirs:
            if d.exists():
                shutil.rmtree(d)
                logger.info(f"Cleared cache: {d}")
        
        self.ensure_dirs()
    
    @staticmethod
    def get_model_size_info(model_name: str) -> str:
        """
        Get human-readable size for a model.
        
        Args:
            model_name: Model name or type
            
        Returns:
            Human-readable size string (e.g., "~465 MB")
        """
        return MODEL_SIZES.get(model_name, "Unknown size")


def format_size(size_bytes: int) -> str:
    """Format byte size to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
