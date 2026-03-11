#!/usr/bin/env python3
"""
TTS Engine Availability Checker for CopyTalker.

This script checks which TTS engines are available and provides installation suggestions.
"""

import sys
import subprocess
from typing import Tuple, Optional


def check_package(package_name: str) -> Tuple[bool, Optional[str]]:
    """Check if a package is installed."""
    try:
        __import__(package_name.replace("-", "_"))
        return True, None
    except ImportError as e:
        return False, str(e)


def check_edge_tts() -> Tuple[bool, str]:
    """Check if Edge TTS is available and working."""
    installed, error = check_package("edge_tts")
    if not installed:
        return False, f"Package not installed: {error}"
    
    # Try to list voices to verify network access
    try:
        result = subprocess.run(
            ["edge-tts", "--list-voices"],
            capture_output=True,
            timeout=10
        )
        if result.returncode == 0:
            return True, "Edge TTS is available"
        else:
            return False, f"Edge TTS CLI failed: {result.stderr.decode()}"
    except FileNotFoundError:
        return False, "edge-tts CLI not found (package may be corrupted)"
    except subprocess.TimeoutExpired:
        return False, "Edge TTS network timeout (check internet connection)"
    except Exception as e:
        return False, f"Edge TTS error: {e}"


def check_kokoro() -> Tuple[bool, str]:
    """Check if Kokoro TTS is available."""
    installed, error = check_package("kokoro")
    if not installed:
        return False, f"Package not installed: {error}"
    
    # Check CJK dependencies
    cjk_deps = []
    for pkg in ["cn2an", "pypinyin", "jieba"]:
        ok, _ = check_package(pkg)
        if not ok:
            cjk_deps.append(pkg)
    
    jp_deps = []
    for pkg in ["fugashi", "jaconv", "unidic"]:
        ok, _ = check_package(pkg)
        if not ok:
            jp_deps.append(pkg)
    
    status = "Kokoro TTS is available"
    if cjk_deps:
        status += f"\n  ⚠️  Missing CJK dependencies: {', '.join(cjk_deps)}"
        status += "\n    Install with: pip install cn2an pypinyin jieba"
    if jp_deps:
        status += f"\n  ⚠️  Missing Japanese dependencies: {', '.join(jp_deps)}"
        status += "\n    Install with: pip install fugashi jaconv unidic-lite"
    
    return True, status


def check_pyttsx3() -> Tuple[bool, str]:
    """Check if pyttsx3 is available."""
    installed, error = check_package("pyttsx3")
    if not installed:
        return False, f"Package not installed: {error}"
    
    try:
        import pyttsx3
        engine = pyttsx3.init()
        return True, "pyttsx3 is available (offline system TTS)"
    except Exception as e:
        return False, f"pyttsx3 initialization failed: {e}"


def check_sounddevice() -> Tuple[bool, str]:
    """Check if sounddevice is available."""
    installed, error = check_package("sounddevice")
    if not installed:
        return False, f"Package not installed: {error}"
    
    return True, "sounddevice is available (audio I/O)"


def print_status(name: str, available: bool, details: str) -> None:
    """Print status with color coding."""
    status = "✓" if available else "✗"
    color = "\033[92m" if available else "\033[91m"
    reset = "\033[0m"
    
    print(f"{color}{status}\033[0m {name}")
    if details:
        print(f"   {details}")


def main():
    """Main function."""
    print("=" * 60)
    print("CopyTalker TTS Engine Availability Checker")
    print("=" * 60)
    print()
    
    results = []
    
    # Check audio backend
    print("Audio Backend:")
    available, details = check_sounddevice()
    print_status("sounddevice", available, details)
    results.append(("sounddevice", available))
    print()
    
    # Check TTS engines
    print("TTS Engines:")
    
    available, details = check_kokoro()
    print_status("Kokoro TTS", available, details)
    results.append(("Kokoro", available))
    
    available, details = check_edge_tts()
    print_status("Edge TTS", available, details)
    results.append(("Edge TTS", available))
    
    available, details = check_pyttsx3()
    print_status("pyttsx3", available, details)
    results.append(("pyttsx3", available))
    
    print()
    print("=" * 60)
    
    # Summary
    available_count = sum(1 for _, avail in results if avail)
    total_count = len(results) - 1  # Exclude sounddevice
    
    if available_count == 0:
        print("\033[91m⚠️  No TTS engines available!\033[0m")
        print("\nInstall at least one TTS engine:")
        print("  pip install kokoro              # Recommended (high quality)")
        print("  pip install edge-tts            # Cloud-based (requires internet)")
        print("  pip install pyttsx3             # Offline fallback")
        print("\nOr install all at once:")
        print("  pip install copytalker[full]")
    elif available_count == 1:
        print(f"\033[93m✓ {available_count} TTS engine available\033[0m")
        print("\nConsider installing additional engines for better quality:")
        if not results[0][1]:
            print("  pip install kokoro              # Recommended upgrade")
        if not results[1][1]:
            print("  pip install edge-tts            # For cloud voices")
        if not results[2][1]:
            print("  pip install pyttsx3             # For offline fallback")
    else:
        print(f"\033[92m✓ All {available_count} TTS engines are available!\033[0m")
    
    print()
    print("Installation guides:")
    print("  - Full guide: INSTALL.md")
    print("  - README: README.md / README_CN.md")
    print("=" * 60)
    
    # Return exit code based on availability
    if available_count == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
