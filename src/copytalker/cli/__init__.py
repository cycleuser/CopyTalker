"""
CLI module for CopyTalker.
"""

__all__ = ["main"]


def __getattr__(name: str):
    """Lazy import for CLI main."""
    if name == "main":
        from copytalker.cli.main import main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
