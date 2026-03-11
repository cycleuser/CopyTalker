"""
GUI module for CopyTalker.
"""

from copytalker.gui.app import main, CopyTalkerApp

# Keep legacy import available for backwards compatibility
from copytalker.gui.main_window import CopyTalkerGUI

__all__ = ["main", "CopyTalkerApp", "CopyTalkerGUI"]
