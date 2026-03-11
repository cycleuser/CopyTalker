# CopyTalker Agent Guidelines

## 1. Project Overview
**CopyTalker** is a cross-modal data conversion driven asynchronous multi-voice translation system. It supports real-time speech-to-speech translation with multiple languages and TTS engines (Kokoro, Edge TTS, pyttsx3).

**Architecture**:
- `src/copytalker/`: Main source code
  - `core/`: Configuration, pipeline, constants
  - `audio/`: Audio capture and VAD
  - `speech/`: Speech-to-text (Whisper)
  - `translation/`: Translation models (Helsinki-NLP, NLLB)
  - `tts/`: Text-to-speech engines
  - `gui/`: Tkinter GUI
  - `cli/`: Command-line interface
  - `api.py`: Unified Python API for agent integration

## 2. Build, Lint, and Test Commands

### Prerequisites
Install dev dependencies:
```bash
pip install -e .[dev]
```

### Code Quality (Linting & Formatting)
Run these commands before committing:
```bash
# Format code (Black & isort)
black src/copytalker tests
isort src/copytalker tests

# Lint check (Ruff)
ruff check src/copytalker

# Type checking (Mypy)
mypy src/copytalker
```

### Testing
Run the test suite using `pytest`:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=copytalker

# Run only unit tests
pytest tests/unit/

# Run fast tests only (skip slow)
pytest -m "not slow"
```

**Running a Single Test**:
To run a specific test function, use the format `pytest <file_path>::<test_function_name>`:
```bash
# Example: Run a specific test in test_config.py
pytest tests/unit/test_config.py::test_app_config_defaults
```

## 3. Code Style Guidelines

### Imports
- **Organization**: Group imports in the following order:
  1. Standard library imports (`os`, `sys`, `pathlib`, etc.)
  2. Third-party imports (`numpy`, `torch`, `transformers`, etc.)
  3. Local application imports (`from copytalker...`)
- **Formatting**: Use `isort` to enforce this (configured in `pyproject.toml`).
- **Lazy Loading**: Heavy dependencies in `copytalker/__init__.py` use `__getattr__` for lazy loading to reduce startup time.

### Formatting
- **Line Length**: 100 characters (configured in `pyproject.toml`).
- **Style**: Use `black` for consistent formatting.
- **Quotes**: Prefer double quotes for docstrings and strings (Black default).

### Typing
- **Python Version**: 3.9+.
- **Type Hints**: Use standard Python type hints.
- **Dataclasses**: Prefer `@dataclass` for configuration objects (e.g., `AudioConfig`, `STTConfig`).
- **Optional/Union**: Use `str | None` (Python 3.10+ syntax is preferred as 3.9 supports `from __future__ import annotations`).
- **Any**: Use `Any` sparingly; prefer specific types or `dict[str, Any]`.

### Naming Conventions
- **Variables/Functions**: `snake_case`.
- **Classes**: `PascalCase`.
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_SAMPLE_RATE`).
- **Private Members**: Prefix with underscore `_`.

### Error Handling
- **Exceptions**: Wrap API calls in `try...except` blocks.
- **Return Type**: Use the `ToolResult` dataclass (from `api.py`) for standardized returns:
  ```python
  @dataclass
  class ToolResult:
      success: bool
      data: Any = None
      error: Optional[str] = None
      metadata: dict = field(default_factory=dict)
  ```
- **Logging**: Use `logging.getLogger(__name__)` for debug logs.

### Documentation
- **Docstrings**: Use Google-style or NumPy-style docstrings.
- **Comments**: Avoid inline comments unless necessary; explain "why" not "what".

### Pre-commit Hooks
If available, run pre-commit hooks to ensure code quality before pushing.

## 4. Context & Configuration
- **Configuration**: Managed via `AppConfig` dataclass in `core/config.py`.
- **Device Detection**: Uses `torch` for CUDA/MPS detection; falls back to CPU.
- **Models**: Heavily relies on Hugging Face `transformers` and `faster-whisper`.
- **Audio**: Uses `sounddevice` for I/O and `webrtcvad` for Voice Activity Detection.

## 5. Agent Integration
CopyTalker exposes OpenAI-compatible tools via `copytalker.tools.TOOLS` and `copytalker.api`.
- Use `translate()`, `tts_synthesize()`, etc., for programmatic access.
- Always check `ToolResult.success` before accessing data.
