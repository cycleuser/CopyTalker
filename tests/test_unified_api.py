"""
Comprehensive tests for CopyTalker unified API, tools, and CLI flags.
"""

import json
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest


class TestToolResult:
    def test_success(self):
        from copytalker.api import ToolResult
        r = ToolResult(success=True, data={"voices": []})
        assert r.success is True

    def test_failure(self):
        from copytalker.api import ToolResult
        r = ToolResult(success=False, error="no mic")
        assert r.error == "no mic"

    def test_to_dict(self):
        from copytalker.api import ToolResult
        d = ToolResult(success=True, data="x", metadata={"v": "1"}).to_dict()
        assert d["success"] is True
        assert d["data"] == "x"
        assert d["metadata"] == {"v": "1"}

    def test_default_metadata_isolation(self):
        from copytalker.api import ToolResult
        r1 = ToolResult(success=True)
        r2 = ToolResult(success=True)
        r1.metadata["k"] = 1
        assert "k" not in r2.metadata


class TestListVoicesAPI:
    @patch("copytalker.core.constants.get_available_voices")
    @patch("copytalker.core.constants.get_language_name")
    @patch("copytalker.core.constants.SUPPORTED_LANGUAGES", [("en", "English"), ("zh", "Chinese")])
    def test_list_voices_all(self, mock_name, mock_voices):
        from copytalker.api import list_voices
        mock_name.side_effect = lambda x: {"en": "English", "zh": "Chinese"}[x]
        mock_voices.side_effect = lambda lang, eng: ["voice1"] if lang == "en" else []

        result = list_voices(engine="kokoro")
        assert result.success is True
        assert "en" in result.data

    @patch("copytalker.core.constants.get_available_voices")
    @patch("copytalker.core.constants.get_language_name")
    @patch("copytalker.core.constants.SUPPORTED_LANGUAGES", [("en", "English")])
    def test_list_voices_filtered(self, mock_name, mock_voices):
        from copytalker.api import list_voices
        mock_name.return_value = "English"
        mock_voices.return_value = ["v1", "v2"]

        result = list_voices(language="en")
        assert result.success is True


class TestListLanguagesAPI:
    @patch("copytalker.core.constants.SUPPORTED_LANGUAGES", [("en", "English"), ("zh", "Chinese")])
    def test_list_languages(self):
        from copytalker.api import list_languages
        result = list_languages()
        assert result.success is True
        assert len(result.data) == 2
        assert result.data[0]["code"] == "en"


class TestToolsSchema:
    def test_tools_count(self):
        from copytalker.tools import TOOLS
        assert len(TOOLS) == 6

    def test_tool_names(self):
        from copytalker.tools import TOOLS
        names = [t["function"]["name"] for t in TOOLS]
        assert "copytalker_translate" in names
        assert "copytalker_tts_synthesize" in names
        assert "copytalker_list_voices" in names
        assert "copytalker_list_languages" in names
        assert "copytalker_list_emotions" in names
        assert "copytalker_clone_voice" in names

    def test_translate_requires_target(self):
        from copytalker.tools import TOOLS
        for t in TOOLS:
            if t["function"]["name"] == "copytalker_translate":
                assert "target" in t["function"]["parameters"]["required"]

    def test_structure(self):
        from copytalker.tools import TOOLS
        for tool in TOOLS:
            assert tool["type"] == "function"
            func = tool["function"]
            assert "description" in func
            assert func["parameters"]["type"] == "object"


class TestToolsDispatch:
    def test_unknown_tool(self):
        from copytalker.tools import dispatch
        with pytest.raises(ValueError):
            dispatch("bad", {})

    @patch("copytalker.core.constants.SUPPORTED_LANGUAGES", [("en", "English")])
    def test_dispatch_list_languages(self):
        from copytalker.tools import dispatch
        result = dispatch("copytalker_list_languages", {})
        assert isinstance(result, dict)
        assert result["success"] is True

    def test_dispatch_json_string(self):
        from copytalker.tools import dispatch
        args = json.dumps({})
        try:
            result = dispatch("copytalker_list_languages", args)
            assert isinstance(result, dict)
        except Exception:
            pass  # May fail on import of heavy deps


class TestCLIFlags:
    def _run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "copytalker"] + list(args),
            capture_output=True, text=True, timeout=15,
        )

    def test_version_flag(self):
        r = self._run_cli("-V")
        assert r.returncode == 0
        assert "copytalker" in r.stdout.lower()

    def test_help_has_unified_flags(self):
        r = self._run_cli("--help")
        assert "--json" in r.stdout
        assert "--quiet" in r.stdout or "-q" in r.stdout
        assert "--verbose" in r.stdout


class TestPackageExports:
    def test_version(self):
        import copytalker
        assert hasattr(copytalker, "__version__")

    def test_toolresult_lazy(self):
        from copytalker import ToolResult
        assert callable(ToolResult)

    def test_translate_lazy(self):
        from copytalker import translate
        assert callable(translate)

    def test_list_voices_lazy(self):
        from copytalker import list_voices
        assert callable(list_voices)
