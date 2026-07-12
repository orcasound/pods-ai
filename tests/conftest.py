# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Pytest configuration for pods-ai unit tests.

Adds the src directory to sys.path so that modules under src can
be imported directly, and mocks heavy dependencies (ML, audio) that are not
needed for unit tests so the suite can run without a full GPU/fastai environment.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Ensure src/ and bootstrap/src are on the path before any test module is imported.
sys.path.insert(0, str(Path(__file__).parent.parent / 'bootstrap' / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Packages that are either heavy (torch/fastai) or optional (numpy/pandas) in
# this environment.  Each is stubbed out only when it cannot be genuinely
# imported; CI (which runs pip install -r requirements.txt) will use the real
# packages.
#
# NOTE: 'audio' and 'audio.data' are intentionally NOT in this list.
# Unit tests that need to mock AudioList should use @patch() explicitly.
# Integration tests need the real audio module to function.
_OPTIONAL_DEPS = [
    'azure',
    'azure.cosmos',
    'azure.storage',
    'azure.storage.blob',
    'dotenv',
    'numpy',
    'opencensus',
    'opencensus.ext',
    'opencensus.ext.azure',
    'opencensus.ext.azure.log_exporter',
    'pandas',
    'pytz',
    'structlog',
    'torch',
    'torchvision',
    'torchaudio',
    'torchaudio.transforms',
    'fastai',
    'fastai.basic_train',
    'pydub',
    'pydub.audio_segment',
    'librosa',
    'soundfile',
    'scipy',
    'scipy.signal',
    'huggingface_hub',
    'yaml',
]
for _dep in _OPTIONAL_DEPS:
    if _dep not in sys.modules:
        try:
            __import__(_dep)
        except ImportError:
            sys.modules[_dep] = MagicMock()

# Special handling for mcp modules: requires a proper FastMCP mock so that
# @mcp.tool() decorators preserve the original functions rather than wrapping
# them in MagicMocks.  These stubs are installed early so that pytest test
# collection does not block on MCP initialisation.
if 'mcp.server.fastmcp' not in sys.modules:
    try:
        import mcp.server.fastmcp  # noqa: F401
    except ImportError:
        class _MockFastMCP:
            """Minimal FastMCP stub: tool() returns a pass-through decorator."""
            def __init__(self, name):
                self.name = name

            def tool(self, *args, **kwargs):
                def decorator(func):
                    return func
                return decorator

            def run(self):
                pass

            def _handle_raw_json(self, line):
                pass

        _mock_mcp_module = MagicMock()
        _mock_mcp_module.server.fastmcp.FastMCP = _MockFastMCP
        sys.modules.setdefault('mcp', _mock_mcp_module)
        sys.modules.setdefault('mcp.server', _mock_mcp_module.server)
        sys.modules['mcp.server.fastmcp'] = _mock_mcp_module.server.fastmcp
