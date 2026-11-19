# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT

import importlib.util
import sys

import pytest


@pytest.mark.parametrize("module_name", ["torch", "torchaudio.transforms", "scipy.signal", "audio.data"])
def test_optional_dependency_stub_is_findable(module_name):
    module = sys.modules[module_name]
    assert module.__spec__ is not None
    assert importlib.util.find_spec(module_name) is module.__spec__
