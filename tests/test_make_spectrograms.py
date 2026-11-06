# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT

import pytest

import make_spectrograms


def test_help_displays_usage(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["make_spectrograms.py", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        make_spectrograms.main()

    assert exc_info.value.code == 0
    help_output = capsys.readouterr().out
    assert "usage:" in help_output
    assert "Generate spectrograms for WAV files in output/wav." in help_output
