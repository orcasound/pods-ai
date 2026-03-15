# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Unit tests for get_best_timestamp.py.

Tests focus on node_name_to_slug() and build_sample(), which contain the
helper logic specific to this script.
"""
import pytest

from get_best_timestamp import build_sample, node_name_to_slug


# ---------------------------------------------------------------------------
# node_name_to_slug
# ---------------------------------------------------------------------------

class TestNodeNameToSlug:
    """Tests for converting a node_name to a URL slug."""

    def test_strips_rpi_prefix(self):
        """'rpi_' prefix should be removed."""
        assert node_name_to_slug('rpi_orcasound_lab') == 'orcasound-lab'

    def test_replaces_underscores_with_hyphens(self):
        """Underscores in the remainder should become hyphens."""
        assert node_name_to_slug('rpi_andrews_bay') == 'andrews-bay'

    def test_single_word_after_prefix(self):
        """A single-word name after 'rpi_' should convert correctly."""
        assert node_name_to_slug('rpi_bush_point') == 'bush-point'

    def test_multi_word_node(self):
        """Multi-word names should convert all underscores to hyphens."""
        assert node_name_to_slug('rpi_sunset_bay') == 'sunset-bay'

    def test_no_rpi_prefix(self):
        """Names without an 'rpi_' prefix should only have underscores replaced."""
        assert node_name_to_slug('orcasound_lab') == 'orcasound-lab'

    def test_already_slug(self):
        """A value that is already a slug should be returned unchanged."""
        assert node_name_to_slug('andrews-bay') == 'andrews-bay'


# ---------------------------------------------------------------------------
# build_sample
# ---------------------------------------------------------------------------

class TestBuildSample:
    """Tests for constructing a sample dict from node_name and timestamp_str."""

    NODE = 'rpi_orcasound_lab'
    TIMESTAMP = '2023_08_18_00_59_53_PST'

    def _sample(self):
        return build_sample(self.NODE, self.TIMESTAMP)

    def test_node_name_preserved(self):
        """NodeName should be the original node_name argument."""
        assert self._sample()['NodeName'] == self.NODE

    def test_timestamp_preserved(self):
        """Timestamp should be the original timestamp_str argument."""
        assert self._sample()['Timestamp'] == self.TIMESTAMP

    def test_notes_is_tp_human_only(self):
        """Notes should be 'tp_human_only' so process_sample uses model correction."""
        assert self._sample()['Notes'] == 'tp_human_only'

    def test_uri_contains_slug(self):
        """URI should contain the slug derived from node_name."""
        assert 'orcasound-lab' in self._sample()['URI']

    def test_uri_starts_with_bouts_base(self):
        """URI should start with the Orcasound bouts interface base URL."""
        assert self._sample()['URI'].startswith(
            'https://live.orcasound.net/bouts/new/orcasound-lab'
        )

    def test_uri_contains_encoded_timestamp(self):
        """URI should contain the UTC-encoded version of the given timestamp."""
        # 2023-08-18 is in summer (PDT = UTC-7), so 00:59:53 PDT = 07:59:53 UTC.
        assert '2023-08-18T07%3A59%3A53' in self._sample()['URI']
