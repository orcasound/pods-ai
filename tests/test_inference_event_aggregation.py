# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Unit tests for non-adjacent positive-event aggregation (#413)."""

import pytest

from podsai_inference import (
    count_non_adjacent_positive_events,
    meets_min_positive_event_threshold,
)


@pytest.mark.parametrize(
    "mask, expected_events",
    [
        ([], 0),
        ([0], 0),
        ([1], 1),
        ([0, 0, 0, 0], 0),
        ([1, 1, 1, 0], 1),
        ([1, 0, 1, 0], 2),
        ([1, 1, 0, 1, 1], 2),
        ([1, 1, 1, 1], 1),
        ([0, 1, 1, 0, 1], 2),
        ([True, True, False, True], 2),
        ([False, False], 0),
    ],
)
def test_count_non_adjacent_positive_events(mask, expected_events):
    assert count_non_adjacent_positive_events(mask) == expected_events


@pytest.mark.parametrize(
    "mask, threshold, expected",
    [
        ([0, 0, 0, 0], 1, False),
        ([0, 0, 0, 0], 3, False),
        ([1, 1, 1, 0], 1, True),
        ([1, 1, 1, 0], 2, False),
        ([1, 1, 1, 0], 3, False),
        ([1, 0, 1, 0], 1, True),
        ([1, 0, 1, 0], 2, True),
        ([1, 0, 1, 0], 3, False),
        ([1, 1, 0, 1, 1], 2, True),
        ([1, 1, 0, 1, 1], 3, False),
        ([1, 0, 1, 0, 1], 3, True),
        ([], 1, False),
        ([1], 1, True),
        ([1], 2, False),
    ],
)
def test_meets_min_positive_event_threshold(mask, threshold, expected):
    assert meets_min_positive_event_threshold(mask, threshold) is expected
