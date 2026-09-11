# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Unit tests for non-adjacent positive-event aggregation."""

import pytest

from podsai_inference import (
    _positive_event_ids,
    count_non_adjacent_positive_events,
    meets_min_positive_event_threshold,
)


def test_global_event_boundaries_are_shared_across_classes():
    """Class jitter within one global event must not create extra boundaries."""
    assert _positive_event_ids([1, 1, 1, 1, 1, 0, 1]) == [0, 0, 1, 1, 2, None, 3]


@pytest.mark.parametrize(
    "mask, expected_events",
    [
        ([], 0),
        ([0], 0),
        ([1], 1),
        ([0, 0, 0, 0], 0),
        ([1, 1, 1, 0], 2),
        ([1, 0, 1, 0], 2),
        ([1, 1, 0, 1, 1], 2),
        ([1, 1, 1, 1], 2),
        ([1, 1, 1, 1, 1], 3),
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
        ([1, 1, 1, 0], 2, True),
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
