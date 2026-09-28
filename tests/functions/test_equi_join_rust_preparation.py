import numpy as np
import pandas as pd

from janitor.functions._conditional_join._equi_join_rust import (
    RangeInput,
    _select_range_pair_maybe,
    prepare_equi_join,
)


def test_unique_right_keys_use_get_indexer_positions():
    left = pd.DataFrame({"key": ["b", "missing", "a"]})
    right = pd.DataFrame({"key": ["a", "b"]})

    prepared = prepare_equi_join(left, right, [("key", "key", "==")])

    assert prepared is not None
    assert prepared.lookup.right_is_unique
    np.testing.assert_array_equal(prepared.lookup.left_indexer, [1, -1, 0])


def test_duplicate_right_keys_use_factorized_codes():
    left = pd.DataFrame({"key": ["b", "a", "missing"]})
    right = pd.DataFrame({"key": ["a", "b", "a"]})

    prepared = prepare_equi_join(left, right, [("key", "key", "==")])

    assert prepared is not None
    assert not prepared.lookup.right_is_unique
    np.testing.assert_array_equal(prepared.lookup.left_codes, [1, 0, -1])
    np.testing.assert_array_equal(prepared.lookup.right_codes, [0, 1, 0])


def test_no_equi_key_match_returns_none():
    left = pd.DataFrame({"key": ["missing"]})
    right = pd.DataFrame({"key": ["present"]})

    assert prepare_equi_join(left, right, [("key", "key", "==")]) is None


def test_nulls_are_removed_before_equi_lookup_but_not_equal_is_preserved():
    left = pd.DataFrame({"key": [1, None], "value": [10, None]})
    right = pd.DataFrame({"key": [1, None], "value": [20, None]})

    prepared = prepare_equi_join(
        left,
        right,
        [("key", "key", "=="), ("value", "value", "!=")],
    )

    assert prepared is not None
    assert len(prepared.left) == 1
    assert len(prepared.right) == 1
    assert prepared.residual_predicates == (("value", "value", "!="),)


def test_aligned_dual_ranges_are_sent_to_rust_for_intersection():
    left = pd.DataFrame({"key": [1], "lower": [2], "upper": [8]})
    right = pd.DataFrame(
        {"key": [1, 1], "lower": [3, 7], "upper": [7, 9]},
        index=[10, 11],
    )

    prepared = prepare_equi_join(
        left,
        right,
        [
            ("key", "key", "=="),
            ("upper", "upper", ">="),
            ("lower", "lower", "<="),
        ],
    )

    assert prepared is not None
    assert prepared.primary_range is prepared.ranges[0]
    assert prepared.secondary_range is prepared.ranges[1]
    assert prepared.residual_ranges == ()
    assert prepared.residual_predicates == ()


def test_no_range_candidates_return_no_range_pair():
    left = pd.DataFrame({"key": [1]})
    right = pd.DataFrame({"key": [1]})

    assert _select_range_pair_maybe(left, right, [("key", "key", "==")]) == (
        None,
        None,
    )


def test_one_range_candidate_returns_only_the_sorted_first_pair():
    left = pd.DataFrame({"key": [1], "bound": [5]})
    right = pd.DataFrame({"key": [1, 1], "bound": [8, 3]}, index=[10, 11])

    first, second = _select_range_pair_maybe(
        left, right, [("key", "key", "=="), ("bound", "bound", "<=")]
    )

    assert second is None
    np.testing.assert_array_equal(first.right_values, [3, 8])
    np.testing.assert_array_equal(first.right_index, [11, 10])


def test_second_range_must_be_increasing_in_first_range_layout():
    left = pd.DataFrame({"key": [1], "first": [5], "second": [5]})
    right = pd.DataFrame(
        {"key": [1, 1, 1], "first": [8, 3, 5], "second": [1, 9, 4]},
        index=[10, 11, 12],
    )

    first, second = _select_range_pair_maybe(
        left,
        right,
        [("key", "key", "=="), ("first", "first", "<="), ("second", "second", "<=")],
    )

    assert second is None
    np.testing.assert_array_equal(first.right_index, [11, 12, 10])


def test_rejected_second_range_is_a_residual_predicate():
    left = pd.DataFrame({"key": [1], "first": [5], "second": [5]})
    right = pd.DataFrame(
        {"key": [1, 1], "first": [8, 3], "second": [1, 9]},
    )

    prepared = prepare_equi_join(
        left,
        right,
        [
            ("key", "key", "=="),
            ("first", "first", "<="),
            ("second", "second", "<="),
        ],
    )

    assert prepared is not None
    assert prepared.secondary_range is None
    assert prepared.residual_predicates == (("second", "second", "<="),)


def test_different_dual_range_layouts_keep_second_as_residual():
    left = pd.DataFrame({"key": [1], "lower": [2], "upper": [8]})
    right = pd.DataFrame(
        {"key": [1, 1], "lower": [3, 7], "upper": [9, 7]},
        index=[10, 11],
    )

    ranges = (
        RangeInput(
            left_values=np.array([2]),
            right_values=np.array([3, 7]),
            left_index=np.array([0]),
            right_index=np.array([10, 11]),
            operator="<=",
        ),
        RangeInput(
            left_values=np.array([8]),
            right_values=np.array([7, 9]),
            left_index=np.array([0]),
            right_index=np.array([11, 10]),
            operator=">=",
        ),
    )
    prepared = prepare_equi_join(
        left,
        right,
        [
            ("key", "key", "=="),
            ("lower", "lower", "<="),
            ("upper", "upper", ">="),
        ],
        range_inputs=ranges,
    )

    assert prepared is not None
    assert prepared.secondary_range is None
    assert prepared.residual_ranges == (prepared.ranges[1],)
