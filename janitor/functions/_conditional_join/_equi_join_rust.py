"""Python preparation for the fresh equi/Rust join boundary."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from janitor.functions._conditional_join import _helpers
from janitor.functions._conditional_join._helpers import _prepare_range_anchor

RANGE_OPERATORS = frozenset({"<", "<=", ">", ">="})
EQUI_OPERATOR = "=="
NOT_EQUAL_OPERATOR = "!="

_RANGE_PAIR_PRIORITY = (
    (">", "<"),
    (">", "<="),
    (">=", "<"),
    (">=", "<="),
    (">", ">"),
    (">", ">="),
    (">=", ">"),
    (">=", ">="),
    ("<", "<"),
    ("<", "<="),
    ("<=", "<"),
    ("<=", "<="),
)


@dataclass(frozen=True)
class RangeInput:
    """One range predicate aligned to a physical right-side permutation."""

    left_values: np.ndarray
    right_values: np.ndarray
    left_index: np.ndarray
    right_index: np.ndarray
    operator: str


@dataclass(frozen=True)
class EquiLookup:
    """Direct or factorized equi lookup prepared for Rust."""

    left_keys: pd.Index
    right_keys: pd.Index
    left_indexer: np.ndarray
    left_codes: np.ndarray | None
    right_codes: np.ndarray | None
    right_index: np.ndarray
    uniques: pd.Index | None

    @property
    def right_is_unique(self) -> bool:
        return self.left_codes is None


@dataclass(frozen=True)
class PreparedEquiJoin:
    """Complete prepared payload for one fresh equi-led join."""

    left: pd.DataFrame
    right: pd.DataFrame
    lookup: EquiLookup
    ranges: tuple[RangeInput, ...]
    primary_range: RangeInput | None
    secondary_range: RangeInput | None
    residual_ranges: tuple[RangeInput, ...]
    residual_predicates: tuple[tuple[str, str, str], ...]


def prepare_equi_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[tuple[str, str, str]],
    range_inputs: tuple[RangeInput, ...] | None = None,
) -> PreparedEquiJoin | None:
    """Prepare Python-owned inputs for the fresh Rust equi path."""
    left, right = _get_indices(left, right, conditions)
    if left is None or right is None:
        return None

    equals = [
        (left_column, right_column)
        for left_column, right_column, operator in conditions
        if operator == EQUI_OPERATOR
    ]
    range_positions = [
        index
        for index, condition in enumerate(conditions)
        if condition[2] in RANGE_OPERATORS
    ]

    if range_inputs is None:
        primary_range, secondary_range, selected_positions = (
            _select_range_pair_plan_maybe(left, right, conditions)
        )
        ranges = tuple(
            range_input
            for range_input in (primary_range, secondary_range)
            if range_input is not None
        )
        selected_range_condition_positions = selected_positions
    else:
        ranges = tuple(range_inputs)
        primary_range = ranges[0] if ranges else None
        secondary_range = ranges[1] if len(ranges) > 1 else None
        selected_range_condition_positions = set(range_positions[: len(ranges)])

    lookup = _prepare_equi_lookup(left, right, equals)
    if lookup is None:
        return None

    residual_ranges: tuple[RangeInput, ...] = ()
    if len(ranges) > 1:
        if _same_right_layout(ranges[0], ranges[1]):
            residual_ranges = ranges[2:]
        else:
            secondary_range = None
            residual_ranges = ranges[1:]

    residual_predicates = tuple(
        condition
        for condition_index, condition in enumerate(conditions)
        if condition[2] != EQUI_OPERATOR
        and (
            condition[2] not in RANGE_OPERATORS
            or condition_index not in selected_range_condition_positions
        )
    )
    return PreparedEquiJoin(
        left=left,
        right=right,
        lookup=lookup,
        ranges=ranges,
        primary_range=primary_range,
        secondary_range=secondary_range,
        residual_ranges=residual_ranges,
        residual_predicates=residual_predicates,
    )


def _prepare_equi_lookup(
    left: pd.DataFrame,
    right: pd.DataFrame,
    equi_columns: list[tuple[str, str]],
) -> EquiLookup | None:
    left_keys, right_keys = _build_equi_keys(left, right, equi_columns)
    right_index = np.asarray(right.index._values, dtype=np.int64)
    try:
        left_indexer = np.asarray(right_keys.get_indexer(left_keys), dtype=np.int64)
        if np.all(left_indexer == -1):
            return None
        return EquiLookup(
            left_keys=left_keys,
            right_keys=right_keys,
            left_indexer=left_indexer,
            left_codes=None,
            right_codes=None,
            right_index=right_index,
            uniques=None,
        )
    except pd.errors.InvalidIndexError:
        right_codes, uniques = right_keys.factorize(sort=False)
        left_codes = uniques.get_indexer(left_keys)
        if np.all(left_codes == -1):
            return None
        return EquiLookup(
            left_keys=left_keys,
            right_keys=right_keys,
            left_indexer=np.asarray(left_codes, dtype=np.int64),
            left_codes=np.asarray(left_codes, dtype=np.int64),
            right_codes=np.asarray(right_codes, dtype=np.int64),
            right_index=right_index,
            uniques=uniques,
        )


def _build_equi_keys(
    left: pd.DataFrame,
    right: pd.DataFrame,
    equi_columns: list[tuple[str, str]],
) -> tuple[pd.Index, pd.Index]:
    left_arrays = [left[left_column]._values for left_column, _ in equi_columns]
    right_arrays = [right[right_column]._values for _, right_column in equi_columns]
    if len(left_arrays) == 1:
        return pd.Index(left_arrays[0]), pd.Index(right_arrays[0])
    return (
        pd.MultiIndex.from_arrays(left_arrays),
        pd.MultiIndex.from_arrays(right_arrays),
    )


def _range_input_from_anchor(anchor, operator: str) -> RangeInput:
    return RangeInput(
        left_values=anchor.left_array,
        right_values=anchor.right_array,
        left_index=anchor.left_index,
        right_index=anchor.right_index,
        operator=operator,
    )


def _select_range_pair_maybe(
    left: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[tuple[str, str, str]],
) -> tuple[RangeInput | None, RangeInput | None]:
    """Return the prioritized range pair, hiding condition positions."""
    first, second, _ = _select_range_pair_plan_maybe(left, right, conditions)
    return first, second


def _select_range_pair_plan_maybe(
    left: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[tuple[str, str, str]],
) -> tuple[RangeInput | None, RangeInput | None, set[int]]:
    """Adapt ``_range_join``'s prioritized dual-range selection."""
    range_conditions = [
        (position, condition)
        for position, condition in enumerate(conditions)
        if condition[2] in RANGE_OPERATORS
    ]
    if not range_conditions:
        return None, None, set()

    anchors = {}
    for position, condition in range_conditions:
        left_column, right_column, operator = condition
        anchor = _prepare_range_anchor(left[left_column], right[right_column])
        if anchor is not None:
            anchors[position] = (condition, anchor)
    if not anchors:
        return None, None, set()

    for first_operator, second_operator in _RANGE_PAIR_PRIORITY:
        for first_position, (first_condition, first_anchor) in anchors.items():
            if first_condition[2] != first_operator:
                continue
            for second_position, (second_condition, _) in anchors.items():
                if (
                    second_position == first_position
                    or second_condition[2] != second_operator
                ):
                    continue
                second_values = right.loc[first_anchor.right_index, second_condition[1]]
                if second_values.is_monotonic_increasing:
                    return (
                        _range_input_from_anchor(first_anchor, first_operator),
                        RangeInput(
                            left_values=np.asarray(left[second_condition[0]]._values),
                            right_values=np.asarray(second_values._values),
                            left_index=first_anchor.left_index,
                            right_index=first_anchor.right_index,
                            operator=second_condition[2],
                        ),
                        {first_position, second_position},
                    )

    first_position, (first_condition, first_anchor) = next(iter(anchors.items()))
    return (
        _range_input_from_anchor(first_anchor, first_condition[2]),
        None,
        {first_position},
    )


def _get_indices(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[tuple[str, str, str]],
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Apply the existing null policy before equi/range preparation."""
    left_columns = {
        left_column
        for left_column, _, operator in conditions
        if operator != NOT_EQUAL_OPERATOR
    }
    df = _helpers._maybe_remove_nulls_from_dataframe(df=df, columns=left_columns)
    if df is None:
        return None, None
    right_columns = {
        right_column
        for _, right_column, operator in conditions
        if operator != NOT_EQUAL_OPERATOR
    }
    right = _helpers._maybe_remove_nulls_from_dataframe(df=right, columns=right_columns)
    if right is None:
        return None, None
    return df, right


def _same_right_layout(first: RangeInput, second: RangeInput) -> bool:
    return np.array_equal(first.right_index, second.right_index)
