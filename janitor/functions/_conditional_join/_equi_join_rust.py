"""Python preparation for the fresh equi/Rust join boundary."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from janitor.functions._conditional_join import _helpers



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


    left_index: np.ndarray
    right_index: np.ndarray
    left_indexer: np.ndarray
    original_right_positions: np.ndarray | None



@dataclass(frozen=True)
class PreparedEquiJoin:
    """Complete prepared payload for one fresh equi-led join."""

    left: pd.DataFrame
    right: pd.DataFrame
    lookup: EquiLookup
    ranges: tuple[RangeInput, ...]
    residual_predicates: tuple[tuple[str, str, str], ...]





def _prepare_equi_lookup(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_keys: pd.Index,
    right_keys: pd.Index,
) -> EquiLookup | None:
    try:
        left_indexer = right_keys.get_indexer(left_keys)
        if np.all(left_indexer == -1):
            return None
        return EquiLookup(
            left_index = _helpers._convert_array_to_numpy(array=left.index._values),
            left_indexer=left_indexer,
            original_right_positions=None,
            right_index=_helpers._convert_array_to_numpy(array=right.index._values),
        )
    except pd.errors.InvalidIndexError:
        right_codes, uniques = right_keys.factorize(sort=False)
        left_indexer = uniques.get_indexer(left_keys)
        if np.all(left_indexer == -1):
            return None
        return EquiLookup(
            left_index = _helpers._convert_array_to_numpy(array=left.index._values),
            left_indexer=left_indexer,
            original_right_positions=np.asarray(right_codes, dtype=np.int64),
            right_index=_helpers._convert_array_to_numpy(array=right.index._values),
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




def _get_indices(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[tuple[str, str, str]],
) -> tuple|None:
    """Apply the existing null policy before equi/range preparation."""
    columns = {
        left_column
        for left_column, _, operator in conditions
        if operator != _helpers._JoinOperator.NOT_EQUAL.value
    }
    df = _helpers._maybe_remove_nulls_from_dataframe(df=df, columns=columns)
    if df is None:
        return None
    columns = {
        right_column
        for _, right_column, operator in conditions
        if operator != _helpers._JoinOperator.NOT_EQUAL.value
    }
    right = _helpers._maybe_remove_nulls_from_dataframe(df=right, columns=columns)
    if right is None:
        return None
    conditions_  = _helpers._separate_conditions_based_on_join_op(conditions=conditions)
    # do we have range joins?
    le_lt = conditions_.get('le_lt')
    ge_gt = conditions_.get('ge_gt')
    range_maybe = []
    if le_lt is not None:
        range_maybe.append(le_lt[0])
    if ge_gt is not None:
        range_maybe.append(ge_gt[0])
    # do we have range joins?
    if range_maybe:
        first_range = range_maybe[0]
        first_range = right[first_range]
        first_range, _ = _helpers._sort_if_not_monotonic(series=first_range)
        if len(range_maybe) > 1:
            second_range = range_maybe[1]
            second_range = right.loc[first_range.index, second_range]
            if not second_range.is_monotonic_increasing:
                range_maybe = range_maybe[0]
        right_index = first_range.index
    else:
        right_index = right.index
    right = right.loc[right_index]
    left_keys, right_keys = _build_equi_keys(left=df,
                                            right=right,
                                            equi_columns=conditions_['equals'])
    indexed_equis = _prepare_equi_lookup(left=df,
                                        right=right,
                                        left_keys=left_keys,
                                        right_keys=right_keys)
    if indexed_equis is None:
        return None
    return df, right, indexed_equis
