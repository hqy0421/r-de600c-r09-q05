"""Pure unit tests for the bitemporal replay helpers (energydb/replay.py).

No database: pins the knowledge-time-first ordering contract, the batch
partitioning semantics, the empty-frame schemas, and the single-series
guard. The ClickHouse SQL itself is covered by the Memory-engine tests in
``test_replay_clickhouse.py`` and the end-to-end PG+CH tests in
``test_replay_integration.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from energydb.errors import ValidationError
from energydb.replay import (
    REPLAY_SCHEMA,
    VERSION_SCHEMA,
    ReplayWindow,
    _window_where,
    empty_replay_frame,
    empty_version_frame,
    partition_by_batch,
    require_single_series,
)

KT_1 = datetime(2026, 1, 1, 6, tzinfo=UTC)
KT_2 = datetime(2026, 1, 1, 7, tzinfo=UTC)
KT_3 = datetime(2026, 1, 1, 8, tzinfo=UTC)
VT_0 = datetime(2026, 1, 1, 0, tzinfo=UTC)
VT_1 = VT_0 + timedelta(hours=1)
VT_2 = VT_0 + timedelta(hours=2)


def _replay_frame(rows: list[tuple[datetime, datetime, float, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema=["knowledge_time", "valid_time", "value", "run_id"],
        orient="row",
    ).cast(REPLAY_SCHEMA)


# ---------------------------------------------------------------------------
# Empty frames
# ---------------------------------------------------------------------------


def test_empty_replay_frame_schema_and_order():
    frame = empty_replay_frame()
    assert frame.columns == list(REPLAY_SCHEMA)
    assert frame.schema == REPLAY_SCHEMA
    assert frame.is_empty()


def test_empty_version_frame_schema_and_order():
    frame = empty_version_frame()
    assert frame.columns == list(VERSION_SCHEMA)
    assert frame.schema == VERSION_SCHEMA
    assert frame.is_empty()


# ---------------------------------------------------------------------------
# partition_by_batch — the knowledge axis must stay the primary ordering
# ---------------------------------------------------------------------------


def test_partition_preserves_arrival_order_even_when_valid_times_interleave():
    # Batch 1 covers vt0..vt2, batch 2 revises only vt1: a valid-time-sorted
    # read would interleave these; the replay partition must keep the two
    # arrivals whole and in knowledge_time order.
    frame = _replay_frame(
        [
            (KT_1, VT_0, 1.0, 10),
            (KT_1, VT_1, 2.0, 10),
            (KT_1, VT_2, 3.0, 10),
            (KT_2, VT_1, 20.0, 20),
        ]
    )
    batches = partition_by_batch(frame)

    assert [kt for kt, _ in batches] == [KT_1, KT_2]

    first_kt, first = batches[0]
    assert first_kt == KT_1
    # The redundant knowledge_time column is dropped inside each batch.
    assert first.columns == ["valid_time", "value", "run_id"]
    assert first.sort("valid_time")["valid_time"].to_list() == [VT_0, VT_1, VT_2]

    second_kt, second = batches[1]
    assert second_kt == KT_2
    assert second["valid_time"].to_list() == [VT_1]
    assert second["value"].to_list() == [20.0]
    assert second["run_id"].to_list() == [20]


def test_partition_sorts_each_batch_by_valid_time():
    # Input deliberately out of valid-time order inside one knowledge_time.
    frame = _replay_frame(
        [
            (KT_1, VT_2, 3.0, 10),
            (KT_1, VT_0, 1.0, 10),
            (KT_1, VT_1, 2.0, 10),
            (KT_2, VT_0, 100.0, 20),
            (KT_2, VT_2, 300.0, 20),
            (KT_2, VT_1, 200.0, 20),
        ]
    )
    batches = partition_by_batch(frame)
    assert [kt for kt, _ in batches] == [KT_1, KT_2]
    for _, sub in batches:
        vts = sub["valid_time"].to_list()
        assert vts == sorted(vts)


def test_partition_empty_frame_returns_empty_list():
    assert partition_by_batch(empty_replay_frame()) == []


def test_same_valid_time_across_batches_is_not_collapsed():
    # The core bitemporal guarantee: one valid_time, several revisions.
    frame = _replay_frame(
        [
            (KT_1, VT_1, 2.0, 10),
            (KT_2, VT_1, 20.0, 20),
            (KT_3, VT_1, 200.0, 30),
        ]
    )
    batches = partition_by_batch(frame)
    assert len(batches) == 3
    assert [sub["value"].item() for _, sub in batches] == [2.0, 20.0, 200.0]
    # Every batch still points at the same valid_time.
    assert all(sub["valid_time"].item() == VT_1 for _, sub in batches)


# ---------------------------------------------------------------------------
# require_single_series
# ---------------------------------------------------------------------------


def _meta(ids: list[int]) -> pl.DataFrame:
    return pl.DataFrame({"series_id": ids, "canonical_unit": ["MW"] * len(ids)})


def test_require_single_series_returns_the_id():
    assert require_single_series(_meta([42])) == 42


def test_require_single_series_dedupes_repeated_id():
    # Resolve meta can carry one logical series on several rows pre-unique.
    assert require_single_series(_meta([7, 7])) == 7


def test_require_single_series_rejects_none_and_empty():
    with pytest.raises(ValidationError, match="exactly one series"):
        require_single_series(None)
    with pytest.raises(ValidationError, match="exactly one series"):
        require_single_series(pl.DataFrame(schema={"series_id": pl.Int64}))


def test_require_single_series_rejects_multiple():
    with pytest.raises(ValidationError, match="single series"):
        require_single_series(_meta([1, 2, 3]))


# ---------------------------------------------------------------------------
# ReplayWindow — the two axes produce independent SQL predicates
# ---------------------------------------------------------------------------


def test_window_where_filters_both_axes_independently():
    where, params = _window_where(
        99,
        ReplayWindow(
            start_valid=VT_0,
            end_valid=VT_2,
            start_known=KT_1,
            end_known=KT_3,
        ),
    )
    assert "series_id = {series_id:UInt64}" in where
    assert "valid_time >= {start_valid" in where
    assert "valid_time < {end_valid" in where
    assert "knowledge_time >= {start_known" in where
    assert "knowledge_time < {end_known" in where
    assert params == {
        "series_id": 99,
        "start_valid": VT_0,
        "end_valid": VT_2,
        "start_known": KT_1,
        "end_known": KT_3,
    }


def test_window_where_series_only_when_unbounded():
    where, params = _window_where(5, ReplayWindow())
    assert where == "WHERE series_id = {series_id:UInt64}"
    assert params == {"series_id": 5}
