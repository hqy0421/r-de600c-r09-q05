"""ClickHouse SQL-semantics tests for the replay queries.

The shared R09 ClickHouse has a read-only MergeTree store (filesystem
permission on part renames), so these tests build a **Memory**-engine table
with the same columns and ORDER BY shape as timedb's ``series_values`` and
point the replay module at it by monkeypatching the table name in the SQL
constants. Memory inserts need no disk, so the knowledge-first ordering,
the LIMIT 1 BY correction rule, the two-axis window predicates, the
NaN-as-null sentinel, and the version-listing aggregation all run against a
real ClickHouse server.

Skipped when TIMEDB_CH_URL is unset. The table is created and dropped per
module and never touches ``series_values``.
"""

from __future__ import annotations

import os

import polars as pl
import pytest

from energydb import replay as replay_mod

if not os.environ.get("TIMEDB_CH_URL"):
    pytest.skip("TIMEDB_CH_URL not set: skipping ClickHouse replay tests", allow_module_level=True)

from datetime import UTC, datetime, timedelta

import clickhouse_connect

_TABLE = "_replay_probe"

# Same columns as timedb's series_values DDL; Memory instead of MergeTree.
_DDL = f"""
CREATE TABLE {_TABLE} (
    series_id      UInt64,
    valid_time     DateTime64(6, 'UTC'),
    knowledge_time DateTime64(6, 'UTC'),
    change_time    DateTime64(6, 'UTC'),
    value          Float64,
    valid_time_end DateTime64(6, 'UTC') DEFAULT toDateTime64('2200-01-01 00:00:00', 6, 'UTC'),
    run_id         UInt64,
    changed_by     LowCardinality(String),
    annotation     String,
    retention      LowCardinality(String)
) ENGINE = Memory
"""

KT_1 = datetime(2026, 1, 1, 6, tzinfo=UTC)
KT_2 = datetime(2026, 1, 1, 7, tzinfo=UTC)
KT_3 = datetime(2026, 1, 1, 8, tzinfo=UTC)
VT_0 = datetime(2026, 1, 1, 0, tzinfo=UTC)
VT_1 = VT_0 + timedelta(hours=1)
VT_2 = VT_0 + timedelta(hours=2)
SID = 101
OTHER_SID = 202


@pytest.fixture(scope="module")
def ch():
    client = clickhouse_connect.get_client(dsn=os.environ["TIMEDB_CH_URL"])
    client.command(f"DROP TABLE IF EXISTS {_TABLE}")
    client.command(_DDL)
    yield client
    client.command(f"DROP TABLE IF EXISTS {_TABLE}")


def _row(sid, kt, vt, value, run_id, *, change_time=None, nan=False):
    return [
        sid,
        vt,
        kt,
        change_time or kt,
        float("nan") if nan else value,
        datetime(2200, 1, 1, tzinfo=UTC),
        run_id,
        "",
        "",
        "medium",
    ]


_COLUMNS = [
    "series_id",
    "valid_time",
    "knowledge_time",
    "change_time",
    "value",
    "valid_time_end",
    "run_id",
    "changed_by",
    "annotation",
    "retention",
]


@pytest.fixture
def seeded(ch):
    rows = [
        # Batch 1 (kt1, run 10): three valid-time points.
        _row(SID, KT_1, VT_0, 1.0, 10),
        _row(SID, KT_1, VT_1, 2.0, 10),
        _row(SID, KT_1, VT_2, 3.0, 10),
        # Batch 2 (kt2, run 20): revises vt1 only.
        _row(SID, KT_2, VT_1, 20.0, 20),
        # Batch 3 (kt3, run 30): two corrections of vt2 within ONE batch —
        # the largest change_time must win (LIMIT 1 BY rule).
        _row(SID, KT_3, VT_2, 300.0, 30, change_time=KT_3 + timedelta(minutes=1)),
        _row(SID, KT_3, VT_2, 301.0, 30, change_time=KT_3 + timedelta(minutes=2)),
        # A null value (NaN storage sentinel) for vt0 in batch 3.
        _row(SID, KT_3, VT_0, None, 30, nan=True),
        # A second series that every query must exclude.
        _row(OTHER_SID, KT_1, VT_0, 999.0, 99),
    ]
    ch.insert(_TABLE, rows, column_names=_COLUMNS)
    yield
    ch.command(f"TRUNCATE TABLE {_TABLE}")


def test_replay_orders_by_knowledge_time_then_valid_time(ch, seeded):
    frame = replay_mod.query_replay(ch, SID, replay_mod.ReplayWindow(), table=_TABLE)

    assert frame.columns == ["series_id", "knowledge_time", "valid_time", "value", "run_id"]
    # Arrival axis first — NOT the (valid_time, knowledge_time) order the
    # ordinary overlapping read returns.
    assert frame["knowledge_time"].to_list() == [KT_1, KT_1, KT_1, KT_2, KT_3, KT_3]
    # Within each batch the subject axis is sorted.
    kt1 = frame.filter(pl.col("knowledge_time") == KT_1)
    assert kt1["valid_time"].to_list() == [VT_0, VT_1, VT_2]
    # The other series never leaks in.
    assert frame["series_id"].unique().to_list() == [SID]


def test_replay_latest_correction_wins_within_batch(ch, seeded):
    frame = replay_mod.query_replay(ch, SID, replay_mod.ReplayWindow(), table=_TABLE)
    kt3_vt2 = frame.filter((pl.col("knowledge_time") == KT_3) & (pl.col("valid_time") == VT_2))
    assert kt3_vt2.height == 1
    assert kt3_vt2["value"].item() == 301.0  # the 08:02 correction, not 08:01
    assert kt3_vt2["run_id"].item() == 30


def test_replay_same_valid_time_appears_once_per_batch(ch, seeded):
    frame = replay_mod.query_replay(ch, SID, replay_mod.ReplayWindow(), table=_TABLE)
    vt2 = frame.filter(pl.col("valid_time") == VT_2).sort("knowledge_time")
    # vt2 was covered by batch 1 (value 3) and batch 3 (value 301): two rows,
    # two arrivals, never collapsed.
    assert vt2.height == 2
    assert vt2["knowledge_time"].to_list() == [KT_1, KT_3]
    assert vt2["value"].to_list() == [3.0, 301.0]


def test_replay_nan_sentinel_becomes_null(ch, seeded):
    frame = replay_mod.query_replay(ch, SID, replay_mod.ReplayWindow(), table=_TABLE)
    null_row = frame.filter((pl.col("knowledge_time") == KT_3) & (pl.col("valid_time") == VT_0))
    assert null_row.height == 1
    assert null_row["value"].item() is None


def test_replay_valid_window_filters_subject_axis_only(ch, seeded):
    # Restrict to vt1: batches still arrive in knowledge order, but only their
    # vt1 rows come back.
    frame = replay_mod.query_replay(
        ch, SID, replay_mod.ReplayWindow(start_valid=VT_1, end_valid=VT_2),
        table=_TABLE,
    )
    assert frame["valid_time"].unique().to_list() == [VT_1]
    # vt1 was revised in kt1 and kt2.
    assert frame["knowledge_time"].to_list() == [KT_1, KT_2]
    assert frame["value"].to_list() == [2.0, 20.0]


def test_replay_known_window_filters_arrival_axis_only(ch, seeded):
    # Restrict the arrivals to [kt2, kt3): only batch 2's rows.
    frame = replay_mod.query_replay(
        ch, SID, replay_mod.ReplayWindow(start_known=KT_2, end_known=KT_3),
        table=_TABLE,
    )
    assert frame["knowledge_time"].unique().to_list() == [KT_2]
    assert frame["valid_time"].to_list() == [VT_1]


def test_replay_empty_window_returns_typed_empty_frame(ch, seeded):
    frame = replay_mod.query_replay(
        ch,
        SID,
        replay_mod.ReplayWindow(start_known=datetime(2030, 1, 1, tzinfo=UTC)),
        table=_TABLE,
    )
    assert frame.is_empty()
    assert frame.columns == ["series_id", "knowledge_time", "valid_time", "value", "run_id"]


def test_versions_lists_distinct_arrivals_oldest_first(ch, seeded):
    frame = replay_mod.query_versions(ch, SID, replay_mod.ReplayWindow(), table=_TABLE)
    assert frame.columns == ["knowledge_time", "run_id", "n_valid_times"]
    assert frame["knowledge_time"].to_list() == [KT_1, KT_2, KT_3]
    assert frame["run_id"].to_list() == [10, 20, 30]
    # Distinct valid-time points covered by each batch (the two kt3
    # corrections of vt2 count once).
    counts = dict(zip(frame["knowledge_time"].to_list(), frame["n_valid_times"].to_list(), strict=True))
    assert counts == {KT_1: 3, KT_2: 1, KT_3: 2}


def test_versions_respects_valid_window(ch, seeded):
    frame = replay_mod.query_versions(
        ch, SID, replay_mod.ReplayWindow(start_valid=VT_1, end_valid=VT_2),
        table=_TABLE,
    )
    # Only vt1 in window: batches kt1 and kt2 cover it; kt3 does not.
    assert frame["knowledge_time"].to_list() == [KT_1, KT_2]
    assert frame["n_valid_times"].to_list() == [1, 1]


def test_versions_empty_for_unknown_series(ch, seeded):
    frame = replay_mod.query_versions(ch, 999999, replay_mod.ReplayWindow(), table=_TABLE)
    assert frame.is_empty()
    assert frame.columns == ["knowledge_time", "run_id", "n_valid_times"]
