"""Bitemporal replay: the knowledge-time-ordered version stream of one series.

The ordinary read path (:func:`energydb._io.execute_read`) answers
"what is the value for valid time *t*?" and returns rows ordered by
``valid_time``. Replay answers a different question: "in what order did
forecast revisions actually **arrive**, and what did each arrival say
about each valid time?" It therefore orders by ``knowledge_time`` first.

The two time axes are deliberately never merged into one timeline:

* ``knowledge_time`` is the *arrival* axis — distinct knowledge_times are
  the forecast batches (revisions) as they were issued.
* ``valid_time`` is the *subject* axis — the real-world moments the
  forecasts describe. Windows along it (``start_valid`` / ``end_valid``)
  only filter which subjects each batch's rows cover.

The same ``valid_time`` therefore legitimately appears once per batch that
covered it, each row carrying that batch's ``knowledge_time`` (and
``run_id``); nothing collapses or compares them across batches. That is the
property a revision analysis needs: flattening onto one axis would destroy
either the arrival order or the valid-time identity of the points.

The queries here read the same ``series_values`` table as timedb but issue
their own SQL, because timedb 0.12's overlapping read orders by
``(series_id, valid_time, knowledge_time)`` — the subject axis first — and
exposes no knowledge-first ordering.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from typing import NamedTuple

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
from timedb import profiling

from energydb.errors import ValidationError

# Timestamps are stored DateTime64(6, 'UTC'); mirror timedb's read types so a
# replay frame is byte-compatible with an ordinary read frame where the
# columns overlap.
_TS_TYPE = pa.timestamp("us", tz="UTC")
_REPLAY_ARROW_TYPES: dict[str, pa.DataType] = {
    "series_id": pa.uint64(),
    "knowledge_time": _TS_TYPE,
    "valid_time": _TS_TYPE,
    "value": pa.float64(),
    "run_id": pa.uint64(),
    "n_valid_times": pa.uint32(),
}

# Public polars schema of a replay frame (the internal series_id is dropped
# before the result leaves the scope layer).
REPLAY_COLUMNS: tuple[str, ...] = ("knowledge_time", "valid_time", "value", "run_id")
REPLAY_SCHEMA: dict[str, pl.DataType] = {
    "knowledge_time": pl.Datetime("us", "UTC"),
    "valid_time": pl.Datetime("us", "UTC"),
    "value": pl.Float64,
    "run_id": pl.UInt64,
}

# A distinct-knowledge_times listing.
VERSION_COLUMNS: tuple[str, ...] = ("knowledge_time", "run_id", "n_valid_times")
VERSION_SCHEMA: dict[str, pl.DataType] = {
    "knowledge_time": pl.Datetime("us", "UTC"),
    "run_id": pl.UInt64,
    "n_valid_times": pl.UInt32,
}


class ReplayWindow(NamedTuple):
    """The independent window filters for a replay read.

    The two bounds belong to different axes and are applied as such in SQL:
    ``start_known`` / ``end_known`` constrain the arrival axis
    (``knowledge_time``), ``start_valid`` / ``end_valid`` the subject axis
    (``valid_time``). They never substitute for one another.
    """

    start_valid: datetime | None = None
    end_valid: datetime | None = None
    start_known: datetime | None = None
    end_known: datetime | None = None


def empty_replay_frame() -> pl.DataFrame:
    """A zero-row replay frame with the public :data:`REPLAY_SCHEMA`."""
    return pl.DataFrame(schema=REPLAY_SCHEMA)


def empty_version_frame() -> pl.DataFrame:
    """A zero-row version-listing frame with the public :data:`VERSION_SCHEMA`."""
    return pl.DataFrame(schema=VERSION_SCHEMA)


def _window_where(
    series_id: int,
    window: ReplayWindow,
) -> tuple[str, dict]:
    """Build the parameterized WHERE clause for one series' replay.

    The ``series_id`` equality pins the single series; the four optional
    bounds filter the two axes independently (half-open intervals, matching
    timedb's ``>= start`` / ``< end`` convention).
    """
    conds = ["series_id = {series_id:UInt64}"]
    params: dict = {"series_id": int(series_id)}
    if window.start_valid is not None:
        conds.append("valid_time >= {start_valid:DateTime64(6, 'UTC')}")
        params["start_valid"] = window.start_valid
    if window.end_valid is not None:
        conds.append("valid_time < {end_valid:DateTime64(6, 'UTC')}")
        params["end_valid"] = window.end_valid
    if window.start_known is not None:
        conds.append("knowledge_time >= {start_known:DateTime64(6, 'UTC')}")
        params["start_known"] = window.start_known
    if window.end_known is not None:
        conds.append("knowledge_time < {end_known:DateTime64(6, 'UTC')}")
        params["end_known"] = window.end_known
    return "WHERE " + " AND ".join(conds), params


# The ClickHouse value table (timedb's DDL). A parameter rather than something
# tests monkeypatch onto the module, so the query functions stay pure with
# respect to global state.
DEFAULT_TABLE = "series_values"

# ClickHouse binds values, not identifiers, so the table name is interpolated.
# Restricting it to bare identifier characters makes that interpolation safe;
# every real table name here (series_values, a Memory-engine probe) matches.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _check_table_name(table: str) -> str:
    if not isinstance(table, str) or not _IDENTIFIER_RE.match(table):
        raise ValidationError(f"Invalid ClickHouse table name {table!r}: expected a bare identifier.")
    return table


# One row per (knowledge_time, valid_time): the latest correction within the
# batch wins. This mirrors timedb._read_overlapping's LIMIT 1 BY pass but with
# the axes swapped: the input ORDER BY puts the arrival (knowledge_time) axis
# first — the ordering the whole module exists to give callers — and the
# subject (valid_time) axis second, so a batch's rows read in valid-time order.
# Reversing change_time makes the largest-change_time row the first seen per
# (sid, kt, vt) group, which LIMIT 1 BY keeps; equivalent to
# argMax(value, change_time) GROUP BY (sid, kt, vt) without aggregation state.
# The table sort key is (sid, vt, kt, ct), so CH streams one series' slice and
# does a bounded re-sort, not a full-table sort.
_REPLAY_SQL = """
SELECT series_id, knowledge_time, valid_time, value, run_id
FROM {table}
{where}
ORDER BY series_id, knowledge_time, valid_time, change_time DESC
LIMIT 1 BY series_id, knowledge_time, valid_time
"""

# The batch index: one row per distinct arrival (knowledge_time). run_id is
# aggregated alongside so callers can join back to the energydb.runs catalog
# without re-reading values; a knowledge_time written by several runs keeps
# the latest one (any() inside argMax would be nondeterministic).
_VERSIONS_SQL = """
SELECT
    knowledge_time,
    argMax(run_id, change_time) AS run_id,
    toUInt32(count(DISTINCT valid_time)) AS n_valid_times
FROM {table}
{where}
GROUP BY knowledge_time
ORDER BY knowledge_time ASC
"""


def _mask_nan_nulls(table: pa.Table, value_idx: int) -> pa.Table:
    """Rebuild the ``value`` column turning stored NaN sentinels into nulls.

    Identical convention to timedb's read path (NaN is the storage sentinel
    for null); only rebuilds when NaNs are actually present.
    """
    col = table.column(value_idx)
    if len(col) and pc.any(pc.is_nan(col)).as_py():
        arr = col.to_numpy(zero_copy_only=False)
        table = table.set_column(value_idx, "value", pa.array(arr, mask=np.isnan(arr)))
    return table


def _query_arrow(ch_client, sql: str, params: dict, cols: list[str]) -> pa.Table:
    """Run a replay query and return a typed, NaN-masked Arrow table."""
    _prof = profiling._enabled
    _t = _time_perf_counter(_prof)
    result = ch_client.query_arrow(sql, parameters=params)
    if _prof:
        profiling._record(profiling.PHASE_READ_SQL_EXEC, _time_perf_counter(_prof) - _t)

    _t = _time_perf_counter(_prof)
    table = (
        pa.table({c: pa.array([], type=_REPLAY_ARROW_TYPES[c]) for c in cols})
        if result.num_rows == 0
        else result.select(cols)
    )
    if "value" in table.schema.names:
        table = _mask_nan_nulls(table, table.schema.get_field_index("value"))
    if _prof:
        profiling._record(profiling.PHASE_READ_BUILD_ARROW, _time_perf_counter(_prof) - _t)
    return table


def _time_perf_counter(enabled: bool) -> float:
    """perf_counter read only when profiling is on (zero overhead otherwise)."""
    import time

    return time.perf_counter() if enabled else 0.0


def _to_polars(table: pa.Table) -> pl.DataFrame:
    _prof = profiling._enabled
    _t = _time_perf_counter(_prof)
    result = pl.from_arrow(table)
    if _prof:
        profiling._record(profiling.PHASE_READ_TO_POLARS, _time_perf_counter(_prof) - _t)
    assert isinstance(result, pl.DataFrame)
    return result


def query_replay(
    ch_client,
    series_id: int,
    window: ReplayWindow,
    *,
    table: str = DEFAULT_TABLE,
) -> pl.DataFrame:
    """Read one series' knowledge-time-ordered version stream from ClickHouse.

    Returns the internal frame carrying ``series_id``; the scope layer drops
    that column. Rows are ordered ``(knowledge_time ASC, valid_time ASC)``:
    arrival batches first, and within a batch the valid-time points in order.
    One row per ``(knowledge_time, valid_time)`` — the latest correction
    (largest ``change_time``) within the batch wins, exactly like timedb's
    overlapping read.
    """
    _check_table_name(table)
    where, params = _window_where(series_id, window)
    arrow = _query_arrow(
        ch_client,
        _REPLAY_SQL.format(table=table, where=where),
        params,
        ["series_id", "knowledge_time", "valid_time", "value", "run_id"],
    )
    return _to_polars(arrow)


def query_versions(
    ch_client,
    series_id: int,
    window: ReplayWindow,
    *,
    table: str = DEFAULT_TABLE,
) -> pl.DataFrame:
    """List one series' distinct arrivals (knowledge_times), oldest first.

    One row per distinct ``knowledge_time`` with the ``run_id`` of the write
    that last touched it (joinable to the PG ``runs`` catalog) and the number
    of distinct valid times covered by that batch within the window.
    """
    _check_table_name(table)
    where, params = _window_where(series_id, window)
    arrow = _query_arrow(
        ch_client,
        _VERSIONS_SQL.format(table=table, where=where),
        params,
        ["knowledge_time", "run_id", "n_valid_times"],
    )
    return _to_polars(arrow)


async def replay_series(
    td,
    series_id: int,
    window: ReplayWindow,
    *,
    table: str = DEFAULT_TABLE,
) -> pl.DataFrame:
    """Async wrapper: run :func:`query_replay` on a worker thread.

    timedb's ClickHouse client is synchronous, so every value read in the
    codebase is offloaded this way to keep the event loop free.
    """
    return await asyncio.to_thread(query_replay, td._ch, series_id, window, table=table)


async def versions_for_series(
    td,
    series_id: int,
    window: ReplayWindow,
    *,
    table: str = DEFAULT_TABLE,
) -> pl.DataFrame:
    """Async wrapper: run :func:`query_versions` on a worker thread."""
    return await asyncio.to_thread(query_versions, td._ch, series_id, window, table=table)


def partition_by_batch(frame: pl.DataFrame) -> list[tuple[datetime, pl.DataFrame]]:
    """Split a replay frame into one sub-frame per arrival, preserving order.

    Returns ``[(knowledge_time, batch_frame), ...]`` in arrival order
    (oldest batch first). Each ``batch_frame`` keeps the replay schema minus
    the now-redundant ``knowledge_time`` column — i.e.
    ``(valid_time, value, run_id)`` sorted by ``valid_time`` — so iterating
    the result replays batch by batch. A pure-polars helper (no I/O), which
    makes the batching semantics unit-testable without a database.
    """
    if frame.is_empty():
        return []
    parts = frame.partition_by("knowledge_time", as_dict=True, maintain_order=True)
    out: list[tuple[datetime, pl.DataFrame]] = []
    for key_tuple, sub in parts.items():
        knowledge_time = key_tuple[0]
        out.append(
            (
                knowledge_time,
                sub.drop("knowledge_time").sort("valid_time"),
            )
        )
    return out


def require_single_series(meta: pl.DataFrame | None) -> int:
    """Resolve a scope's meta frame to exactly one ``series_id`` or raise.

    Replay is defined for one series' version stream. A scope that resolves
    to no series returns an empty stream; one that resolves to several is a
    caller mistake (narrow with ``data_type=`` / ``name=``), since silently
    interleaving two series' knowledge streams would be meaningless.
    """
    if meta is None or meta.is_empty() or "series_id" not in meta.columns:
        raise ValidationError(
            "replay() needs exactly one series, but the scope resolved to none. "
            "Register the series or narrow the scope."
        )
    ids = meta["series_id"].unique().to_list()
    if len(ids) > 1:
        raise ValidationError(
            f"replay() is defined for a single series, but the scope resolved to {len(ids)} "
            "series. Narrow it with data_type= and name= (or navigate to one node/edge)."
        )
    return int(ids[0])
