"""End-to-end replay tests through the public scope API.

These exercise the full energydb stack for the replay feature: real
PostgreSQL series resolution (scope → ``series`` catalog → series_id) and
the real replay queries, including single-series validation, unit
conversion, and batch partitioning.

The shared R09 ClickHouse store rejects MergeTree inserts (read-only part
directory), so instead of writing through ``td.write`` the tests mirror the
written rows into a **Memory**-engine table carrying the exact
``series_values`` columns and redirect only the two replay SQL constants at
it. Everything above the value table — registration, resolution, the scope
methods themselves — is the production code path. On a deployment with a
writable MergeTree store the same tests work against the real table by
dropping the monkeypatch.

Skipped when TIMEDB_PG_DSN / TIMEDB_CH_URL are unset.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

import energydb as edb
from energydb import Client
from energydb import replay as replay_mod
from energydb.errors import ValidationError

if not (os.environ.get("TIMEDB_PG_DSN") and os.environ.get("TIMEDB_CH_URL")):
    pytest.skip(
        "TIMEDB_PG_DSN / TIMEDB_CH_URL not set: skipping replay integration tests",
        allow_module_level=True,
    )

import clickhouse_connect

_TABLE = "_replay_e2e_probe"

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

KT_1 = datetime(2026, 1, 1, 6, tzinfo=UTC)
KT_2 = datetime(2026, 1, 1, 7, tzinfo=UTC)
KT_3 = datetime(2026, 1, 1, 8, tzinfo=UTC)
VT_0 = datetime(2026, 1, 1, 0, tzinfo=UTC)
VT_1 = VT_0 + timedelta(hours=1)
VT_2 = VT_0 + timedelta(hours=2)


@pytest.fixture
def stack(monkeypatch):
    client = Client()
    client.delete()
    client.create()

    root = edb.Portfolio(
        name="rp",
        members=[
            edb.wind.WindTurbine(name="T1", capacity=3.5),
            edb.wind.WindTurbine(name="T2", capacity=3.5),
        ],
    )
    client.register_tree(root)
    t1 = client.get_node("rp", "T1")
    sid = t1.register_series(
        name="power",
        canonical_unit="MW",
        data_type="forecast",
        timeseries_type="OVERLAPPING",
        retention="medium",
    )
    # A second series on the same subtree to exercise the single-series guard.
    t1.register_series(
        name="wind",
        canonical_unit="m/s",
        data_type="forecast",
        timeseries_type="OVERLAPPING",
        retention="medium",
    )

    ch = clickhouse_connect.get_client(dsn=os.environ["TIMEDB_CH_URL"])
    ch.command(f"DROP TABLE IF EXISTS {_TABLE}")
    ch.command(_DDL)

    # Redirect only the value-table read at the wrapper boundary; the SQL
    # templates and every layer above (resolve, scope, framing) are untouched.
    # Patching the symbols scope.py imported binds the patch to this test's
    # monkeypatch context, so no module-global state leaks across the suite.
    async def _replay_probe(td, series_id, window, *, table=_TABLE):
        return await replay_mod.replay_series(td, series_id, window, table=table)

    async def _versions_probe(td, series_id, window, *, table=_TABLE):
        return await replay_mod.versions_for_series(td, series_id, window, table=table)

    monkeypatch.setattr("energydb.scope.replay_series", _replay_probe)
    monkeypatch.setattr("energydb.scope.versions_for_series", _versions_probe)

    def seed(rows):
        ch.insert(_TABLE, rows, column_names=_COLUMNS)

    yield client, t1, sid, seed, ch

    ch.command(f"DROP TABLE IF EXISTS {_TABLE}")
    client.delete()
    client.close()


def _row(sid, kt, vt, value, run_id, *, change_time=None):
    return [
        sid,
        vt,
        kt,
        change_time or kt,
        value,
        datetime(2200, 1, 1, tzinfo=UTC),
        run_id,
        "",
        "",
        "medium",
    ]


def _three_batches(sid):
    return [
        _row(sid, KT_1, VT_0, 1.0, 10),
        _row(sid, KT_1, VT_1, 2.0, 10),
        _row(sid, KT_1, VT_2, 3.0, 10),
        _row(sid, KT_2, VT_1, 20.0, 20),
        _row(sid, KT_3, VT_2, 30.0, 30),
        _row(sid, KT_3, VT_2, 31.0, 30, change_time=KT_3 + timedelta(minutes=1)),
    ]


def test_replay_frame_is_knowledge_ordered_and_identity_free(stack):
    _client, t1, sid, seed, _ = stack
    seed(_three_batches(sid))

    frame = t1.replay(data_type="forecast", name="power")

    assert frame.columns == ["knowledge_time", "valid_time", "value", "run_id"]
    # Arrival axis first; the ordinary read would return valid_time ordering.
    assert frame["knowledge_time"].to_list() == [KT_1, KT_1, KT_1, KT_2, KT_3]
    # Batch 3's correction: latest change_time wins, one row per (kt, vt).
    last = frame.filter(pl.col("knowledge_time") == KT_3)
    assert last["value"].to_list() == [31.0]
    # No internal identifiers leak onto the public frame.
    assert "series_id" not in frame.columns


def test_replay_two_axes_filter_independently(stack):
    _client, t1, sid, seed, _ = stack
    seed(_three_batches(sid))

    # Subject-axis window: only vt1 rows, but both batches that covered it.
    by_valid = t1.replay(
        data_type="forecast", name="power", start_valid=VT_1, end_valid=VT_2
    )
    assert by_valid["valid_time"].unique().to_list() == [VT_1]
    assert by_valid["knowledge_time"].to_list() == [KT_1, KT_2]

    # Arrival-axis window: only batch 2, whatever valid times it held.
    by_known = t1.replay(
        data_type="forecast", name="power", start_known=KT_2, end_known=KT_3
    )
    assert by_known["knowledge_time"].unique().to_list() == [KT_2]
    assert by_known["valid_time"].to_list() == [VT_1]


def test_replay_as_batches_replays_arrival_by_arrival(stack):
    _client, t1, sid, seed, _ = stack
    seed(_three_batches(sid))

    batches = t1.replay(data_type="forecast", name="power", as_batches=True)

    assert [kt for kt, _ in batches] == [KT_1, KT_2, KT_3]
    for _, sub in batches:
        assert sub.columns == ["valid_time", "value", "run_id"]
        assert sub["valid_time"].to_list() == sorted(sub["valid_time"].to_list())
    # Batch 1 carried three points; batch 2 revised one; batch 3 one.
    assert [sub.height for _, sub in batches] == [3, 1, 1]
    assert batches[1][1]["value"].to_list() == [20.0]


def test_replay_pandas_backend_converts_frame_and_batches(stack):
    _client, t1, sid, seed, _ = stack
    seed(_three_batches(sid))

    import pandas as pd

    frame = t1.replay(data_type="forecast", name="power", backend="pandas")
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == ["knowledge_time", "valid_time", "value", "run_id"]

    batches = t1.replay(
        data_type="forecast", name="power", as_batches=True, backend="pandas"
    )
    assert all(isinstance(sub, pd.DataFrame) for _, sub in batches)


def test_replay_versions_lists_batches_with_coverage(stack):
    _client, t1, sid, seed, _ = stack
    seed(_three_batches(sid))

    versions = t1.replay_versions(data_type="forecast", name="power")
    assert versions.columns == ["knowledge_time", "run_id", "n_valid_times"]
    assert versions["knowledge_time"].to_list() == [KT_1, KT_2, KT_3]
    assert versions["run_id"].to_list() == [10, 20, 30]
    assert versions["n_valid_times"].to_list() == [3, 1, 1]


def test_replay_unknown_series_raises(stack):
    _client, t1, sid, seed, _ = stack
    seed(_three_batches(sid))
    with pytest.raises(ValidationError, match="exactly one series"):
        t1.replay(data_type="forecast", name="does-not-exist")


def test_replay_ambiguous_subtree_requires_narrowing(stack):
    _client, _t1, sid, seed, _ = stack
    seed(_three_batches(sid))
    # No data_type/name narrowing would resolve the whole subtree; here we
    # prove the guard by handing the method a two-series meta directly.
    two_series = pl.DataFrame(
        {"series_id": [1, 2], "canonical_unit": ["MW", "m/s"]}
    )
    with pytest.raises(ValidationError, match="single series"):
        replay_mod.require_single_series(two_series)


def test_replay_empty_result_has_typed_schemas(stack):
    _client, t1, _sid, _seed, _ = stack
    # Series registered, no rows.
    frame = t1.replay(data_type="forecast", name="power")
    assert frame.is_empty()
    assert frame.columns == ["knowledge_time", "valid_time", "value", "run_id"]

    versions = t1.replay_versions(data_type="forecast", name="power")
    assert versions.is_empty()
    assert versions.columns == ["knowledge_time", "run_id", "n_valid_times"]

    assert t1.replay(data_type="forecast", name="power", as_batches=True) == []


def test_replay_unit_conversion_rescales_values(stack):
    _client, t1, sid, seed, _ = stack
    seed(_three_batches(sid))
    # MW -> kW is a fixed x1000 factor; values scale, time columns do not.
    frame = t1.replay(data_type="forecast", name="power", unit="kW")
    base = t1.replay(data_type="forecast", name="power")
    assert frame["value"].to_list() == [v * 1000.0 for v in base["value"].to_list()]
    assert frame["knowledge_time"].to_list() == base["knowledge_time"].to_list()


def test_replay_works_on_edge_scope(stack):
    from energydatamodel.reference import Reference

    client, _t1, _sid, seed, _ = stack
    # Build an edge with its own OVERLAPPING series and replay it.
    t3 = edb.wind.WindTurbine(name="T3", capacity=2.0)
    t4 = edb.wind.WindTurbine(name="T4", capacity=2.0)
    client.register_tree(edb.Portfolio(name="rp2", members=[t3, t4]))
    client.create_edge(
        edb.grid.Line(
            from_element=Reference(t3),
            to_element=Reference(t4),
        )
    )
    edge_scope = client.get_edge("rp2/T3", "rp2/T4", type="Line")
    edge_sid = edge_scope.register_series(
        name="flow",
        canonical_unit="MW",
        data_type="forecast",
        timeseries_type="OVERLAPPING",
        retention="medium",
    )
    seed([_row(edge_sid, KT_1, VT_0, 7.0, 70), _row(edge_sid, KT_2, VT_0, 8.0, 71)])

    frame = edge_scope.replay(data_type="forecast", name="flow")
    assert frame["knowledge_time"].to_list() == [KT_1, KT_2]
    assert frame["value"].to_list() == [7.0, 8.0]
    assert frame["run_id"].to_list() == [70, 71]
