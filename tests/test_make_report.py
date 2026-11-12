"""SDK-free tests for make_report.py, covering the intra-op-capped (HC1/HC3)
re-benchmark handling added for the November 2026 AWS re-test:

- the Step 3 primary series switches to the capped phase (HC3) when present and
  falls back to the uncapped phase (HB3) when it is absent;
- worker counts measured on only one side render as blanks in the combined
  table;
- multiple --csv inputs are concatenated into one result set;
- SMOKE / WU rows are excluded from the measured set;
- the regenerated AWS report drops the old, incorrect failure prose
  ("non-deterministic", "24 files").

matplotlib is monkeypatched out so the tests run in a chart-less environment.
"""
import csv
import importlib

import pytest

mr = importlib.import_module("make_report")


CSV_COLUMNS = [
    "run_id", "timestamp", "phase", "step", "env_tag", "accelerator",
    "n_files", "param_name", "param_value", "extra", "wall_s", "sub_timings",
    "throughput", "peak_rss_mb", "peak_gpu_mb", "n_ok", "n_failed", "status",
    "note",
]


def _row(**kw):
    r = {c: "" for c in CSV_COLUMNS}
    r["status"] = "OK"
    r["note"] = ""
    r["extra"] = ""
    r["sub_timings"] = ""
    r.update(kw)
    return r


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _indexing(phase, env_tag, nw, wall, n_files=10715):
    return _row(
        phase=phase, step="indexing", env_tag=env_tag, accelerator="cpu",
        n_files=n_files, param_name="num_workers", param_value=nw,
        wall_s=wall, n_ok=n_files, n_failed=0,
    )


@pytest.fixture(autouse=True)
def _no_charts(monkeypatch):
    """Keep the tests free of matplotlib / rendering."""
    monkeypatch.setattr(mr, "worker_scaling_chart",
                        lambda *a, **k: "", raising=True)
    monkeypatch.setattr(mr, "worker_overlay_chart",
                        lambda *a, **k: "", raising=True)
    monkeypatch.setattr(mr, "cpu_gpu_stacked_chart",
                        lambda *a, **k: "", raising=True)


@pytest.fixture(autouse=True)
def _reset_globals(monkeypatch):
    monkeypatch.setattr(mr, "PNG_DIR", None, raising=True)
    monkeypatch.setattr(mr, "LANG", "en", raising=True)
    yield


def test_step3_primary_series_is_capped_when_hc3_present(tmp_path, monkeypatch):
    csvp = tmp_path / "hc.csv"
    _write_csv(csvp, [
        _indexing("HB3", "cpu1.1", 8, 400.0),
        _indexing("HB3", "cpu1.1", 16, 350.0),
        _indexing("HC3", "cpu1.1", 8, 300.0),
        _indexing("HC3", "cpu1.1", 16, 200.0),
    ])
    monkeypatch.setattr(mr, "CSV_PATHS", [csvp], raising=True)
    rows = mr.load_rows()
    out = "\n".join(mr.worker_scaling_section(
        rows, "indexing", "num_workers", "num_workers scaling", "aws"))
    # The overlay (capped-primary) table has a 'capped (s)' column; the legacy
    # single-series table has a 'parallel eff.' column instead.
    assert "capped (s)" in out
    assert "parallel eff." not in out


def test_step3_falls_back_to_uncapped_when_no_hc3(tmp_path, monkeypatch):
    csvp = tmp_path / "hb.csv"
    _write_csv(csvp, [
        _indexing("HB3", "cpu1.1", 8, 400.0),
        _indexing("HB3", "cpu1.1", 16, 350.0),
    ])
    monkeypatch.setattr(mr, "CSV_PATHS", [csvp], raising=True)
    rows = mr.load_rows()
    out = "\n".join(mr.worker_scaling_section(
        rows, "indexing", "num_workers", "num_workers scaling", "aws"))
    assert "parallel eff." in out
    assert "capped (s)" not in out


def test_worker_only_on_one_side_renders_blank(tmp_path, monkeypatch):
    # num_workers=24 exists only on the uncapped side -> its capped cell blanks.
    csvp = tmp_path / "mix.csv"
    _write_csv(csvp, [
        _indexing("HB3", "cpu1.1", 8, 400.0),
        _indexing("HB3", "cpu1.1", 16, 350.0),
        _indexing("HB3", "cpu1.1", 24, 360.0),
        _indexing("HC3", "cpu1.1", 8, 300.0),
        _indexing("HC3", "cpu1.1", 16, 200.0),
    ])
    monkeypatch.setattr(mr, "CSV_PATHS", [csvp], raising=True)
    rows = mr.load_rows()
    out = "\n".join(mr.worker_scaling_section(
        rows, "indexing", "num_workers", "num_workers scaling", "aws"))
    # The 24-worker row must appear (uncapped 360.0) but with an empty capped
    # cell: the uncapped value is present, the capped time is not.
    assert "360.0" in out
    assert "24" in out


def test_multiple_csv_inputs_are_concatenated(tmp_path, monkeypatch):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a, [_indexing("HB3", "cpu1.1", 8, 400.0)])
    _write_csv(b, [_indexing("HC3", "cpu1.1", 8, 300.0)])
    monkeypatch.setattr(mr, "CSV_PATHS", [a, b], raising=True)
    rows = mr.load_rows()
    phases = sorted({r["phase"] for r in rows})
    assert phases == ["HB3", "HC3"]


def test_smoke_and_warmup_rows_are_excluded(tmp_path, monkeypatch):
    csvp = tmp_path / "noise.csv"
    _write_csv(csvp, [
        _indexing("HC3", "cpu1.1", 8, 300.0),
        _indexing("SMOKE", "cpu1.1", 1, 10.0),
        _indexing("WU", "cpu1.1", 8, 20.0),
        _row(phase="HC3", step="indexing", env_tag="cpu1.1", accelerator="cpu",
             n_files=100, param_name="num_workers", param_value=8, wall_s=5.0,
             n_ok=100, n_failed=0, note="warm-up pass"),
    ])
    monkeypatch.setattr(mr, "CSV_PATHS", [csvp], raising=True)
    rows = mr.load_rows()
    phases = {r["phase"] for r in rows}
    assert phases == {"HC3"}
    assert all(r["param_value"] not in ("1",) for r in rows)
    # The warm-up-noted HC3 row is dropped too.
    assert all("warm-up" not in (r.get("note") or "").lower() for r in rows)


def _full_aws_rows():
    rows = []
    # Step 1 encode: uncapped HB1 + capped HC1 (cpu + gpu).
    for nw, w in ((16, 1912.0), (24, 1733.0), (32, 1653.0)):
        rows.append(_row(phase="HC1", step="dataprep", env_tag="cpu1.1",
                         accelerator="cpu", n_files=10715,
                         param_name="max_workers", param_value=nw, wall_s=w,
                         sub_timings='{"EncodingTask": %.1f}' % (w - 200),
                         n_ok=10499, n_failed=216))
    for nw, w in ((16, 1800.0), (24, 1750.0), (32, 1718.0)):
        rows.append(_row(phase="HB1", step="dataprep", env_tag="cpu1.1",
                         accelerator="cpu", n_files=10715,
                         param_name="max_workers", param_value=nw, wall_s=w,
                         sub_timings='{"EncodingTask": %.1f}' % (w - 200),
                         n_ok=10499, n_failed=216))
    rows.append(_row(phase="HC1", step="dataprep", env_tag="gpu1.1",
                     accelerator="gpu", n_files=10715,
                     param_name="max_workers", param_value=32, wall_s=1679.0,
                     sub_timings='{"EncodingTask": 1479.0}',
                     n_ok=10499, n_failed=216))
    # Step 2 training HB2 cpu + gpu.
    rows.append(_row(phase="HB2", step="training", env_tag="cpu1.1",
                     accelerator="cpu", n_files=10715, param_name="batch_size",
                     param_value=64, wall_s=4972.0,
                     sub_timings='{"s_per_epoch": 497.0}', n_ok=10499,
                     n_failed=0, peak_rss_mb=20000))
    rows.append(_row(phase="HB2", step="training", env_tag="gpu1.1",
                     accelerator="gpu", n_files=10715, param_name="batch_size",
                     param_value=64, wall_s=1010.0,
                     sub_timings='{"s_per_epoch": 101.0}', n_ok=10499,
                     n_failed=0, peak_rss_mb=15000))
    # Step 3 indexing: uncapped HB3 + capped HC3 (cpu + gpu).
    for nw, w in ((8, 3486.0), (12, 3377.0), (16, 4124.0)):
        rows.append(_indexing("HB3", "cpu1.1", nw, w))
    for nw, w in ((8, 3163.0), (16, 1797.0), (32, 1504.0)):
        rows.append(_indexing("HC3", "cpu1.1", nw, w))
    rows.append(_indexing("HC3", "gpu1.1", 32, 1528.0))
    return rows


def test_aws_report_drops_incorrect_failure_prose(tmp_path, monkeypatch):
    csvp = tmp_path / "aws.csv"
    _write_csv(csvp, _full_aws_rows())
    monkeypatch.setattr(mr, "CSV_PATHS", [csvp], raising=True)
    rows = mr.load_rows()
    all_rows = mr.load_all_rows()
    md = "\n".join(mr.build_doc(rows, all_rows, "aws"))
    assert "non-deterministic" not in md
    assert "24 files" not in md
    # The corrected, deterministic narrative is present instead.
    assert "216" in md
    assert "10,499" in md
