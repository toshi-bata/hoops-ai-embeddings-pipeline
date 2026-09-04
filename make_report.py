"""Turn results/results.csv into REPORT.md + REPORT.html.

Deliberately dependency-free (csv + json + stdlib only) so it runs in either
venv, and readable-by-hand output so the numbers can be checked against the
raw logs.

Analyses produced:
  * step 1 / step 3: wall time vs worker count, speedup, parallel efficiency,
    and an Amdahl serial-fraction estimate from the two best-separated points.
  * step 2: s/epoch for every (env, accelerator, batch_size, num_workers) cell,
    GPU-vs-CPU speedup, TF32 on/off delta, epoch-count linearity check.
  * cross-env parity check for the CPU-bound encoding stage.
"""
from __future__ import annotations

import csv
import json
import statistics
from datetime import datetime
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent
RES = BENCH_ROOT / "results"
CSV_PATH = RES / "results.csv"

# Current output language, set per-report in main(). Section builders read this
# global through L() so their signatures stay unchanged.
LANG = "en"

# Phases produced on the AWS EC2 heavy-scale host (Step 1 n=2,000 sweep, Step 2
# n=10,000 CPU-vs-GPU, Step 3 n=2,000 sweep). Everything else is the local
# Windows laptop (parts500 500-part sweeps). The report is split by machine so
# each document only shows one environment's data.
AWS_PHASES = {"HB1", "HB2", "HB3"}


def scope_rows(rows: list[dict], scope: str) -> list[dict]:
    """Filter rows to one machine: 'aws' = AWS_PHASES, 'local' = the rest."""
    if scope == "aws":
        return [r for r in rows if r.get("phase") in AWS_PHASES]
    return [r for r in rows if r.get("phase") not in AWS_PHASES]


def L(en: str, ja: str) -> str:
    """Return the English or Japanese variant of a string based on LANG."""
    return ja if LANG == "ja" else en


def load_rows() -> list[dict]:
    if not CSV_PATH.is_file():
        raise SystemExit(f"no results yet: {CSV_PATH}")
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("wall_s", "throughput", "peak_rss_mb", "peak_gpu_mb"):
            try:
                r[k] = float(r[k]) if r[k] not in ("", None) else None
            except (TypeError, ValueError):
                r[k] = None
        for k in ("n_files", "n_ok", "n_failed"):
            try:
                r[k] = int(float(r[k])) if r[k] not in ("", None) else None
            except (TypeError, ValueError):
                r[k] = None
        for k in ("sub_timings", "extra"):
            try:
                r[k + "_d"] = json.loads(r[k]) if r[k] else {}
            except Exception:
                r[k + "_d"] = {}
    # Exclude non-measurement runs:
    #   phase 0 / smoke   = small smoke test, would show up as a bogus group
    #   warm-up runs      = deliberately absorb cold-cache cost
    #   S1K / keep-best   = CPU dataset-materialisation runs; they duplicate the
    #                       best max_workers point already present in the S1 sweep
    # Accepted phases: legacy numeric "1".."5" and the local-sweep tags S1/S2/S3.
    ok_phase = {"1", "2", "3", "4", "5", "S1", "S2", "S3", "HB1", "HB2", "HB3"}
    drop_note = ("warm-up", "smoke", "calibration", "keep-best")
    out = []
    for r in rows:
        if r.get("status") != "OK":
            continue
        if str(r.get("phase")) not in ok_phase:
            continue
        note = (r.get("note") or "").lower()
        if any(x in note for x in drop_note):
            continue
        out.append(r)
    return out


def encode_seconds(r: dict) -> float | None:
    """The parallel part of step 1 (EncodingTask), falling back to total wall."""
    st = r.get("sub_timings_d") or {}
    for key in ("EncodingTask", "encode_s"):
        if isinstance(st.get(key), (int, float)):
            return float(st[key])
    return r.get("wall_s")


def md_table(headers: list[str], rows: list[list]) -> str:
    def cell(c):
        if c is None:
            return ""
        return str(c).replace("|", "\\|").replace("\n", " ")

    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(cell(c) for c in r) + " |")
    return "\n".join(out)


def html_table(headers: list[str], rows: list[list],
               highlight_first=None) -> str:
    """Raw HTML table (passed through by marked.js) so a single row can be
    tinted. The row whose first cell equals `highlight_first` gets class
    'peak'. Emitted as an HTML block so it survives markdown rendering."""
    import html as _html

    def esc(c):
        return _html.escape("" if c is None else str(c))

    hi = None if highlight_first is None else str(highlight_first)
    parts = ["<table>", "<thead><tr>"
             + "".join(f"<th>{esc(h)}</th>" for h in headers)
             + "</tr></thead>", "<tbody>"]
    for r in rows:
        cls = " class='peak'" if r and hi is not None and str(r[0]) == hi else ""
        parts.append(f"<tr{cls}>"
                     + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


def html_cell_table(headers: list[str], rows: list[list],
                    marks=()) -> str:
    """Raw HTML table where individual cells can be tinted. `marks` is an
    iterable of (row_index, col_index) tuples that get class 'peak'."""
    import html as _html

    def esc(c):
        return _html.escape("" if c is None else str(c))

    marks = set(marks)
    parts = ["<table>", "<thead><tr>"
             + "".join(f"<th>{esc(h)}</th>" for h in headers)
             + "</tr></thead>", "<tbody>"]
    for i, r in enumerate(rows):
        cells = []
        for j, c in enumerate(r):
            cls = " class='peak'" if (i, j) in marks else ""
            cells.append(f"<td{cls}>{esc(c)}</td>")
        parts.append("<tr>" + "".join(cells) + "</tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


def amdahl_serial_fraction(points: list[tuple[int, float]]) -> float | None:
    """Estimate the serial fraction f from T(n) = T1*(f + (1-f)/n).

    Uses the widest worker spread available. Returns None if under-determined.
    """
    pts = sorted(points)
    if len(pts) < 2:
        return None
    (n1, t1), (n2, t2) = pts[0], pts[-1]
    if n1 == n2 or t1 <= 0:
        return None
    # t1/t2 = (f + (1-f)/n1) / (f + (1-f)/n2)
    r = t1 / t2
    denom = (r - 1) * 1.0
    a, b = 1.0 / n1, 1.0 / n2
    #  f + a - a f = r (f + b - r b f)  ->  solve linearly
    #  f(1-a) + a = r [ f(1-b) + b ]
    num = r * b - a
    den = (1 - a) - r * (1 - b)
    if abs(den) < 1e-12:
        return None
    f = num / den
    if not (0.0 <= f <= 1.0):
        return None
    return f


def disp_tag(tag: str) -> str:
    """Display form of an env tag: cpu1.1 -> CPU1.1, gpu1.1 -> GPU1.1."""
    return tag.replace("cpu", "CPU").replace("gpu", "GPU")


def env_cores(env_tag: str) -> tuple[int, int] | None:
    """(physical_cores, logical_cores) from results/env_<tag>.json, or None."""
    p = RES / f"env_{env_tag}.json"
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        phys = int(d.get("physical_cores")) if d.get("physical_cores") else None
        logi = int(d.get("logical_cores")) if d.get("logical_cores") else None
        return (phys, logi) if phys else None
    except Exception:
        return None


def worker_scaling_section(rows: list[dict], step: str, param: str,
                           title: str, scope: str = "") -> list[str]:
    # generate_images=True is a different workload; it must not sit in the
    # scaling table next to the plain runs at the same worker count.
    sel = [r for r in rows
           if r["step"] == step and r["param_name"] == param
           and not (r.get("extra_d") or {}).get("gen_images")]
    if not sel:
        return [f"### {title}\n\n_{L('no data', 'データなし')}_\n"]
    out = [f"### {title}\n"]
    by_env_n: dict[tuple, list[dict]] = {}
    for r in sel:
        by_env_n.setdefault((r["env_tag"], r["n_files"]), []).append(r)

    for (env_tag, n_files), group in sorted(by_env_n.items()):
        # A scaling table needs at least two worker counts; single-point groups
        # (e.g. one-off preflight or full-run points recorded during Step 2 prep,
        # or a lone GPU parity point) are not a sweep and would render as noise.
        if len(group) < 2:
            continue
        group.sort(key=lambda r: int(float(r["param_value"])))
        base = group[0]
        base_t = encode_seconds(base) if step == "dataprep" else base["wall_s"]
        base_n = int(float(base["param_value"]))
        table = []
        pts = []
        for r in group:
            n = int(float(r["param_value"]))
            t = encode_seconds(r) if step == "dataprep" else r["wall_s"]
            if not t:
                continue
            pts.append((n, t))
            speedup = base_t / t if base_t else None
            ideal = n / base_n
            eff = (speedup / ideal * 100) if speedup and ideal else None
            table.append([
                n, f"{t:.1f}",
                f"{(n_files or 0) / t:.2f}" if t else "",
                f"{speedup:.2f}x" if speedup else "",
                f"{eff:.0f}%" if eff else "",
                f"{r['peak_rss_mb']:.0f}" if r.get("peak_rss_mb") else "",
                r["n_failed"] if r["n_failed"] is not None else "",
            ])
        out.append(f"**{disp_tag(env_tag)}, n={n_files} files** "
                   f"({L('speedup relative to', '基準')} {param}={base_n})\n")
        # Peak = the worker count with the shortest time in this group; tint it.
        peak_n = min(pts, key=lambda p: p[1])[0] if pts else None
        out.append(html_table(
            [param,
             L("time (s)", "時間 (s)"),
             L("files/s", "ファイル/s"),
             L("speedup", "速度向上"),
             L("parallel eff.", "並列効率"),
             L("peak RSS (MB)", "最大RSS (MB)"),
             L("failed", "失敗")], table, highlight_first=peak_n))
        # Step 3 embeds per *shape*, and a heavy CAD file can hold many shapes
        # (assemblies), so n_ok (shapes embedded) can far exceed n_files (input
        # CAD files). Surface that so files/s here isn't misread as shapes/s.
        if step == "indexing":
            n_ok_vals = {r.get("n_ok") for r in group if r.get("n_ok")}
            if n_ok_vals and max(n_ok_vals) > (n_files or 0) * 1.5:
                shapes = max(n_ok_vals)
                per = shapes / n_files if n_files else 0
                pk_t = min(pts, key=lambda p: p[1])[1] if pts else None
                sps = shapes / pk_t if pk_t else 0
                out.append(L(
                    f"\n_The **files/s** column counts input CAD files. These "
                    f"{n_files:,} files expand to **{shapes:,} B-rep shapes** "
                    f"(~{per:.1f} shapes/file - the heavy corpus is full of "
                    f"assemblies), and Step 3 embeds each shape: at the peak "
                    f"that is ~{sps:.1f} shapes/s._\n",
                    f"\n_**ファイル/s** 列は入力CADファイル数を数える。この "
                    f"{n_files:,} ファイルは **{shapes:,} 個のB-rep形状** に展開され"
                    f"（約{per:.1f} 形状/ファイル - ヘビーコーパスはアセンブリが多い）、"
                    f"Step 3 は各形状を埋め込む: ピークでは約{sps:.1f} 形状/s。_\n"))
        if pts:
            # Peak = the worker count with the shortest time in this group.
            bn, bt = min(pts, key=lambda p: p[1])
            bfps = (n_files or 0) / bt if bt else 0
            # Smallest worker count within 3% of the peak (noise-tolerant), so we
            # can also recommend the cheapest setting that still hits peak speed.
            pn = next((n for n, t in pts if t <= bt * 1.03), bn)
            msg_en = (f"\n**Fastest: {param} = {bn}** "
                      f"({bt:.1f} s, {bfps:.2f} files/s) - the shortest time in "
                      f"this group.")
            msg_ja = (f"\n**最速: {param} = {bn}**"
                      f"（{bt:.1f} s、{bfps:.2f} ファイル/s）- このグループで最短。")
            if pn != bn:
                msg_en += (f" ({param} = {pn} is within 3% - the cheapest "
                           f"setting at peak throughput.)")
                msg_ja += (f"（{param} = {pn} でその3%以内 - 最小コストで"
                           f"ピーク性能。）")
            # Tie the peak to the machine's core count when they coincide - it
            # explains why the optimum sits where it does and makes the result
            # portable to other machines (scale the worker count with the cores).
            cores = env_cores(env_tag)
            if cores:
                phys, logi = cores
                if bn == phys:
                    msg_en += (f" This equals the machine's **{phys} physical "
                               f"cores**, so the optimum tracks the core count.")
                    msg_ja += (f" これはテストマシンの**物理コア数 {phys}** と等しく、"
                               f"最適ワーカー数はコア数に一致している。")
                elif logi and bn == logi:
                    msg_en += (f" This equals the machine's **{logi} logical "
                               f"processors**.")
                    msg_ja += (f" これはテストマシンの**論理プロセッサ数 {logi}** と等しい。")
            out.append(L(msg_en, msg_ja))
        f = amdahl_serial_fraction(pts)
        if f is not None and len(pts) >= 3:
            max_speedup = 1 / f
            # workers needed to reach 80% of the asymptote
            n80 = (1 - f) / (f * (1 / 0.8 - 1))
            out.append(L(
                f"\nAmdahl fit over {param}={pts[0][0]}..{pts[-1][0]}: "
                f"serial fraction f = **{f * 100:.1f}%**, so the asymptotic "
                f"ceiling is {max_speedup:.1f}x no matter how many workers "
                f"you add. Roughly {param} = {n80:.0f} reaches 80% of that "
                f"ceiling.",
                f"\nアムダール則フィット ({param}={pts[0][0]}..{pts[-1][0]}): "
                f"直列成分 f = **{f * 100:.1f}%**。ワーカーをいくら増やしても "
                f"漸近的な上限は {max_speedup:.1f}倍。おおよそ {param} = {n80:.0f} で "
                f"その上限の80%に到達する。"))
        out.append("")
    if step == "dataprep" and scope != "aws":
        # This "peak is noise, read it as a plateau" caveat comes from the
        # local Windows sweeps, where thermal throttling and other running
        # apps moved the fastest max_workers around (CPU 12-14, GPU 10-12).
        # The dedicated AWS VM runs this task alone, so its Step 1 sweep is a
        # clean monotonic curve peaking at the logical-core count - the caveat
        # does not apply there and would contradict the measured peak.
        out.append(L(
            "> **The exact peak worker count is run-to-run noise, not a stable "
            "optimum.** Across repeated full sweeps the fastest max_workers has "
            "moved between 12 and 14 on CPU and between 10 and 12 on GPU, while "
            "every point from roughly the physical-core count upward stays within "
            "~5% of the best. Read this as a flat plateau near the core count, "
            "not a single best value - do not tune the exact peak.",
            "> **厳密なピークのワーカー数は実行間ノイズであり、安定した最適値ではない。** "
            "フルスイープを繰り返すと、最速の max_workers は CPU で 12〜14、"
            "GPU で 10〜12 の間を移動し、物理コア数付近から上のどの点も最速の"
            "約5%以内に収まる。単一の最良値ではなく、コア数付近の平坦なプラトーとして"
            "読むべきで、厳密なピークをチューニングする意味はない。"))
        out.append("")
    return out


def training_section(rows: list[dict]) -> list[str]:
    """Step 2 - training, reframed on the 10k-file heavy CAD dataset.

    Run on AWS g6.8xlarge (AMD EPYC 16 physical cores + NVIDIA L4).  CPU and GPU
    train the *same* model - identical dataset pointer, batch_size=64, 10 epochs,
    fixed seed, num_workers=0, matmul=high - so wall-clock difference is a pure
    device speed test.  A tiny 30-file / 2-epoch preflight is shown alongside to
    expose how the GPU advantage grows with dataset scale.
    """
    sel = [r for r in rows if r["step"] == "training" and r["phase"] == "HB2"]
    if not sel:
        return [f"### {L('Step 2 - training', 'Step 2 - 学習')}\n\n_{L('no data', 'データなし')}_\n"]

    def hb(accel: str, n: int) -> dict | None:
        for r in sel:
            if r["accelerator"] == accel and r["n_files"] == n:
                return r
        return None

    def spe(r: dict | None) -> float | None:
        if not r:
            return None
        v = (r.get("sub_timings_d") or {}).get("s_per_epoch")
        return float(v) if v else None

    # n_files present in this HB2 batch: the large one is the headline run.
    ns = sorted({r["n_files"] for r in sel if r["n_files"]})
    n_full = ns[-1] if ns else None
    n_pre = ns[0] if len(ns) >= 2 else None

    cpu_full, gpu_full = hb("cpu", n_full), hb("gpu", n_full)
    cs, gs = spe(cpu_full), spe(gpu_full)

    out = [f"### {L('Training speed - CPU vs GPU (10k heavy dataset)', '学習速度 - CPU vs GPU（1万ファイルのヘビーデータ）')}\n"]
    out.append(L(
        f"Ran on **AWS g6.8xlarge** (AMD EPYC, 16 physical cores + **NVIDIA L4** "
        f"GPU) against a **{n_full:,}-file** heavy mechanical-CAD corpus (`mechcad`), "
        f"the same corpus used for the Step 1 / Step 3 worker sweeps above. "
        f"Both devices train from scratch for 10 epochs with the **same dataset "
        f"pointer**, batch_size=64, num_workers=0, matmul=high and a fixed seed, "
        f"so the two runs produce the *same model* and the wall-clock gap is a "
        f"pure CPU-vs-GPU speed test.\n\n"
        f"**Why batch_size is held fixed.** This embedding model is trained with "
        f"*contrastive learning* (SimCLR / NT-Xent): the batch supplies the "
        f"in-batch negatives, so changing batch_size trains a *different* model. "
        f"Holding batch_size=64 (the tutorial default) keeps the model identical "
        f"and isolates device speed. At this scale each epoch is ~204 batches.\n",
        f"**AWS g6.8xlarge**（AMD EPYC・物理16コア + **NVIDIA L4** GPU）で、"
        f"上記 Step 1 / Step 3 のワーカースイープと同じ全コーパス "
        f"**{n_full:,}ファイル**のヘビーな機械CADコーパス（`mechcad`）に対して実行。"
        f"両デバイスとも**同一の"
        f"データセットポインタ**・batch_size=64・num_workers=0・matmul=high・固定seed で"
        f"スクラッチから10 epoch学習するため、両実行は*同一モデル*を生成し、wall時間の"
        f"差は純粋なCPU vs GPUの速度テストになる。\n\n"
        f"**なぜ batch_size を固定するか。** 本埋め込みモデルは*対比学習*"
        f"（SimCLR / NT-Xent）で学習される。バッチがバッチ内の負例を供給するため、"
        f"batch_size を変えると*別のモデル*が学習される。batch_size=64（チュートリアル"
        f"既定）に固定すればモデルは同一に保たれ、device 速度だけを切り出せる。"
        f"この規模では1 epochは約204バッチ。\n"))

    # Submitted vs succeeded: Step 1 tolerates HOOPS' non-deterministic ~2%
    # re-encode failures, so the model trains on the successes only.
    hb1 = None
    for r in load_all_rows():
        if (r.get("phase") == "HB1" and r.get("step") == "dataprep"
                and str(r.get("n_files")) == str(n_full)):
            hb1 = r
            break
    if hb1:
        try:
            n_ok = int(float(hb1.get("n_ok")))
            n_fail = int(float(hb1.get("n_failed")))
        except (TypeError, ValueError):
            n_ok = n_fail = None
        if n_ok is not None and n_fail:
            pct = 100.0 * n_fail / (n_ok + n_fail)
            out.append(L(
                f"**Submitted vs trained.** The {n_full:,} files are the *submitted* "
                f"count. Step 1 encoding tolerates HOOPS AI's known "
                f"**non-deterministic ~{pct:.0f}% re-encode failures** (typically "
                f"`division by zero` or `Data key 'graph' not found`, and a "
                f"*different* set of files each run), so this run dropped "
                f"**{n_fail}** files and the model actually trained on the "
                f"**{n_ok:,} successes**. These are separate from the 24 files "
                f"pre-excluded from the file list - they cannot be filtered ahead "
                f"of time because the failures are not repeatable.\n",
                f"**投入数 vs 学習数。** {n_full:,}ファイルは*投入*数。Step 1 の"
                f"エンコードは、HOOPS AI の既知の**非決定的な約{pct:.0f}%の"
                f"再エンコード失敗**（典型例 `division by zero` や "
                f"`Data key 'graph' not found`。しかも失敗するファイルは実行ごとに"
                f"*異なる*）を許容する。今回は **{n_fail}** ファイルが落ち、モデルは"
                f"実際には **{n_ok:,}件の成功分**で学習されている。これは事前に"
                f"ファイルリストから除外した24件とは別物で、失敗が再現しないため"
                f"事前フィルタで除ききれない。\n"))

    # ---- headline: CPU vs GPU at 10k scale ----
    out.append(f"#### {L('CPU vs GPU at identical settings (batch_size=64)', '同一設定でのCPU vs GPU（batch_size=64）')}\n")
    if cs and gs:
        out.append(L(
            f"**At {n_full:,} files - same model, same 10 epochs, same seed, only "
            f"the device differs - the L4 GPU is {cs / gs:.1f}x faster than the "
            f"16-core CPU:** CPU {cs:.1f} vs GPU {gs:.1f} s/epoch "
            f"({cs / 60:.1f} vs {gs / 60:.1f} min/epoch). Over 10 epochs that is "
            f"CPU {cs * 10 / 60:.0f} min vs GPU {gs * 10 / 60:.0f} min of "
            f"training wall time.\n",
            f"**{n_full:,}ファイルで（同一モデル・同一10 epoch・同一seed、変えるのは "
            f"device のみ）、L4 GPUは16コアCPUの {cs / gs:.1f}倍 高速:** "
            f"CPU {cs:.1f} vs GPU {gs:.1f} s/epoch"
            f"（{cs / 60:.1f} vs {gs / 60:.1f} 分/epoch）。10 epochでは学習wall時間が "
            f"CPU {cs * 10 / 60:.0f}分 vs GPU {gs * 10 / 60:.0f}分。\n"))

        def cell(r: dict | None):
            if not r:
                return ["-", "-", "-", "-", "-"]
            s = spe(r)
            st = r.get("sub_timings_d") or {}
            train_s = st.get("train_s")
            return [
                f"{s:.1f}" if s else "-",
                f"{s / 60:.2f}" if s else "-",
                f"{train_s / 60:.1f}" if train_s else "-",
                f"{r.get('peak_rss_mb'):.0f}" if r.get("peak_rss_mb") else "-",
                f"{r.get('peak_gpu_mb'):.0f}" if r.get("peak_gpu_mb") else "-",
            ]

        table = [
            [L("CPU (EPYC 16-core)", "CPU (EPYC 16コア)")] + cell(cpu_full),
            [L("GPU (NVIDIA L4)", "GPU (NVIDIA L4)")] + cell(gpu_full),
        ]
        out.append(html_table(
            [L("device", "デバイス"), "s/epoch", L("min/epoch", "分/epoch"),
             L("train wall (min)", "学習wall (分)"),
             L("peak RSS (MB)", "最大RSS (MB)"),
             L("peak GPU (MB)", "最大GPU (MB)")],
            table, highlight_first=L("GPU (NVIDIA L4)", "GPU (NVIDIA L4)")))
        out.append(L(
            f"\n_Speedup = CPU s/epoch / GPU s/epoch = "
            f"{cs:.1f} / {gs:.1f} = **{cs / gs:.1f}x**. The GPU also uses far less "
            f"host RAM (peak RSS) because the heavy tensor work lives on the "
            f"device.\n",
            f"\n_速度向上 = CPU s/epoch / GPU s/epoch = "
            f"{cs:.1f} / {gs:.1f} = **{cs / gs:.1f}倍**。重いテンソル演算がデバイス側で"
            f"動くため、GPUはホストRAM（最大RSS）も大幅に少ない。\n"))
        out.append("")

    # ---- scale effect: preflight vs full ----
    if n_pre:
        cpu_pre, gpu_pre = hb("cpu", n_pre), hb("gpu", n_pre)
        cps, gps = spe(cpu_pre), spe(gpu_pre)
        if cps and gps and cs and gs:
            out.append(f"#### {L('The GPU advantage grows with dataset scale', 'GPUの優位はデータ規模とともに拡大')}\n")
            srows = [
                [L(f"{n_pre}-file preflight (2 ep)", f"{n_pre}ファイルのプリフライト (2 ep)"),
                 f"{cps:.2f}", f"{gps:.2f}", f"{cps / gps:.1f}x"],
                [L(f"{n_full:,}-file full run (10 ep)", f"{n_full:,}ファイルの本番 (10 ep)"),
                 f"{cs:.1f}", f"{gs:.1f}", f"{cs / gs:.1f}x"],
            ]
            out.append(html_table(
                [L("dataset", "データセット"),
                 L("CPU s/epoch", "CPU s/epoch"),
                 L("GPU s/epoch", "GPU s/epoch"),
                 L("GPU speedup", "GPU速度向上")], srows, highlight_first=None))
            out.append(L(
                f"\nOn a trivially small dataset the GPU wins by only "
                f"~{cps / gps:.1f}x - launch, host-to-device transfer and Python "
                f"overhead dominate and the L4 is mostly idle. At {n_full:,} files "
                f"(~204 batches/epoch) the GPU is fully fed and the speedup rises "
                f"to **{cs / gs:.1f}x**. This matches the local 500-part finding "
                f"(~1.5x) and the general rule: **the larger the training set, the "
                f"more a GPU pays off for Step 2**.\n",
                f"\n極端に小さいデータセットではGPUの優位は約{cps / gps:.1f}倍にとどまる"
                f"（起動・ホスト→デバイス転送・Pythonオーバーヘッドが支配的で、L4は"
                f"ほぼアイドル）。{n_full:,}ファイル（約204バッチ/epoch）ではGPUが"
                f"十分に稼働し、速度向上は **{cs / gs:.1f}倍** に上がる。これはローカルの"
                f"500パーツの結果（約1.5倍）とも整合し、一般則「**学習データが大きいほど "
                f"Step 2 でGPUが効く**」を裏付ける。\n"))
            out.append("")

    # ---- findings ----
    notes = []
    if cs and gs:
        notes.append(L(
            f"- **Compare on s/epoch, not total wall.** Split/setup is a fixed "
            f"cost; the marginal training cost is CPU {cs:.1f} vs GPU {gs:.1f} "
            f"s/epoch (**{cs / gs:.1f}x**). Project longer runs from s/epoch.",
            f"- **比較は総wallではなく s/epoch で。** split/セットアップは固定コスト。"
            f"限界学習コストは CPU {cs:.1f} vs GPU {gs:.1f} s/epoch"
            f"（**{cs / gs:.1f}倍**）。長時間実行の見積りは s/epoch から行う。"))
    if gpu_full and gpu_full.get("peak_gpu_mb"):
        notes.append(L(
            f"- **L4 memory headroom:** peak GPU {gpu_full['peak_gpu_mb']:.0f} MB "
            f"at batch_size=64 - well under the L4's 23 GB, so larger batches or a "
            f"bigger model would still fit.",
            f"- **L4のメモリ余裕:** batch_size=64で最大GPU "
            f"{gpu_full['peak_gpu_mb']:.0f} MB — L4の23 GBに対して十分小さく、"
            f"より大きなバッチやモデルでも収まる。"))
    if cpu_full and gpu_full and cpu_full.get("peak_rss_mb") and gpu_full.get("peak_rss_mb"):
        notes.append(L(
            f"- **Host RAM:** CPU training peaked at "
            f"{cpu_full['peak_rss_mb'] / 1024:.1f} GB RSS vs the GPU run's "
            f"{gpu_full['peak_rss_mb'] / 1024:.1f} GB - offloading to the device "
            f"cuts host memory pressure too.",
            f"- **ホストRAM:** CPU学習の最大RSSは "
            f"{cpu_full['peak_rss_mb'] / 1024:.1f} GB、GPU実行は "
            f"{gpu_full['peak_rss_mb'] / 1024:.1f} GB — デバイスへのオフロードは"
            f"ホストのメモリ負荷も下げる。"))
    if notes:
        out.append(L("**Findings**\n", "**知見**\n"))
        out.extend(notes)
        out.append("")
    return out


def step3_cpu_gpu_section(rows: list[dict]) -> list[str]:
    sel = [r for r in rows if r["step"] == "indexing"
           and r["param_name"] == "num_workers"
           and not (r.get("extra_d") or {}).get("gen_images")]
    if not sel:
        return []
    by: dict[str, dict[int, float]] = {}
    for r in sel:
        if r["wall_s"]:
            by.setdefault(r["env_tag"], {})[int(float(r["param_value"]))] = r["wall_s"]
    cpu, gpu = by.get("cpu1.1", {}), by.get("gpu1.1", {})
    if not cpu or not gpu:
        return []
    gpu_zero = all((r.get("peak_gpu_mb") in (0.0, 0, None))
                   for r in sel if r["env_tag"] == "gpu1.1")
    cpu_pn = min(cpu, key=lambda n: cpu[n])
    gpu_pn = min(gpu, key=lambda n: gpu[n])
    cpu_pt, gpu_pt = cpu[cpu_pn], gpu[gpu_pn]

    out = [f"### {L('CPU vs GPU', 'CPU vs GPU')}\n"]
    faster = L("faster", "高速") if gpu_pt < cpu_pt else L("slower", "低速")
    out.append(L(
        f"**Step 3 indexing is CPU-bound - the GPU gives no benefit.** At each "
        f"platform's peak, CPU {cpu_pt:.1f} s (num_workers={cpu_pn}) vs GPU "
        f"{gpu_pt:.1f} s (num_workers={gpu_pn}): the GPU install is "
        f"**{cpu_pt / gpu_pt:.2f}x** the CPU throughput (i.e. {faster}). "
        + ("Peak GPU memory was 0 MB in every indexing run - "
           "`embed_shape_batch` runs the B-rep encoding on CPU workers, so a "
           "GPU adds nothing here.\n" if gpu_zero else "\n"),
        f"**Step 3 の索引化もCPUバウンドで、GPUの恩恵はない。** 各プラットフォームの"
        f"ピークで、CPU {cpu_pt:.1f} s (num_workers={cpu_pn}) vs GPU "
        f"{gpu_pt:.1f} s (num_workers={gpu_pn}): GPUインストールはCPUの "
        f"**{cpu_pt / gpu_pt:.2f}倍** のスループット（つまり{faster}）。"
        + ("索引化の全実行で最大GPUメモリは0 MB - `embed_shape_batch` は"
           "B-repエンコードをCPUワーカー上で実行するため、ここではGPUを足しても"
           "効果がない。\n" if gpu_zero else "\n")))

    common = sorted(set(cpu) & set(gpu))
    if common:
        mt = [[n, f"{cpu[n]:.1f}", f"{gpu[n]:.1f}", f"{cpu[n] / gpu[n]:.2f}x"]
              for n in common]
        marks = []
        for i, n in enumerate(common):
            if n == cpu_pn:
                marks.append((i, 1))   # CPU time cell at CPU peak
            if n == gpu_pn:
                marks.append((i, 2))   # GPU time cell at GPU peak
        out.append(L("Matched num_workers, CPU vs GPU:\n",
                     "同一 num_workers でのCPU vs GPU:\n"))
        out.append(html_cell_table(
            ["num_workers", L("CPU time (s)", "CPU 時間 (s)"),
             L("GPU time (s)", "GPU 時間 (s)"),
             L("GPU speedup", "GPU速度向上")], mt, marks))
        out.append(L(
            "_Highlighted: each platform's peak (CPU num_workers=" f"{cpu_pn}, "
            f"GPU num_workers={gpu_pn}). GPU speedup = CPU time / GPU time; "
            "below 1.00x means the GPU install is slower._\n",
            f"_ハイライト: 各プラットフォームのピーク（CPU num_workers={cpu_pn}、"
            f"GPU num_workers={gpu_pn}）。GPU速度向上 = CPU時間 / GPU時間。"
            "1.00x未満はGPUインストールの方が遅いことを意味する。_\n"))
    return out


def load_all_rows() -> list[dict]:
    if not CSV_PATH.is_file():
        return []
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def parity_section(rows: list[dict]) -> list[str]:
    sel = [r for r in rows if r["step"] == "dataprep"]
    # Multiple dataset scales (n=500, n=2000, ...) can coexist. Keying CPU/GPU
    # times by max_workers alone would collide across scales, so first pick the
    # single n_files whose CPU and GPU sweeps overlap on the most worker counts -
    # that gives the richest matched comparison. Any other scale is summarised
    # separately below.
    by_n: dict[int, dict[str, dict[int, float]]] = {}
    for r in sel:
        t = encode_seconds(r)
        if not t:
            continue
        mw = int(float(r["param_value"]))
        env = r["env_tag"]
        d = by_n.setdefault(r["n_files"] or 0, {"cpu1.1": {}, "gpu1.1": {}})
        if env in d:
            d[env][mw] = t
    scored = [(n, len(set(d["cpu1.1"]) & set(d["gpu1.1"])), d)
              for n, d in by_n.items()]
    scored = [s for s in scored if s[1] > 0]
    if not scored:
        return []
    main_n, _, main_d = max(scored, key=lambda s: (s[1], s[0]))
    cpu = main_d["cpu1.1"]
    gpu = main_d["gpu1.1"]
    if not cpu or not gpu:
        return []
    # Comparison point: prefer a worker count both installs actually ran, so the
    # CPU/GPU numbers are like-for-like; otherwise fall back to each install's
    # own peak. (On AWS the GPU install has a single parity point.)
    common = sorted(set(cpu) & set(gpu))
    if common:
        w_cmp = min(common, key=lambda n: cpu[n] + gpu[n])
        cpu_pn = gpu_pn = w_cmp
        cpu_pt, gpu_pt = cpu[w_cmp], gpu[w_cmp]
    else:
        cpu_pn = min(cpu, key=lambda n: cpu[n])
        gpu_pn = min(gpu, key=lambda n: gpu[n])
        cpu_pt, gpu_pt = cpu[cpu_pn], gpu[gpu_pn]

    out = [f"### {L('CPU vs GPU (does the venv matter?)', 'CPU vs GPU（venvは影響するか？）')}\n"]

    close = abs(cpu_pt - gpu_pt) / max(cpu_pt, gpu_pt) < 0.05
    at_en = (f"At max_workers={cpu_pn}" if common
             else f"At each install's peak (CPU max_workers={cpu_pn}, "
                  f"GPU max_workers={gpu_pn})")
    at_ja = (f"max_workers={cpu_pn} で" if common
             else f"各インストールのピーク（CPU max_workers={cpu_pn}、"
                  f"GPU max_workers={gpu_pn}）で")
    out.append(L(
        f"**Step 1 is CPU-bound - the GPU install buys nothing.** "
        f"{at_en}, CPU {cpu_pt:.1f} s vs GPU {gpu_pt:.1f} s: the GPU install is "
        f"**{cpu_pt / gpu_pt:.2f}x** the CPU throughput"
        + (" - within run-to-run noise, i.e. parity. " if close else ". ")
        + "Encoding is HOOPS Exchange + numpy on the CPU workers and never runs a "
        "model forward pass, so the torch wheel is irrelevant. That parity is "
        "exactly what licenses the Step 2/3 CPU-vs-GPU comparisons: the two "
        "installs are identical everywhere except the device-bound stages.\n",
        f"**Step 1 はCPUバウンドで、GPUインストールの恩恵はない。** "
        f"{at_ja}、CPU {cpu_pt:.1f} s vs GPU {gpu_pt:.1f} s: GPUインストールはCPUの "
        f"**{cpu_pt / gpu_pt:.2f}倍** のスループット"
        + ("（実行ごとのノイズの範囲内＝ほぼ同等）。" if close else "。")
        + "エンコードはCPUワーカー上のHOOPS Exchange + numpyで、モデルの順伝播を"
        "一切行わないためtorch wheelは無関係。このパリティこそが Step 2/3 の "
        "CPU vs GPU 比較の妥当性を担保する — 両インストールはデバイス依存の"
        "段階以外はすべて同一である。\n"))

    if len(common) >= 3:
        mt = [[n, f"{cpu[n]:.1f}", f"{gpu[n]:.1f}", f"{cpu[n] / gpu[n]:.2f}x"]
              for n in common]
        marks = [(i, 1) for i, n in enumerate(common) if n == cpu_pn]
        marks += [(i, 2) for i, n in enumerate(common) if n == gpu_pn]
        out.append(L("Matched max_workers, CPU vs GPU:\n",
                     "同一 max_workers でのCPU vs GPU:\n"))
        out.append(html_cell_table(
            ["max_workers", L("CPU time (s)", "CPU 時間 (s)"),
             L("GPU time (s)", "GPU 時間 (s)"),
             L("GPU speedup", "GPU速度向上")], mt, marks))
        out.append(L(
            f"_Highlighted: the compared point (max_workers={cpu_pn}). "
            "GPU speedup = CPU time / GPU time; near 1.00x means the two "
            "installs are equivalent on this CPU-bound stage._\n",
            f"_ハイライト: 比較点（max_workers={cpu_pn}）。"
            "GPU速度向上 = CPU時間 / GPU時間。1.00x付近はこのCPUバウンド段階で"
            "両インストールが同等であることを意味する。_\n"))
    elif common:
        # Single shared worker count (the AWS parity point). Render it in the
        # same side-by-side matched layout as the Step 3 CPU-vs-GPU table, and
        # since the two installs are within run-to-run noise here, highlight
        # BOTH the CPU and GPU time cells (parity, not a single winner).
        w = w_cmp
        ct, gt = cpu[w], gpu[w]
        mt = [[w, f"{ct:.1f}", f"{gt:.1f}", f"{ct / gt:.2f}x"]]
        marks = [(0, 1), (0, 2)]
        out.append(L(
            f"Matched max_workers, CPU vs GPU (both installs run at "
            f"max_workers={w} on the {main_n:,}-file corpus):\n",
            f"同一 max_workers でのCPU vs GPU（{main_n:,}ファイルの全コーパスで"
            f"両インストールを max_workers={w} で実行）:\n"))
        out.append(html_cell_table(
            ["max_workers", L("CPU time (s)", "CPU 時間 (s)"),
             L("GPU time (s)", "GPU 時間 (s)"),
             L("GPU speedup", "GPU速度向上")], mt, marks))
        out.append(L(
            f"_Highlighted: both installs at max_workers={w} - the encoding "
            f"never touches the GPU, so the two are within run-to-run noise "
            f"(parity). GPU speedup = CPU time / GPU time._\n",
            f"_ハイライト: max_workers={w} の両インストール — エンコードはGPUを"
            f"一切使わないため、両者は実行ごとのノイズの範囲内（ほぼ同等）。"
            f"GPU速度向上 = CPU時間 / GPU時間。_\n"))
    # GPU-idle evidence: sampled during the local 500-part GPU sweep, so only
    # attach it to that scale's report (not the AWS heavy run).
    gpu_util = RES / "step1_gpu_util.json"
    if gpu_util.is_file() and main_n and main_n <= 1000:
        try:
            u = json.loads(gpu_util.read_text(encoding="utf-8"))
            out.append(L(
                f"Measured during the GPU-venv run: GPU utilisation averaged "
                f"**{u['gpu_util_avg']}%** (max {u['gpu_util_max']}%, "
                f"{u['samples']} samples, {u['gpu_mem_used_max_mib']} MiB peak). "
                f"The GPU is effectively idle - Step 1 does no model forward pass, "
                f"so it runs on the CPU workers regardless of the installed torch "
                f"wheel, and a GPU buys nothing for this stage.\n",
                f"GPU用venv実行中の実測: GPU使用率は平均 **{u['gpu_util_avg']}%**"
                f"（最大 {u['gpu_util_max']}%、{u['samples']}サンプル、"
                f"ピークメモリ {u['gpu_mem_used_max_mib']} MiB）。"
                f"GPUは実質的にアイドル。Step 1 はモデルの順伝播を行わないため、"
                f"インストールされたtorch wheelに関わらずCPUワーカー上で実行され、"
                f"この段階でGPUを用いても効果はない。\n"))
        except Exception:
            pass
    return out


def _env_fields(d: dict) -> list[tuple[str, str]]:
    return [
        ("CPU", str(d.get("cpu_brand"))),
        (L("Cores", "コア"),
         f"{d.get('physical_cores')} {L('physical', '物理')} / "
         f"{d.get('logical_cores')} {L('logical', '論理')}"),
        ("RAM", f"{d.get('ram_gb')} GB"),
        ("GPU (nvidia-smi)", "; ".join(d.get("gpus") or []) or "-"),
        ("torch", f"{d.get('torch_version')} (cuda {d.get('torch_cuda')})"),
        ("cuda_available", str(d.get("cuda_available"))),
        ("hoops_ai", str(d.get("hoops_ai_version"))),
        ("OS / Python", f"{d.get('os')} / {d.get('python')}"),
    ]


def _machine_block(prefix: str, title_en: str, title_ja: str,
                   note_en: str, note_ja: str) -> list[str]:
    """Render a merged CPU/GPU env table for one machine.

    `prefix` selects the source files results/{prefix}{tag}.json (prefix
    "env_" = local dev machine, "env_aws_" = the AWS heavy-scale host). Returns
    an empty list when that machine has no env files, so the block is optional.
    """
    tags = [t for t in ("cpu1.1", "gpu1.1")
            if (RES / f"{prefix}{t}.json").is_file()]
    if not tags:
        return []
    envs = {t: json.loads((RES / f"{prefix}{t}.json").read_text(encoding="utf-8"))
            for t in tags}
    per = {t: _env_fields(envs[t]) for t in tags}
    labels = [lbl for lbl, _ in per[tags[0]]]

    import html as _html

    def esc(s):
        return _html.escape("" if s is None else str(s))

    out = [f"#### {L(title_en, title_ja)}\n", L(note_en, note_ja)]
    header = f"<th>{esc(L('item', '項目'))}</th>" + \
        "".join(f"<th>{esc(disp_tag(t))}</th>" for t in tags)
    body = []
    for i, lbl in enumerate(labels):
        vals = [per[t][i][1] for t in tags]
        if len(set(vals)) == 1:
            cell = f"<td colspan='{len(tags)}'>{esc(vals[0])}</td>"
        else:
            cell = "".join(f"<td>{esc(v)}</td>" for v in vals)
        body.append(f"<tr><td>{esc(lbl)}</td>{cell}</tr>")
    out.append("<table>\n<thead><tr>" + header + "</tr></thead>\n<tbody>\n"
               + "\n".join(body) + "\n</tbody></table>")
    out.append("")
    return out


def env_section(scope: str) -> list[str]:
    out = [f"### {L('Test environment', 'テスト環境')}\n"]
    if scope == "aws":
        block = _machine_block(
            "env_aws_",
            "AWS g6.8xlarge (EC2)",
            "AWS g6.8xlarge (EC2)",
            "Heavy-scale host: the ~10k-file Step 1 worker sweep and the "
            "Step 2 CPU-vs-GPU training run were executed here. Both installs "
            "share the same instance; only the torch wheel differs (shared rows "
            "are merged).",
            "ヘビースケール実行環境: 約1万ファイルの Step 1 ワーカースイープと "
            "Step 2 CPU vs GPU 学習をこの環境で実行。両インストールは"
            "同一インスタンス上で、異なるのは torch wheel のみ（共通行はセル結合）。")
    else:
        block = _machine_block(
            "env_",
            "Local development laptop (Windows)",
            "ローカル開発ノートPC (Windows)",
            "The 500-file Step 1 / Step 3 sweeps and preflight checks were run "
            "here. Both installs share the same machine; only the torch wheel "
            "differs (shared rows are merged).",
            "500ファイルの Step 1 / Step 3 スイープとプリフライトはこの環境で実行。"
            "両インストールは同一マシン上で、異なるのは torch wheel のみ"
            "（共通行はセル結合）。")
    if not block:
        return out
    out += block
    return out


def build_doc(rows: list[dict], all_rows: list[dict], scope: str) -> list[str]:
    rows = scope_rows(rows, scope)
    all_rows = [r for r in all_rows
                if (r.get("phase") in AWS_PHASES) == (scope == "aws")]
    ok, failed = len(rows), len([r for r in all_rows if r.get("status") != "OK"])
    is_aws = scope == "aws"
    has_training = any(r["step"] == "training" and r["phase"] == "HB2"
                       for r in rows)
    # Data-driven headline figures for the intro/summary (Step 2 CPU vs GPU),
    # so the prose never contradicts the tables below when the run changes.
    _hb2 = [r for r in rows if r["step"] == "training" and r["phase"] == "HB2"]
    _ns2 = sorted({r["n_files"] for r in _hb2 if r["n_files"]})
    n_full = _ns2[-1] if _ns2 else None

    def _spe_at(accel: str) -> float | None:
        for r in _hb2:
            if r["accelerator"] == accel and r["n_files"] == n_full:
                v = (r.get("sub_timings_d") or {}).get("s_per_epoch")
                try:
                    return float(v) if v else None
                except (TypeError, ValueError):
                    return None
        return None

    cpu_spe = _spe_at("cpu")
    gpu_spe = _spe_at("gpu")
    gpu_x = (cpu_spe / gpu_spe) if (cpu_spe and gpu_spe) else None
    n_disp = f"{n_full:,}" if n_full else "~10k"
    _gx = f"{gpu_x:.1f}" if gpu_x else "?"
    _cs = f"{cpu_spe:.0f}" if cpu_spe else "?"
    _gs = f"{gpu_spe:.0f}" if gpu_spe else "?"
    if is_aws:
        title = L("# HOOPS AI 1.1 benchmark (AWS EC2) - CPU vs GPU",
                  "# HOOPS AI 1.1 ベンチマーク (AWS EC2) - CPU vs GPU")
        dataset_line = L(
            "Machine: **AWS g6.8xlarge** (AMD EPYC 16-core + NVIDIA L4). "
            f"Dataset: `mechcad` heavy mechanical-CAD corpus - the same full "
            f"{n_disp}-file corpus is used for the Step 1 worker sweep, the "
            f"Step 2 CPU-vs-GPU training run, and the Step 3 embedding + FAISS "
            f"indexing (using the tutorial's pretrained model "
            f"`ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt`).",
            "マシン: **AWS g6.8xlarge**（AMD EPYC 16コア + NVIDIA L4）。"
            f"データセット: `mechcad` ヘビー機械CADコーパス - Step 1 の"
            f"ワーカースイープ、Step 2 の CPU vs GPU 学習、そして Step 3 の"
            f"埋め込み + FAISS索引化（チュートリアルの事前学習モデル "
            f"`ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt` を使用）のすべてで、"
            f"同じ{n_disp}ファイルの全コーパスを使用。")
    else:
        title = L("# HOOPS AI 1.1 benchmark (local Windows) - CPU vs GPU",
                  "# HOOPS AI 1.1 ベンチマーク (ローカル Windows) - CPU vs GPU")
        dataset_line = L(
            "Machine: **local Windows laptop** (Intel 14-core + RTX 2000 Ada). "
            "Dataset: `parts500` failure-free 500-part sample.",
            "マシン: **ローカル Windows ノートPC**（Intel 14コア + RTX 2000 Ada）。"
            "データセット: `parts500`（失敗のない500パーツのサンプル）。")
    doc = [
        title,
        "",
        dataset_line,
        L(f"Benchmark runs: {ok} succeeded, {failed} failed or skipped. "
          f"Raw data: `results/results.csv`, per-run logs in `logs/`.",
          f"ベンチマーク実行: 成功 {ok} 件、失敗/スキップ {failed} 件。"
          f"生データ: `results/results.csv`、各実行ログは `logs/`。"),
        "",
        L("See also the companion report for the other machine "
          "(`REPORT_local.*` / `REPORT_aws.*`).",
          "もう一方のマシンの対になるレポートも参照"
          "（`REPORT_local.*` / `REPORT_aws.*`）。"),
        "",
        L("## Summary", "## 結論（要約）"),
        "",
        L("Conclusions first; the supporting numbers and per-setting sweeps "
          "follow in each step's section.",
          "結論を先に述べる。裏付けとなる数値と設定ごとのスイープは各Stepの"
          "セクションを参照。"),
        "",
        L("- **Step 1 (encoding) and Step 3 (embedding + indexing) are "
          "CPU-bound** - the GPU sits idle, so the GPU install buys nothing. "
          "Step 1 throughput scales with worker count up to roughly the "
          "machine's physical core count, then flattens; Step 3 peaks at a "
          "lower worker count and then *declines* as more workers are added, "
          "because each worker process spawns its own pool of CPU compute "
          "(intra-op) threads: past a low worker count the total thread count far "
          "exceeds the machine's cores, and that CPU-thread oversubscription - not "
          "RAM or file-descriptor pressure - is what slows every worker down.",
          "- **Step 1（エンコード）と Step 3（埋め込み + 索引化）はCPUバウンド** "
          "— GPUはアイドルで、GPUインストールの恩恵はない。Step 1 のスループットは"
          "ワーカー数に応じて物理コア数付近まで向上し、その後は頭打ち。Step 3 はより"
          "少ないワーカー数でピークに達し、それ以上増やすと*低下*する — 各ワーカー"
          "プロセスが独自のCPU計算（intra-op）スレッド群を起動するため、少数のワーカーを"
          "超えると総スレッド数がコア数を大きく上回り、そのCPUスレッドの過剰割り当て"
          "（RAM やファイルディスクリプタの負荷ではなく）が全ワーカーを遅くするため。"),
    ]
    if has_training:
        doc.append(L(
          "- **Step 2 (training) is the only GPU-bound stage** and the only "
          "place a GPU pays off. Because the model is trained with contrastive "
          "learning, batch_size changes the trained model, so the fair speed "
          "comparison fixes batch_size: at the tutorial default batch_size=64 "
          "(same model, same 10 epochs), on a "
          f"**{n_disp}-file heavy dataset** "
          "(AWS g6.8xlarge, NVIDIA L4) the GPU is about "
          f"**{_gx}x faster** than the "
          f"16-core CPU ({_cs} vs {_gs} s/epoch). Compare on `s/epoch`, not total "
          "wall (which includes a fixed start-up cost).",
          "- **Step 2（学習）が唯一のGPUバウンドな段階**であり、GPUが効果を発揮する"
          "唯一の場所。本モデルは対比学習のため batch_size を変えると学習されるモデル"
          "自体が変わる。したがって公平な速度比較は batch_size を固定して行う："
          "チュートリアル既定の batch_size=64（同一モデル・同一10 epoch）で、"
          f"**{n_disp}ファイルのヘビーデータ**（AWS g6.8xlarge・NVIDIA L4）では "
          f"GPUは16コアCPUの約**{_gx}倍**高速（{_cs} vs {_gs} s/epoch）。"
          "比較は総wall（固定の起動コストを含む）ではなく `s/epoch` で行う。"))
    else:
        doc.append(L(
          "- **Step 2 (training) is not covered on this machine** - it was "
          "benchmarked separately on the AWS EC2 host (see `REPORT_aws.*`), "
          "where the L4 GPU is ~4.9x faster than the CPU at 10k scale.",
          "- **Step 2（学習）はこのマシンでは対象外** - AWS EC2 環境で別途計測した"
          "（`REPORT_aws.*` を参照）。10kスケールで L4 GPU は CPU の約4.9倍高速。"))
    doc += [
        L("- _How the tables read:_ within each sweep, speedup is quoted "
          "against the smallest setting measured in that group (lowest worker "
          "count, or smallest batch size for step 2); parallel efficiency is "
          "speedup divided by the ideal linear speedup.",
          "- _表の読み方:_ 各スイープ内の速度向上は、そのグループで実測した最小設定を"
          "基準に示す（最小ワーカー数、Step 2 では最小バッチサイズ）。"
          "並列効率は速度向上を理想的な線形速度向上で割った値。"),
        "",
    ]
    doc += env_section(scope)
    doc += [L("## Step 1 - CAD encoding (DataPrep)", "## Step 1 - CADエンコード (DataPrep)"), ""]
    doc += worker_scaling_section(rows, "dataprep", "max_workers",
                                  L("max_workers scaling", "max_workers スケーリング"),
                                  scope)
    doc += parity_section(rows)
    if has_training:
        doc += [L("## Step 2 - training", "## Step 2 - 学習"), ""]
        doc += training_section(rows)
    doc += [L("## Step 3 - embedding + FAISS indexing", "## Step 3 - 埋め込み + FAISS索引化"), ""]
    doc += worker_scaling_section(rows, "indexing", "num_workers",
                                  L("num_workers scaling", "num_workers スケーリング"),
                                  scope)
    doc += step3_cpu_gpu_section(rows)

    # image-generation overhead
    idx = [r for r in rows if r["step"] == "indexing"]
    with_img = [r for r in idx if r["extra_d"].get("gen_images")]
    if with_img:
        lines = []
        for w in with_img:
            peer = [r for r in idx
                    if not r["extra_d"].get("gen_images")
                    and r["env_tag"] == w["env_tag"]
                    and r["n_files"] == w["n_files"]
                    and r["param_value"] == w["param_value"]]
            if peer:
                a = peer[0]["sub_timings_d"].get("embed_s")
                b = w["sub_timings_d"].get("embed_s")
                if a and b:
                    lines.append([disp_tag(w["env_tag"]), w["n_files"], w["param_value"],
                                  f"{a:.1f}", f"{b:.1f}",
                                  f"{(b - a) / a * 100:+.0f}%",
                                  f"{(b - a) / w['n_files']:.2f}"])
        if lines:
            doc += [f"### {L('Cost of', 'コスト:')} `generate_images=True`\n",
                    md_table(["env", L("n files", "ファイル数"), "num_workers",
                              L("embed w/o images (s)", "埋め込み 画像なし (s)"),
                              L("embed w/ images (s)", "埋め込み 画像あり (s)"),
                              L("overhead", "オーバーヘッド"),
                              L("s per part", "パーツあたり秒")], lines),
                    "",
                    L("Turn this off for throughput benchmarking and for production "
                      "index builds where thumbnails are not needed.",
                      "スループット計測や、サムネイル不要の本番索引構築では無効化する。"), ""]

    # failures
    bad = [r for r in all_rows if r.get("status") != "OK"]
    if bad:
        doc += [L("## Failed / skipped runs", "## 失敗 / スキップした実行"), "",
                md_table(["step", "env", "accel", "param", "value", "status",
                          L("note", "備考")],
                         [[r["step"], disp_tag(r["env_tag"]), r["accelerator"], r["param_name"],
                           r["param_value"], r["status"], (r["note"] or "")[:160]]
                          for r in bad]), ""]

    skipped = RES / "skipped.txt"
    if skipped.is_file():
        # utf-8-sig: PowerShell may have written a BOM
        doc += [L("### Configurations dropped for time budget",
                  "### 時間予算のため除外した構成"), "",
                "```", skipped.read_text(encoding="utf-8-sig").strip(), "```", ""]

    doc += [L("## Caveats", "## 注意事項"), "",
            L("- Single machine, single run per configuration: differences under "
              "roughly 5% are inside run-to-run noise. Re-run a cell if a "
              "conclusion depends on a small gap.",
              "- 単一マシン・各構成1回のみの実行。約5%未満の差は実行ごとのノイズの範囲内。"
              "小さな差に結論が依存する場合は再実行すること。")]
    if not is_aws:
        doc.append(L(
            "- The `parts500` sample was pre-filtered to be failure-free at the "
            "chosen worker timeout, so these numbers reflect the clean-data "
            "throughput, not robustness against problematic files.",
            "- `parts500` サンプルは選択したワーカータイムアウトで失敗しないよう "
            "事前にフィルタ済み。これらの数値はクリーンデータでのスループットを表し、"
            "問題のあるファイルに対する堅牢性を示すものではない。"))
    else:
        doc.append(L(
            "- The heavy `mechcad` files trigger HOOPS AI's known "
            "non-deterministic ~2% re-encode failures, so a few dozen files drop "
            "each run; the throughput figures are over the successfully processed "
            "files.",
            "- ヘビーな `mechcad` ファイルは HOOPS AI の既知の非決定的な約2%の"
            "再エンコード失敗を誘発し、実行ごとに数十ファイルが脱落する。"
            "スループットは正常処理できたファイルに対する値。"))
    doc += [L("- Background load on the machine (including antivirus scanning "
              "the STEP files) affects the CPU-bound stages more than the "
              "GPU stage.",
              "- マシンのバックグラウンド負荷（STEPファイルをスキャンする "
              "アンチウイルス等）は、GPU段階よりCPUバウンド段階に強く影響する。"),
            ""]
    return doc


def write_report(rows: list[dict], all_rows: list[dict], lang: str,
                 scope: str) -> None:
    global LANG
    LANG = lang
    doc = build_doc(rows, all_rows, scope)
    md = "\n".join(doc)
    lang_sfx = "_ja" if lang == "ja" else ""
    suffix = f"_{scope}{lang_sfx}"
    base = "HOOPS AI 1.1 ベンチマーク" if lang == "ja" else "HOOPS AI 1.1 benchmark"
    tag = "AWS EC2" if scope == "aws" else ("ローカル Windows" if lang == "ja"
                                            else "local Windows")
    title = f"{base} ({tag})"
    md_path = RES / f"REPORT{suffix}.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"[report] -> {md_path}")

    html = ("<!doctype html><meta charset='utf-8'>"
            f"<html lang='{lang}'><title>{title}</title>"
            "<style>body{font-family:-apple-system,Segoe UI,Meiryo,sans-serif;max-width:1100px;"
            "margin:2rem auto;padding:0 1rem;line-height:1.6;color:#1a1a1a}"
            "table{border-collapse:collapse;margin:1rem 0;font-size:14px}"
            "th,td{border:1px solid #ddd;padding:6px 10px;text-align:right}"
            "th{background:#f5f5f5;text-align:left}td:first-child,th:first-child{text-align:left}"
            "code{background:#f5f5f5;padding:1px 4px;border-radius:3px}"
            "h1,h2,h3{line-height:1.25}h2{border-bottom:2px solid #eee;padding-bottom:.3rem}"
            "tr.peak td{background:#fff3cd;font-weight:600}"
            "td.peak{background:#fff3cd;font-weight:600}"
            "</style>\n<div id='c'></div>\n"
            "<script src='https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js'></script>\n"
            "<script>document.getElementById('c').innerHTML=marked.parse("
            + json.dumps(md) + ");</script>")
    html_path = RES / f"REPORT{suffix}.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"[report] -> {html_path}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Render results.csv into REPORT files.")
    ap.add_argument("--lang", choices=["en", "ja", "both"], default="both",
                    help="Report language(s) to emit (default: both).")
    ap.add_argument("--scope", choices=["local", "aws", "both"], default="both",
                    help="Machine(s) to emit a report for (default: both). "
                         "'local' = Windows laptop, 'aws' = AWS EC2 heavy host.")
    args = ap.parse_args()

    rows = load_rows()
    all_rows = load_all_rows()
    langs = ["en", "ja"] if args.lang == "both" else [args.lang]
    scopes = ["local", "aws"] if args.scope == "both" else [args.scope]
    for scope in scopes:
        if not scope_rows(rows, scope):
            continue
        for lang in langs:
            write_report(rows, all_rows, lang, scope)


if __name__ == "__main__":
    main()
