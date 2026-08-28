"""Build a benchmark file list from an "Add Folder" index report, excluding the
files that failed to convert.

The report groups every input file under a bracketed section header, e.g.

    [LIGHT - added in pass 1] (10673)
    C:/SDK/HOOPS_AI/mechcad/bearing/0.stp  [33.1 KB]
    ...
    [HEAVY - recovered in pass 2] (200)
    ...
    [FAILED - not indexed after retry] (24)
    C:/SDK/HOOPS_AI/mechcad/bearing/453.stp  [542.8 KB]  <= Failed to compute ...
    [HEAVY-FLAGGED (RAM fallback, 1 worker)] (1)
    ...

We keep every file that WAS indexed (all sections whose header does not match
--exclude, i.e. everything except FAILED) and drop the ones that failed. The
result is a clean ~10k list that a fresh benchmark run can process end-to-end
without hitting the known-bad files again.

Portability (Windows PC -> AWS Ubuntu)
--------------------------------------
The report stores absolute Windows-style paths under a common corpus root
(default C:/SDK/HOOPS_AI/mechcad). To make the same list usable on a Linux box
where the corpus lives somewhere else, the list is written with paths RELATIVE
to that root (e.g. "bearing/0.stp"). At run time, point BENCH_FILE_ROOT at the
corpus location and bench_tasks.gather_cad_files prepends it. Use --absolute
(optionally with --target-base) if you would rather bake full paths in.

Examples
--------
    # relative list (recommended) - portable across machines
    python make_filelist_from_log.py \
        "C:/SDK/HOOPS_AI/indexes/add_folder_report_20260809_225535.txt"

    # absolute list rebased for the Ubuntu box
    python make_filelist_from_log.py <report> --absolute \
        --target-base /home/ubuntu/mechcad
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path, PurePosixPath

from bench_common import BENCH_ROOT

SECTION_RE = re.compile(r"^\[(?P<name>.+?)\]\s*\((?P<count>\d+)\)\s*$")


def parse_sections(text: str) -> "dict[str, list[str]]":
    """Return {section_name: [raw_path, ...]} preserving file order."""
    sections: dict[str, list[str]] = {}
    current: "list[str] | None" = None
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = SECTION_RE.match(line.strip())
        if m:
            current = []
            sections[m.group("name")] = current
            continue
        if current is None:
            continue
        # entry line: "<path>  [size]  <= reason". The path is everything before
        # the first run of 2+ spaces (paths in this corpus have no spaces).
        path = re.split(r"\s{2,}", line.strip(), maxsplit=1)[0].strip()
        if path:
            current.append(path)
    return sections


def normalize(p: str) -> str:
    return p.replace("\\", "/").strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="path to the Add Folder report .txt")
    ap.add_argument("--exclude", default="FAILED",
                    help="drop sections whose header matches this (regex, "
                         "case-insensitive). Default: FAILED")
    ap.add_argument("--source-base", default="C:/SDK/HOOPS_AI/mechcad",
                    help="corpus root as it appears in the report; stripped to "
                         "make relative paths (default C:/SDK/HOOPS_AI/mechcad)")
    ap.add_argument("--absolute", action="store_true",
                    help="write absolute paths instead of paths relative to "
                         "--source-base")
    ap.add_argument("--target-base", default=None,
                    help="with --absolute, rebase onto this root (e.g. "
                         "/home/ubuntu/mechcad). Default: keep --source-base")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the list to the first N files (after ordering)")
    ap.add_argument("--out", default=None,
                    help="output path (default filelists/mech_heavy.txt)")
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.is_file():
        raise SystemExit(f"report not found: {log_path}")

    sections = parse_sections(log_path.read_text(encoding="utf-8-sig"))
    if not sections:
        raise SystemExit("no bracketed sections found - is this an Add Folder report?")

    excl = re.compile(args.exclude, re.IGNORECASE)
    src_base = normalize(args.source_base).rstrip("/")

    kept: list[str] = []
    dropped: list[str] = []
    kept_sections: list[str] = []
    seen: set[str] = set()
    for name, paths in sections.items():
        target = dropped if excl.search(name) else kept
        if target is kept:
            kept_sections.append(f"{name} ({len(paths)})")
        for raw in paths:
            np = normalize(raw)
            if target is kept and np in seen:
                continue  # de-dup across sections (e.g. HEAVY-FLAGGED overlap)
            if target is kept:
                seen.add(np)
            target.append(np)

    # Build output paths.
    out_paths: list[str] = []
    skipped_base = 0
    for np in kept:
        rel = np
        low = np.lower()
        if low.startswith(src_base.lower() + "/"):
            rel = np[len(src_base) + 1:]
        elif np.lower() == src_base.lower():
            rel = ""
        if args.absolute:
            base = normalize(args.target_base) if args.target_base else src_base
            out_paths.append(str(PurePosixPath(base) / rel) if rel else base)
        else:
            if rel == np:  # never matched the source base
                skipped_base += 1
            out_paths.append(rel)

    if args.limit is not None:
        out_paths = out_paths[:args.limit]

    out = Path(args.out) if args.out else BENCH_ROOT / "filelists" / "mech_heavy.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(out_paths) + "\n", encoding="utf-8")

    manifest = {
        "source_report": str(log_path),
        "kept_sections": kept_sections,
        "excluded_pattern": args.exclude,
        "n_kept": len(out_paths),
        "n_dropped": len(dropped),
        "dropped_files": dropped,
        "mode": "absolute" if args.absolute else "relative",
        "source_base": src_base,
        "target_base": (normalize(args.target_base) if args.target_base else src_base)
                       if args.absolute else None,
        "run_root_env": None if args.absolute else "BENCH_FILE_ROOT",
        "out": str(out),
    }
    man = out.with_suffix(".manifest.json")
    man.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[filelist] kept {len(out_paths)} files, dropped {len(dropped)} failed")
    print(f"[filelist] kept sections: {', '.join(kept_sections)}")
    if skipped_base:
        print(f"[filelist] WARNING {skipped_base} paths did not start with "
              f"--source-base {src_base!r}; they were written unchanged")
    print(f"[filelist] -> {out}")
    print(f"[filelist] -> {man}")
    if not args.absolute:
        print("[filelist] paths are RELATIVE; set BENCH_FILE_ROOT to the corpus "
              "root at run time (e.g. /home/ubuntu/mechcad).")


if __name__ == "__main__":
    main()
