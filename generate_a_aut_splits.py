#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import numpy as np

# =========================
# CONFIG
# =========================
ROOT = Path("data/apigraph")
SUFFIX = "_selected.npz"

# Define AUT Train windows (inclusive, YYYY-MM)
TRAIN_WINDOWS = [
    #("2012-01", "2012-12"),  # [1-AUT] Train
    ("2013-07", "2014-06"),  # [2-AUT] Train
    ("2015-01", "2015-12"),  # [3-AUT] Train
    ("2016-07", "2017-06"),  # [4-AUT] Train
]
# =========================


def _concat_value(a, b):
    if a is None:
        return b
    if b is None:
        return a

    a_arr = np.asarray(a)
    b_arr = np.asarray(b)

    if a_arr.ndim == 0 and b_arr.ndim == 0:
        return np.array([a_arr.item(), b_arr.item()])
    if a_arr.ndim == 0:
        a_arr = a_arr.reshape(1)
    if b_arr.ndim == 0:
        b_arr = b_arr.reshape(1)

    return np.concatenate([a_arr, b_arr], axis=0)


def yyyymm_to_int(s: str) -> int:
    # "2013-07" -> 201307
    y, m = s.split("-")
    return int(y) * 100 + int(m)


def int_to_yyyymm(v: int) -> str:
    y = v // 100
    m = v % 100
    return f"{y:04d}-{m:02d}"


def iter_months(start_yyyymm: str, end_yyyymm: str) -> list[str]:
    start = yyyymm_to_int(start_yyyymm)
    end = yyyymm_to_int(end_yyyymm)
    out = []
    cur_y = start // 100
    cur_m = start % 100
    while True:
        cur = cur_y * 100 + cur_m
        if cur > end:
            break
        out.append(f"{cur_y:04d}-{cur_m:02d}")
        cur_m += 1
        if cur_m == 13:
            cur_m = 1
            cur_y += 1
    return out


def load_month_npz(path: Path):
    return np.load(path, allow_pickle=True)


def merge_npz_files(tag: str, files: list[Path]) -> Path:
    loaded = [load_month_npz(p) for p in files]
    keys = list(loaded[0].files)

    # Validate same keys across all files
    for i, z in enumerate(loaded[1:], start=1):
        if set(z.files) != set(keys):
            raise ValueError(
                f"[{tag}] Key mismatch between {files[0].name} and {files[i].name}\n"
                f"  first keys: {sorted(keys)}\n"
                f"  other keys: {sorted(z.files)}"
            )

    merged = {}
    for k in keys:
        cur = None
        for z in loaded:
            cur = _concat_value(cur, z[k])
        merged[k] = cur

    out_path = ROOT / f"{tag}{SUFFIX}"
    np.savez_compressed(out_path, **merged)
    print(f"[OK] {tag}: merged {len(files)} files -> {out_path}")

    for z in loaded:
        z.close()

    return out_path


def main():
    if not ROOT.exists():
        raise SystemExit(f"[ERROR] Not found: {ROOT}")

    pat = re.compile(r"^(?P<y>\d{4})-(?P<m>\d{2})" + re.escape(SUFFIX) + r"$")

    # Build index: "YYYY-MM" -> Path
    month2path: dict[str, Path] = {}
    for p in sorted(ROOT.glob(f"*{SUFFIX}")):
        m = pat.match(p.name)
        if not m:
            continue
        y = int(m.group("y"))
        mm = int(m.group("m"))
        month2path[f"{y:04d}-{mm:02d}"] = p

    if not month2path:
        raise SystemExit(f"[ERROR] No matching files like YYYY-MM{SUFFIX} under {ROOT}")

    # Merge each AUT Train window
    for (start_m, end_m) in TRAIN_WINDOWS:
        months = iter_months(start_m, end_m)
        missing = [m for m in months if m not in month2path]
        if missing:
            print(f"[SKIP] {start_m}to{end_m}: missing months {missing}")
            continue

        files = [month2path[m] for m in months]
        tag = f"{start_m}to{end_m}"
        merge_npz_files(tag, files)


if __name__ == "__main__":
    main()
