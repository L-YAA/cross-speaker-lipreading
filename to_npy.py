# to_npy.py
# -*- coding: utf-8 -*-
"""
Convert RAFT flow JPG/PNG folders -> single .npy per folder.

Input:  /path/to/flow_root/.../<rel_key>/flow_000001.jpg ...
Output: /path/to/flow_root/.../<rel_key>.npy

Key fixes:
- Atomic save without np.save auto-appending ".npy" to temp file
- Low-memory write: preallocate (memmap) then fill frame-by-frame (no frames list, no stack)
- Cap fail records to avoid memory blow when many failures
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Tuple, List
import multiprocessing as mp

import cv2
import numpy as np
from tqdm import tqdm

FLOW_NAME_RE = re.compile(r"^flow_(\d{6})\.(jpg|jpeg|png)$", re.IGNORECASE)

# ---------- globals for mp ----------
_G_OVERWRITE = False
_G_DTYPE = "float16"   # "uint8" | "float16" | "float32"
_G_KEEP_FAILS = 2000   # max fail records kept in memory


def _list_flow_dirs(flow_root: Path) -> List[str]:
    """Find all directories under flow_root that contain flow_XXXXXX.(jpg/png)."""
    out = []
    for root, dirs, files in os.walk(str(flow_root)):
        if any(FLOW_NAME_RE.match(fn) for fn in files):
            out.append(root)
            dirs[:] = []  # prune
    out.sort()
    return out


def _safe_sorted_flow_names(flow_dir: Path) -> List[str]:
    names = [n for n in os.listdir(flow_dir) if FLOW_NAME_RE.match(n)]
    # sort by index
    names.sort(key=lambda x: int(FLOW_NAME_RE.match(x).group(1)))
    return names


def _convert_frame(im_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 -> RGB in desired dtype."""
    im = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)

    if _G_DTYPE == "uint8":
        return im  # keep 0..255 uint8

    # float16/float32: normalize to 0..1
    im_f = im.astype(np.float32) / 255.0
    if _G_DTYPE == "float16":
        return im_f.astype(np.float16)
    return im_f.astype(np.float32)


def _read_one_dir_to_npy(flow_dir_str: str) -> Tuple[str, str, str]:
    """
    Returns (status, dir, info)
      status in {"ok","skip","fail"}
    """
    flow_dir = Path(flow_dir_str)
    out_npy = Path(str(flow_dir) + ".npy")

    try:
        if (not _G_OVERWRITE) and out_npy.exists() and out_npy.stat().st_size > 0:
            return ("skip", flow_dir_str, "exists")

        names = _safe_sorted_flow_names(flow_dir)
        if not names:
            return ("fail", flow_dir_str, "no flow_*.jpg/png found")

        # read first frame to get size
        first_fp = flow_dir / names[0]
        first = cv2.imread(str(first_fp), cv2.IMREAD_COLOR)
        if first is None:
            return ("fail", flow_dir_str, f"cv2.imread failed: {first_fp}")

        H, W = first.shape[:2]
        T = len(names)

        # dtype mapping
        if _G_DTYPE == "uint8":
            np_dtype = np.uint8
        elif _G_DTYPE == "float32":
            np_dtype = np.float32
        else:
            np_dtype = np.float16

        # atomic temp path (IMPORTANT: avoid np.save auto suffix issue)
        tmp_path = Path(str(out_npy) + ".tmp")          # e.g. xxx.npy.tmp (NOT a .npy file necessarily)
        mm_path = Path(str(out_npy) + ".tmp.mmap")      # memmap backing file

        # preallocate memmap file: [T,H,W,3]
        mm = np.memmap(mm_path, dtype=np_dtype, mode="w+", shape=(T, H, W, 3))

        # write frame 0
        mm[0] = _convert_frame(first)

        # write remaining frames
        for i, fn in enumerate(names[1:], start=1):
            fp = flow_dir / fn
            im = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            if im is None:
                # cleanup
                try:
                    del mm
                except Exception:
                    pass
                if mm_path.exists():
                    mm_path.unlink(missing_ok=True)
                return ("fail", flow_dir_str, f"cv2.imread failed: {fp}")

            if im.shape[0] != H or im.shape[1] != W:
                im = cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR)

            mm[i] = _convert_frame(im)

        # flush memmap to disk then save as .npy atomically
        mm.flush()
        del mm

        # Load memmap back as ndarray view and save to tmp_path using file handle
        mm2 = np.memmap(mm_path, dtype=np_dtype, mode="r", shape=(T, H, W, 3))

        # Save with open(...) so np.save will NOT append ".npy"
        with open(tmp_path, "wb") as f:
            np.save(f, np.asarray(mm2))

        del mm2
        mm_path.unlink(missing_ok=True)

        os.replace(tmp_path, out_npy)
        return ("ok", flow_dir_str, f"T={T} HW={H}x{W} dtype={_G_DTYPE}")

    except Exception as e:
        return ("fail", flow_dir_str, repr(e))


def _init_worker(overwrite: bool, dtype: str, keep_fails: int):
    global _G_OVERWRITE, _G_DTYPE, _G_KEEP_FAILS
    _G_OVERWRITE = bool(overwrite)
    _G_DTYPE = str(dtype)
    _G_KEEP_FAILS = int(keep_fails)


def parse_args():
    p = argparse.ArgumentParser("Convert flow jpg folders to npy (fixed & low-mem)")
    p.add_argument("--flow_root", type=str, required=True, help="root of flow folders")
    p.add_argument("--workers", type=int, default=8, help="num multiprocessing workers")
    p.add_argument("--overwrite", action="store_true", help="overwrite existing .npy")
    p.add_argument("--dtype", type=str, default="float16",
                   choices=["uint8", "float16", "float32"],
                   help="storage dtype; uint8 smallest, float16 good tradeoff, float32 largest")
    p.add_argument("--report", type=str, default="", help="optional json report path")
    p.add_argument("--keep_fails", type=int, default=2000, help="keep at most N fail records in memory/report")
    p.add_argument("--chunksize", type=int, default=8, help="pool.imap_unordered chunksize")
    return p.parse_args()


def main():
    args = parse_args()
    flow_root = Path(args.flow_root)
    assert flow_root.is_dir(), f"flow_root not found: {flow_root}"

    sample_dirs = _list_flow_dirs(flow_root)
    print(f"[Scan] flow_root={flow_root} sample_dirs={len(sample_dirs)} "
          f"workers={args.workers} overwrite={args.overwrite} dtype={args.dtype}")

    ok = skip = fail = 0
    fails = []

    def _handle(status, d, info):
        nonlocal ok, skip, fail, fails
        if status == "ok":
            ok += 1
        elif status == "skip":
            skip += 1
        else:
            fail += 1
            if len(fails) < args.keep_fails:
                fails.append({"dir": d, "info": info})

    if args.workers <= 1:
        _init_worker(args.overwrite, args.dtype, args.keep_fails)
        for status, d, info in tqdm(map(_read_one_dir_to_npy, sample_dirs),
                                    total=len(sample_dirs), ncols=120):
            _handle(status, d, info)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(args.overwrite, args.dtype, args.keep_fails),
            maxtasksperchild=200
        ) as pool:
            it = pool.imap_unordered(_read_one_dir_to_npy, sample_dirs, chunksize=args.chunksize)
            for status, d, info in tqdm(it, total=len(sample_dirs), ncols=120):
                _handle(status, d, info)

    print(f"[Done] ok={ok} skip={skip} fail={fail}")
    if fail > 0:
        print("[Fail][Example]", fails[0] if fails else "(fail records capped)")

    if args.report:
        rep = {
            "flow_root": str(flow_root),
            "total": len(sample_dirs),
            "ok": ok,
            "skip": skip,
            "fail": fail,
            "dtype": args.dtype,
            "fails_kept": len(fails),
            "fails": fails,
        }
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        print(f"[Report] saved -> {args.report}")


if __name__ == "__main__":
    main()
