#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, io, json, tarfile, argparse
from pathlib import Path
from collections import defaultdict, deque

FLOW_RE = re.compile(r"^flow_(\d{6})\.(jpg|jpeg|png)$", re.IGNORECASE)

def natural_key(s: str):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def safe_key(s: str) -> str:
    # webdataset uses "<key>.<ext>" grouping; avoid '.' in key
    return (s.replace("/", "__")
             .replace("\\", "__")
             .replace("..", "_")
             .replace(".", "_")
             .replace(":", "_"))

def list_flow_frames(flow_dir: Path):
    if not flow_dir.is_dir():
        return []
    names = [p.name for p in flow_dir.iterdir() if p.is_file() and FLOW_RE.match(p.name)]
    names.sort(key=natural_key)
    return names

def discover_samples(flow_root: Path, speaker_prefix="s"):
    """
    Return samples:
      {"spk": "s5", "rel_key": "20170324/section_xxx", "flow_dir": "/.../s5/20170324/section_xxx"}
    """
    samples = []
    spk_dirs = [p for p in flow_root.iterdir() if p.is_dir() and p.name.startswith(speaker_prefix)]
    spk_dirs.sort(key=lambda p: natural_key(p.name))

    for spk_dir in spk_dirs:
        spk = spk_dir.name
        for root, dirs, files in os.walk(spk_dir):
            # if this dir contains any flow_*.jpg, treat as a sample dir
            if any(FLOW_RE.match(fn) for fn in files):
                rootp = Path(root)
                rel_key = str(rootp.relative_to(spk_dir))
                samples.append({"spk": spk, "rel_key": rel_key, "flow_dir": str(rootp)})
    return samples

def make_round_robin_order(samples, seed=42):
    import random
    random.seed(seed)
    by_spk = defaultdict(list)
    for s in samples:
        by_spk[s["spk"]].append(s)
    # shuffle within speaker
    for spk, lst in by_spk.items():
        random.shuffle(lst)
    spks = sorted(by_spk.keys(), key=natural_key)
    qs = {spk: deque(by_spk[spk]) for spk in spks}
    out = []
    alive = True
    while alive:
        alive = False
        for spk in spks:
            if qs[spk]:
                out.append(qs[spk].popleft())
                alive = True
    return out

def write_tar_member(tar: tarfile.TarFile, name: str, data: bytes):
    ti = tarfile.TarInfo(name=name)
    ti.size = len(data)
    tar.addfile(ti, io.BytesIO(data))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--speaker_prefix", type=str, default="s")
    ap.add_argument("--target_gb", type=float, default=2.0, help="target shard size (approx) GB")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_empty", action="store_true", help="skip dirs with 0 frames")
    ap.add_argument("--min_frames", type=int, default=1, help="skip if frames < min_frames")
    args = ap.parse_args()

    flow_root = Path(args.flow_root)
    out_dir = Path(args.out_dir)
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(flow_root, speaker_prefix=args.speaker_prefix)
    print(f"[Scan] samples={len(samples)} under {flow_root}")

    ordered = make_round_robin_order(samples, seed=args.seed)

    target_bytes = int(args.target_gb * (1024**3))
    shard_idx = 0
    cur_bytes = 0
    written = 0
    skipped = 0

    def shard_path(i): return shard_dir / f"shard-{i:06d}.tar"

    tar_path = shard_path(shard_idx)
    tar = tarfile.open(tar_path, "w")
    print(f"[Shard] open {tar_path}")

    for s in ordered:
        spk = s["spk"]
        rel_key = s["rel_key"]
        flow_dir = Path(s["flow_dir"])

        frames = list_flow_frames(flow_dir)
        if len(frames) < args.min_frames:
            skipped += 1
            continue
        if len(frames) == 0 and args.skip_empty:
            skipped += 1
            continue

        key = safe_key(f"{spk}/{rel_key}")

        meta = {
            "spk": spk,
            "rel_key": rel_key,
            "flow_dir": str(flow_dir),
            "n_frames": int(len(frames)),
        }
        meta_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")
        write_tar_member(tar, f"{key}.meta.json", meta_bytes)
        cur_bytes += len(meta_bytes)

        # write raw jpg bytes; NO decode, NO resize, NO frame sampling
        for fn in frames:
            fp = flow_dir / fn
            try:
                b = fp.read_bytes()
            except Exception:
                b = b""
            write_tar_member(tar, f"{key}.{fn.lower()}", b)
            cur_bytes += len(b)

        written += 1

        if cur_bytes >= target_bytes:
            tar.close()
            print(f"[Shard] close {tar_path} size≈{cur_bytes/1024/1024:.1f}MB samples_total={written}")
            shard_idx += 1
            cur_bytes = 0
            tar_path = shard_path(shard_idx)
            tar = tarfile.open(tar_path, "w")
            print(f"[Shard] open {tar_path}")

        if written % 1000 == 0:
            print(f"[Progress] written={written} skipped={skipped}")

    tar.close()
    print(f"[Done] written={written} skipped={skipped} shards={shard_idx+1} dir={shard_dir}")

if __name__ == "__main__":
    main()
