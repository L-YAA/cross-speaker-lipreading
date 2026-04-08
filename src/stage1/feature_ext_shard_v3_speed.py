#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage1 Personality Feature Training - DataParallel version (v3, SPEED FIX)
Major fixes for "training extremely slow + GPU util 0% + slow start":
1) Shard-aware batching (group samples by tar shard) => dramatically reduce tar open thrash.
2) Fast tar member lookup + direct byte read by offset/size (O(1) dict) => avoid linear getmember().
3) PyAV decode path fixed: use seek-by-timestamp for sparse static frames (avoid decoding whole video).
4) Faster CPU preprocessing: cv2.resize + numpy normalize (avoid torch.interpolate on CPU).
5) Avoid pointless per-batch index_select copies when y>=0 for all samples.
6) Thread oversubscription controls in DataLoader workers (OpenCV/PyTorch CPU threads).

Notes:
- Flow shard member naming keeps i:06d (0-based): {key}.flow_{i:06d}.jpg/png/jpeg.
- This script is a drop-in replacement for your v2, with extra args for speed control.
"""

import os
import re
import json
import random
import argparse
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional, Any, Union, Iterable
from functools import lru_cache
import glob
import tarfile
from collections import OrderedDict
import atexit
import math
import torch
import os

# 设置 OpenMP 和 MKL 线程数，避免训练过程中修改
os.environ["OMP_NUM_THREADS"] = "4"  # 根据你的机器调整
os.environ["MKL_NUM_THREADS"] = "4"  # 根据你的机器调整
torch.set_num_threads(4)  # 设置线程数，避免训练期间修改

# -------------------------
# 0) parse args & set CUDA_VISIBLE_DEVICES (MUST before import torch)
# -------------------------
def parse_args():
    p = argparse.ArgumentParser("Stage1 Personality Feature Training - DP + AdaptiveLoss + MI + FlowShards (v3 speed)")

    # train paths
    p.add_argument("--video_root", type=str,
                   default="/mnt/netdisk/dataset/lipreading/datasets/CMLR/cmlr/cmlr_video_seg24s/",
                   help="root of original videos (train)")
    p.add_argument("--flow_root", type=str,
                   default="/home/liuyang/Project/flow/flow_sequence/cmlr2/",
                   help="root of flow frames (train) OR shard out_dir OR shard_dir(shards/)")

    # external val (unseen speakers)
    p.add_argument("--val_video_root", type=str,
                   default="/mnt/netdisk/dataset/lipreading/mpc/preprocess_datasets/CMLR_extra_latest/cmlr_extra/cmlr_extra_video_seg24s/",
                   help="external val video root (unseen speakers)")
    p.add_argument("--val_flow_root", type=str,
                   default="/home/liuyang/Project/flow/flow_sequence/cmlr_extra/",
                   help="external val flow root OR shard out_dir OR shard_dir(shards/)")
    p.add_argument("--disable_external_val", action="store_true", help="do not run external val")

    # debug
    p.add_argument("--debug_shapes", action="store_true", help="print initial shapes once")
    p.add_argument("--print_bad_sample", action="store_true", help="print bad sample path in dataset exception")

    # ---- speed: video decode controls ----
    p.add_argument("--use_av", action="store_true",
                   help="use PyAV with seek for sparse frame sampling. If av not installed, auto fallback to cv2.")
    p.add_argument("--av_threads", type=int, default=2,
                   help="PyAV decode threads (0/1/2/4...). Usually 2 is safe; too high may hurt with many workers.")

    # gpu
    p.add_argument("--gpus", type=str, default="0", help="physical GPU ids, e.g. '0' or '0,2'")
    p.add_argument("--no_cuda", action="store_true", help="force CPU")

    # speed switches
    p.add_argument("--tf32", action="store_true", help="enable TF32 (Ampere+), faster matmul/conv")
    p.add_argument("--cudnn_benchmark", action="store_true", help="enable cudnn benchmark (faster for fixed shapes)")

    # output
    p.add_argument("--save_dir", type=str, default="./ch_stage1_2")
    p.add_argument("--exp_name", type=str, default="personality_stage1_dp_v3_speed")

    # index cache (accelerate scanning)
    p.add_argument("--index_cache", type=str, default="",
                   help="path to index cache file (.pt). empty => auto in save_dir")
    p.add_argument("--rebuild_index", action="store_true", help="force rebuild index cache even if exists")

    # shard meta index cache
    p.add_argument("--flow_shard_index_cache", type=str, default="",
                   help="cache for TRAIN flow shard meta index (.pt). empty => auto in save_dir")
    p.add_argument("--val_flow_shard_index_cache", type=str, default="",
                   help="cache for VAL flow shard meta index (.pt). empty => auto in save_dir")

    # resume / pretrained
    p.add_argument("--resume", type=str, default="", help="resume training from checkpoint (last/best)")
    p.add_argument("--pretrained", type=str, default="", help="load pretrained weights only (model_state)")
    p.add_argument("--reset_optimizer", action="store_true",
                   help="when --resume, ignore optimizer/scaler and restart optimization")

    # train
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)  # DP: global batch (split across visible GPUs)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--use_amp", action="store_true", help="enable AMP")
    p.add_argument("--seed", type=int, default=42)

    # LR schedule
    p.add_argument("--cosine", action="store_true", help="use cosine annealing LR scheduler")

    # shapes
    p.add_argument("--frame_h", type=int, default=112)
    p.add_argument("--frame_w", type=int, default=112)
    p.add_argument("--static_frames", type=int, default=2)
    p.add_argument("--flow_frames", type=int, default=32)
    p.add_argument("--flow_channels", type=int, default=3, help="RGB flow visualization usually=3")

    # model
    p.add_argument("--feat_dim", type=int, default=128)
    p.add_argument("--freeze_resnet_until", type=str, default="layer3",
                   help="freeze resnet until this layer name; use 'none' to disable")

    # losses (base)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--lambda_indep", type=float, default=1.0)

    # MI suppression
    p.add_argument("--lambda_mi", type=float, default=0.2, help="scale for MI suppression (margin anti-InfoNCE)")
    p.add_argument("--mi_tau", type=float, default=0.07, help="temperature for MI suppression")
    p.add_argument("--mi_warmup_epochs", type=int, default=2, help="warmup epochs for MI strength ramp")
    p.add_argument("--mi_max_strength", type=float, default=1.0, help="max ramp multiplier for MI loss")

    # split
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--split_by_speaker", action="store_true")

    # debug scan limit
    p.add_argument("--max_samples", type=int, default=-1, help="limit samples for debug (train scan only)")

    # tar cache
    p.add_argument("--tar_cache_max_open", type=int, default=16,
                   help="max open tar files per worker process (increase if many shards per epoch)")

    # -------- NEW: speed knobs --------
    p.add_argument("--batch_by_shard", action="store_true",
                   help="(recommended for shard backend) batch samples by tar shard to avoid open/close thrash")
    p.add_argument("--no_batch_by_shard", action="store_true",
                   help="disable shard-aware batching even if shard backend detected")

    p.add_argument("--return_fp16", action="store_true",
                   help="dataset returns float16 tensors to reduce CPU->GPU bandwidth/memory (recommended when --use_amp)")
    p.add_argument("--no_return_fp16", action="store_true",
                   help="force dataset to return float32 even with AMP")

    p.add_argument("--prefetch_factor", type=int, default=4,
                   help="DataLoader prefetch_factor (per worker). Larger may help if I/O is fast enough.")
    p.add_argument("--log_every", type=int, default=50,
                   help="print perf statistics every N train iters (smaller => faster diagnosis)")
    p.add_argument("--enable_framefolder_video", action="store_true",
                   help="treat frame folders as videos during index scan (slower). Default off for faster startup.")

    p.add_argument("--index_scan_workers", type=int, default=0,
                   help="thread workers for index scan across speakers (0=>disable). Useful when --rebuild_index.")

    return p.parse_args()


args = parse_args()

# ---- CPU thread oversubscription guard (can override via env) ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

if not args.no_cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

# -------------------------
# 1) import torch & deps
# -------------------------
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.models as models
from tqdm import tqdm

# optional: PyAV
try:
    import av  # pip install av  (or conda install -c conda-forge av)
    _HAS_AV = True
except Exception:
    av = None
    _HAS_AV = False


# =========================
# 2) Config
# =========================
@dataclass
class Config:
    video_root: str
    flow_root: str

    val_video_root: str
    val_flow_root: str
    disable_external_val: bool

    save_dir: str
    exp_name: str

    epochs: int
    batch_size: int
    num_workers: int
    lr: float
    weight_decay: float
    grad_clip: float
    use_amp: bool
    seed: int

    use_cosine: bool

    frame_size: Tuple[int, int]
    static_frames: int
    flow_frames: int
    flow_channels: int

    feat_dim: int
    freeze_resnet_until: Optional[str]

    label_smoothing: float
    lambda_indep: float

    lambda_mi: float
    mi_tau: float
    mi_warmup_epochs: int
    mi_max_strength: float

    val_ratio: float
    split_by_speaker: bool

    # speed
    use_av: bool = False
    av_threads: int = 2
    return_fp16: bool = False
    enable_framefolder_video: bool = False

    speaker_prefix: str = "s"
    unseen_speaker_id: int = -2
    broken_sample_id: int = -1

    video_exts: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    img_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def build_config(a) -> Config:
    frz = None if str(a.freeze_resnet_until).lower() in ("none", "null", "") else a.freeze_resnet_until

    # default: if AMP => return fp16 unless explicitly disabled
    if a.no_return_fp16:
        ret_fp16 = False
    elif a.return_fp16:
        ret_fp16 = True
    else:
        ret_fp16 = bool(a.use_amp) and (not a.no_cuda)

    return Config(
        video_root=a.video_root,
        flow_root=a.flow_root,
        val_video_root=a.val_video_root,
        val_flow_root=a.val_flow_root,
        disable_external_val=bool(a.disable_external_val),

        save_dir=a.save_dir,
        exp_name=a.exp_name,

        epochs=a.epochs,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        lr=a.lr,
        weight_decay=a.weight_decay,
        grad_clip=a.grad_clip,
        use_amp=a.use_amp,
        seed=a.seed,

        use_cosine=bool(a.cosine),

        frame_size=(a.frame_h, a.frame_w),
        static_frames=a.static_frames,
        flow_frames=a.flow_frames,
        flow_channels=a.flow_channels,

        feat_dim=a.feat_dim,
        freeze_resnet_until=frz,

        label_smoothing=a.label_smoothing,
        lambda_indep=a.lambda_indep,

        lambda_mi=a.lambda_mi,
        mi_tau=a.mi_tau,
        mi_warmup_epochs=a.mi_warmup_epochs,
        mi_max_strength=a.mi_max_strength,

        val_ratio=a.val_ratio,
        split_by_speaker=a.split_by_speaker,

        use_av=bool(a.use_av),
        av_threads=int(a.av_threads),
        return_fp16=bool(ret_fp16),
        enable_framefolder_video=bool(a.enable_framefolder_video),
    )


# =========================
# 3) utils
# =========================
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def seed_worker_fn(base_seed: int):
    def _seed_worker(worker_id: int):
        s = base_seed + worker_id
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)

        # prevent oversubscription inside workers
        try:
            torch.set_num_threads(1)
            # ❌ 不要在 worker 里调用 set_num_interop_threads
            # torch.set_num_interop_threads(1)
        except Exception:
            pass
        try:
            cv2.setNumThreads(0)
        except Exception:
            pass
    return _seed_worker


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def is_video_file(p: str, exts: Tuple[str, ...]) -> bool:
    return os.path.splitext(p)[1].lower() in exts

def is_image_file(p: str, exts: Tuple[str, ...]) -> bool:
    return os.path.splitext(p)[1].lower() in exts

def list_speakers(root: str, prefix: str) -> List[str]:
    if (root is None) or (not os.path.isdir(root)):
        return []
    sps = []
    for name in os.listdir(root):
        full = os.path.join(root, name)
        if os.path.isdir(full) and name.startswith(prefix):
            sps.append(name)
    sps.sort(key=natural_key)
    return sps

def sample_indices(n: int, t: int, *, mode: str = "linspace", avoid_ends: bool = False) -> np.ndarray:
    """
    mode:
      - 'linspace': deterministic evenly spaced
      - 'rand': random indices (sorted)
    avoid_ends: avoid picking exactly 0 and n-1 (useful to avoid decoding to last frame)
    """
    if t <= 0:
        return np.zeros((0,), dtype=np.int64)
    if n <= 0:
        return np.zeros((t,), dtype=np.int64)
    if n == 1:
        return np.zeros((t,), dtype=np.int64)

    if mode == "rand":
        idx = np.random.randint(0, n, size=(t,), dtype=np.int64)
        idx.sort()
        return idx

    # linspace
    if avoid_ends and n >= (t + 2):
        # generate t points between (0, n-1) excluding ends
        f = np.linspace(0, n - 1, t + 2, dtype=np.float32)[1:-1]
        return np.round(f).astype(np.int64)
    return np.linspace(0, n - 1, t, dtype=np.float32).round().astype(np.int64)

def linear_ramp(epoch_idx: int, warmup: int, max_strength: float) -> float:
    if warmup <= 0:
        return max_strength
    if epoch_idx < warmup:
        return max_strength * float(epoch_idx + 1) / float(warmup)
    return max_strength

def math_log(x: int, device: torch.device):
    return torch.log(torch.tensor(float(x), device=device) + 1e-12)


# =========================
# 3.1) Flow shard helpers
# =========================
def safe_key(s: str) -> str:
    return (s.replace("/", "__")
             .replace("\\", "__")
             .replace("..", "_")
             .replace(".", "_")
             .replace(":", "_"))

def _detect_shard_dir(flow_root: str) -> Optional[str]:
    """
    Accept:
      - out_dir (contains shards/)
      - shards dir directly (contains shard-*.tar)
      - normal flow dir (no shards)
    """
    if not flow_root:
        return None
    if not os.path.isdir(flow_root):
        return None

    # 1) flow_root directly has shard-*.tar
    if len(glob.glob(os.path.join(flow_root, "shard-*.tar"))) > 0:
        return flow_root

    # 2) flow_root/shards has shard-*.tar
    cand = os.path.join(flow_root, "shards")
    if os.path.isdir(cand) and len(glob.glob(os.path.join(cand, "shard-*.tar"))) > 0:
        return cand

    return None

def _list_tar_shards(shard_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(shard_dir, "shard-*.tar")))

def build_flow_shard_index(shard_dir: str, cache_path: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    index[(spk, rel_key)] = {
        "type": "shard",
        "tar": "/path/shard-000123.tar",
        "key": "<safe_key(spk/rel_key)>",
        "n_frames": int,
        "spk": spk,
        "rel_key": rel_key,
    }
    Only reads *.meta.json from tar. Cached to .pt.
    """
    if os.path.isfile(cache_path):
        try:
            idx = torch.load(cache_path, map_location="cpu")
            if isinstance(idx, dict) and len(idx) > 0:
                return idx
        except Exception:
            pass

    idx: Dict[Tuple[str, str], Dict[str, Any]] = {}
    tars = _list_tar_shards(shard_dir)
    print(f"[ShardIndex] scanning metas: dir={shard_dir} tars={len(tars)}")

    # IMPORTANT: this scan can be slow once; cache it and avoid --rebuild_index in later runs.
    for tp in tars:
        try:
            with tarfile.open(tp, "r:*") as tar:
                for m in tar:
                    if (not m.isfile()) or (not m.name.endswith(".meta.json")):
                        continue
                    f = tar.extractfile(m)
                    if f is None:
                        continue
                    b = f.read()
                    try:
                        meta = json.loads(b.decode("utf-8", errors="ignore"))
                    except Exception:
                        continue
                    spk = meta.get("spk", None)
                    rel_key = meta.get("rel_key", None)
                    n_frames = int(meta.get("n_frames", 0))
                    if (spk is None) or (rel_key is None) or (n_frames <= 0):
                        continue
                    k = safe_key(f"{spk}/{rel_key}")
                    idx[(spk, rel_key)] = {
                        "type": "shard",
                        "tar": tp,
                        "key": k,
                        "n_frames": n_frames,
                        "spk": spk,
                        "rel_key": rel_key,
                    }
        except Exception as e:
            print(f"[ShardIndex][Warn] tar read failed: {tp} err={repr(e)}")

    tmp = cache_path + ".tmp"
    torch.save(idx, tmp)
    os.replace(tmp, cache_path)
    print(f"[ShardIndex] saved: {cache_path} items={len(idx)}")
    return idx


# =========================
# 3.2) Tar LRU cache (FAST: name->(offset,size) dict + close on eviction)
# =========================
class TarLRUCache:
    """
    Cache tar handles per worker process. Each cache entry holds:
      - tarfile.TarFile
      - name2pos: dict[str] -> (offset_data:int, size:int)
    This makes per-frame access O(1) instead of linear getmember().
    """
    def __init__(self, max_open: int = 16):
        self.max_open = int(max_open)
        self._cache: "OrderedDict[str, Tuple[tarfile.TarFile, Dict[str, Tuple[int,int]]]]" = OrderedDict()

    def get(self, tar_path: str) -> Tuple[tarfile.TarFile, Dict[str, Tuple[int,int]]]:
        v = self._cache.get(tar_path, None)
        if v is not None:
            self._cache.move_to_end(tar_path, last=True)
            return v

        tar = tarfile.open(tar_path, "r:*")
        # build O(1) index: keep only needed suffixes to reduce memory
        keep_suffix = (".jpg", ".jpeg", ".png", ".meta.json")
        name2pos: Dict[str, Tuple[int,int]] = {}
        try:
            for m in tar.getmembers():
                if (not m.isfile()):
                    continue
                if not m.name.endswith(keep_suffix):
                    continue
                # offset_data points to file data in the tar stream
                name2pos[m.name] = (int(m.offset_data), int(m.size))
        except Exception:
            # fallback: still keep empty dict; extract will fail -> handled
            name2pos = {}

        self._cache[tar_path] = (tar, name2pos)
        self._cache.move_to_end(tar_path, last=True)

        while len(self._cache) > self.max_open:
            old_path, (old_tar, _) = self._cache.popitem(last=False)
            try:
                old_tar.close()
            except Exception:
                pass
        return tar, name2pos

    def close_all(self):
        for _, (t, _) in list(self._cache.items()):
            try:
                t.close()
            except Exception:
                pass
        self._cache.clear()


_TAR_CACHE = TarLRUCache(max_open=args.tar_cache_max_open)
atexit.register(_TAR_CACHE.close_all)


def _read_tar_member_bytes_fast(tar: tarfile.TarFile,
                               name2pos: Dict[str, Tuple[int,int]],
                               name: str) -> bytes:
    """
    Fast path: seek+read by offset/size.
    """
    pos = name2pos.get(name, None)
    if pos is None:
        return b""
    off, size = pos
    try:
        tar.fileobj.seek(off)
        return tar.fileobj.read(size)
    except Exception:
        # fallback to tar.extractfile (slower)
        try:
            f = tar.extractfile(name)
            if f is None:
                return b""
            return f.read()
        except Exception:
            return b""


def decode_jpg_bytes_to_rgb_uint8(b: bytes) -> Optional[np.ndarray]:
    if not b:
        return None
    arr = np.frombuffer(b, dtype=np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if im is None:
        return None
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im  # uint8


def read_flow_from_shards(flow_ref: Dict[str, Any], indices: np.ndarray,
                          fallback_hw: Tuple[int,int],
                          try_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png")) -> Tuple[List[np.ndarray], int]:
    """
    return (frames_uint8, miss_count)
    NOTE: shard flow names are 0-based: flow_{i:06d}
    """
    H, W = fallback_hw
    n_frames = int(flow_ref.get("n_frames", 0))
    if n_frames <= 0:
        return [], int(indices.size)

    tar, name2pos = _TAR_CACHE.get(flow_ref["tar"])
    key = flow_ref["key"]

    out: List[np.ndarray] = []
    fallback = np.zeros((H, W, 3), dtype=np.uint8)
    miss = 0

    for i in indices:
        i = int(i)
        i = max(0, min(i, n_frames - 1))

        im = None
        for ext in try_exts:
            member = f"{key}.flow_{i:06d}{ext}"
            b = _read_tar_member_bytes_fast(tar, name2pos, member)
            im = decode_jpg_bytes_to_rgb_uint8(b)
            if im is not None:
                break

        if im is None:
            miss += 1
            im = fallback
        else:
            fallback = im
        out.append(im)

    return out, miss


# =========================
# 3.5) checkpoint helpers (DP/DDP/Single compatible)
# =========================
def _unwrap_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not sd:
        return sd
    keys = list(sd.keys())
    if len(keys) > 0 and all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in sd.items()}
    return sd

def _maybe_wrap_module_prefix(sd: Dict[str, torch.Tensor], need_module: bool) -> Dict[str, torch.Tensor]:
    if not sd:
        return sd
    keys = list(sd.keys())
    has_module = (len(keys) > 0 and all(k.startswith("module.") for k in keys))
    if need_module and (not has_module):
        return {("module." + k): v for k, v in sd.items()}
    if (not need_module) and has_module:
        return {k[len("module."):]: v for k, v in sd.items()}
    return sd

def _get_model_state_dict(model: nn.Module) -> Dict:
    return model.state_dict()

def load_checkpoint(path: str):
    return torch.load(path, map_location="cpu")

def smart_load_model_state(model: nn.Module, state: Dict, strict: bool = True):
    need_module = isinstance(model, nn.DataParallel)
    state = _maybe_wrap_module_prefix(_unwrap_state_dict(state), need_module=need_module)
    try:
        model.load_state_dict(state, strict=strict)
        return True, ""
    except Exception as e:
        return False, repr(e)

def save_checkpoint(path: str,
                    cfg: Config,
                    speaker2id: Dict[str, int],
                    model: nn.Module,
                    opt: Optional[torch.optim.Optimizer],
                    scaler: Optional[torch.cuda.amp.GradScaler],
                    epoch: int,
                    best_val_acc: float,
                    ps0=None,
                    pd0=None):
    ckpt = {
        "cfg": asdict(cfg),
        "speaker2id": speaker2id,
        "epoch": int(epoch),
        "best_val_acc": float(best_val_acc),
        "model_state": _get_model_state_dict(model),
        "optimizer_state": opt.state_dict() if opt is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
    }
    if ps0 is not None:
        ckpt["ps0"] = ps0
    if pd0 is not None:
        ckpt["pd0"] = pd0

    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


# =========================
# 4) scan: rel_key = relative path inside speaker dir w/o ext
# =========================
def discover_video_items_with_relkey(speaker_video_dir: str, cfg: Config) -> List[Tuple[str, str]]:
    """
    Faster scan:
    - Default: only video files.
    - Optionally include frame folders when cfg.enable_framefolder_video=True (slower).
    """
    items: List[Tuple[str, str]] = []

    for root, dirs, files in os.walk(speaker_video_dir):
        for fn in files:
            fp = os.path.join(root, fn)
            if is_video_file(fp, cfg.video_exts):
                rel = os.path.relpath(fp, speaker_video_dir)
                rel_no_ext = os.path.splitext(rel)[0]
                items.append((fp, rel_no_ext))

        if cfg.enable_framefolder_video:
            # optional: frame folders as "video" (SLOW)
            for d in dirs:
                dp = os.path.join(root, d)
                try:
                    # quick check: only scan a few entries
                    cnt = 0
                    with os.scandir(dp) as it:
                        for e in it:
                            if e.is_file():
                                ext = os.path.splitext(e.name)[1].lower()
                                if ext in cfg.img_exts:
                                    cnt += 1
                                    if cnt >= 3:
                                        break
                    if cnt >= 3:
                        rel = os.path.relpath(dp, speaker_video_dir)
                        items.append((dp, rel))
                except Exception:
                    pass

    # dedup by rel_key, prefer video file
    uniq: Dict[str, str] = {}
    for path, rel_key in items:
        if rel_key not in uniq:
            uniq[rel_key] = path
        else:
            if is_video_file(path, cfg.video_exts):
                uniq[rel_key] = path

    out = [(p, k) for k, p in uniq.items()]
    out.sort(key=lambda x: natural_key(x[1]))
    return out

def find_flow_by_relkey(flow_speaker_dir: str, rel_key: str) -> Optional[str]:
    cand_dir = os.path.join(flow_speaker_dir, rel_key)
    cand_npy = cand_dir + ".npy"
    if os.path.isdir(cand_dir):
        return cand_dir
    if os.path.isfile(cand_npy):
        return cand_npy
    return None


FlowRef = Union[str, Dict[str, Any]]  # str: dir/npy ; dict: shard ref

def build_samples_from_roots(video_root: str,
                             flow_root: str,
                             cfg: Config,
                             speaker2id: Optional[Dict[str, int]],
                             max_samples: int = -1,
                             unseen_speaker_id: int = -2,
                             shard_index: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None
                             ) -> Tuple[List[Tuple[str, FlowRef, int]], Dict[str, int], int, int]:
    """
    returns: samples, speaker2id_used, miss_flow_count, unseen_speaker_count
    """
    speakers = list_speakers(video_root, cfg.speaker_prefix)

    if speaker2id is None:
        speaker2id = {s: i for i, s in enumerate(speakers)}

    samples: List[Tuple[str, FlowRef, int]] = []
    miss_flow = 0
    unseen_set = set()

    use_shard = shard_index is not None

    # optional: parallel scan across speakers for faster rebuild_index
    if args.index_scan_workers and int(args.index_scan_workers) > 0:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        def _scan_one(spk: str):
            v_dir = os.path.join(video_root, spk)
            if not os.path.isdir(v_dir):
                return spk, []
            return spk, discover_video_items_with_relkey(v_dir, cfg)

        scanned: Dict[str, List[Tuple[str,str]]] = {}
        with ThreadPoolExecutor(max_workers=int(args.index_scan_workers)) as ex:
            futs = [ex.submit(_scan_one, spk) for spk in speakers]
            for fu in as_completed(futs):
                spk, items = fu.result()
                scanned[spk] = items
    else:
        scanned = {}

    for spk in speakers:
        v_dir = os.path.join(video_root, spk)
        if not os.path.isdir(v_dir):
            continue

        if spk in speaker2id:
            sid = speaker2id[spk]
        else:
            sid = unseen_speaker_id
            unseen_set.add(spk)

        if spk in scanned:
            video_items = scanned[spk]
        else:
            video_items = discover_video_items_with_relkey(v_dir, cfg)

        if (not use_shard):
            f_dir = os.path.join(flow_root, spk)

        for vpath, rel_key in video_items:
            if use_shard:
                ref = shard_index.get((spk, rel_key), None)
                if ref is None:
                    miss_flow += 1
                    continue
                samples.append((vpath, ref, sid))
            else:
                flow_path = find_flow_by_relkey(f_dir, rel_key) if os.path.isdir(f_dir) else None
                if flow_path is None:
                    miss_flow += 1
                    continue
                samples.append((vpath, flow_path, sid))

            if (max_samples > 0) and (len(samples) >= max_samples):
                break

        if (max_samples > 0) and (len(samples) >= max_samples):
            break

    return samples, speaker2id, miss_flow, len(unseen_set)


def split_train_val(samples: List[Tuple[str, FlowRef, int]], cfg: Config):
    known = [s for s in samples if s[2] >= 0]

    if cfg.split_by_speaker:
        spk_ids = sorted(list(set([s[2] for s in known])))
        random.shuffle(spk_ids)
        n_val = max(1, int(len(spk_ids) * cfg.val_ratio))
        val_spk = set(spk_ids[:n_val])
        train = [x for x in known if x[2] not in val_spk]
        val = [x for x in known if x[2] in val_spk]
        return train, val

    by_spk: Dict[int, List] = {}
    for s in known:
        by_spk.setdefault(s[2], []).append(s)

    train, val = [], []
    for sid, lst in by_spk.items():
        random.shuffle(lst)
        n_val = max(1, int(len(lst) * cfg.val_ratio)) if len(lst) >= 10 else max(1, int(len(lst) * 0.2))
        val.extend(lst[:n_val])
        train.extend(lst[n_val:])
    random.shuffle(train)
    random.shuffle(val)
    return train, val


def _is_shard_sample_list(samples: List[Tuple[str, FlowRef, int]]) -> bool:
    for _v, fref, _sid in samples[:50]:
        if isinstance(fref, dict) and fref.get("type", "") == "shard":
            return True
    return False


def build_index_with_cache(cfg: Config):
    if args.index_cache.strip():
        cache_path = args.index_cache
    else:
        cache_path = os.path.join(cfg.save_dir, f"{cfg.exp_name}.index_cache.dp.pt")

    os.makedirs(cfg.save_dir, exist_ok=True)

    use_cache = (not args.rebuild_index) and os.path.isfile(cache_path)
    if use_cache:
        try:
            pack = torch.load(cache_path, map_location="cpu")
            speaker2id = pack["speaker2id"]
            train_samples = pack["train_samples"]
            val_samples = pack["val_samples"]
            ext_val_samples = pack.get("ext_val_samples", [])
            info = pack.get("info", {})
            print(f"[IndexCache] loaded: {cache_path}")
            print(f"[IndexCache] train={len(train_samples)} val={len(val_samples)} ext={len(ext_val_samples)} info={info}")
            return speaker2id, train_samples, val_samples, ext_val_samples
        except Exception as e:
            print(f"[IndexCache][Warn] load failed -> rebuild. err={repr(e)}")

    # ---- TRAIN flow backend detect (dir/npy vs shards) ----
    train_shard_dir = _detect_shard_dir(cfg.flow_root)
    train_shard_index = None
    if train_shard_dir is not None:
        if args.flow_shard_index_cache.strip():
            train_shard_cache = args.flow_shard_index_cache
        else:
            train_shard_cache = os.path.join(cfg.save_dir, f"{cfg.exp_name}.train_flow_shard_index.pt")
        train_shard_index = build_flow_shard_index(train_shard_dir, train_shard_cache)
        print(f"[FlowBackend][Train] SHARD dir={train_shard_dir} index_items={len(train_shard_index)}")
    else:
        print(f"[FlowBackend][Train] FS dir/npy flow_root={cfg.flow_root}")

    all_train, speaker2id, miss_flow_tr, _ = build_samples_from_roots(
        cfg.video_root, cfg.flow_root, cfg,
        speaker2id=None,
        max_samples=args.max_samples,
        unseen_speaker_id=cfg.unseen_speaker_id,
        shard_index=train_shard_index
    )
    train_samples, val_samples = split_train_val(all_train, cfg)

    # ---- EXTERNAL VAL flow backend detect (dir/npy vs shards) ----
    ext_val_samples: List[Tuple[str, FlowRef, int]] = []
    miss_flow_ext = 0
    unseen_cnt = 0

    if (not cfg.disable_external_val) and os.path.isdir(cfg.val_video_root):
        val_shard_dir = _detect_shard_dir(cfg.val_flow_root)
        val_shard_index = None
        if val_shard_dir is not None:
            if args.val_flow_shard_index_cache.strip():
                val_shard_cache = args.val_flow_shard_index_cache
            else:
                val_shard_cache = os.path.join(cfg.save_dir, f"{cfg.exp_name}.ext_flow_shard_index.pt")
            val_shard_index = build_flow_shard_index(val_shard_dir, val_shard_cache)
            print(f"[FlowBackend][ExtVal] SHARD dir={val_shard_dir} index_items={len(val_shard_index)}")
        else:
            print(f"[FlowBackend][ExtVal] FS dir/npy flow_root={cfg.val_flow_root}")

        ext_val_samples, _same_map, miss_flow_ext, unseen_cnt = build_samples_from_roots(
            cfg.val_video_root, cfg.val_flow_root, cfg,
            speaker2id=speaker2id,
            max_samples=-1,
            unseen_speaker_id=cfg.unseen_speaker_id,
            shard_index=val_shard_index
        )

    pack = {
        "speaker2id": speaker2id,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "ext_val_samples": ext_val_samples,
        "info": {
            "miss_flow_train": int(miss_flow_tr),
            "miss_flow_ext": int(miss_flow_ext),
            "ext_unseen_speakers": int(unseen_cnt),
            "train_flow_backend": "shard" if train_shard_dir is not None else "fs",
            "ext_flow_backend": "shard" if _detect_shard_dir(cfg.val_flow_root) is not None else "fs",
        }
    }
    torch.save(pack, cache_path)
    print(f"[IndexCache] saved: {cache_path}")
    print(f"[IndexCache] train={len(train_samples)} val={len(val_samples)} ext={len(ext_val_samples)} info={pack['info']}")
    return speaker2id, train_samples, val_samples, ext_val_samples


# =========================
# 5) I/O helpers (video/flow) + caches
# =========================
@lru_cache(maxsize=50000)
def count_video_frames_cv2(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n

@lru_cache(maxsize=50000)
def count_video_frames_av(video_path: str) -> int:
    if (not _HAS_AV) or (av is None):
        return 0
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        n = int(getattr(stream, "frames", 0) or 0)
        container.close()
        return n
    except Exception:
        return 0


def _read_video_frames_by_indices_cv2(video_path: str, indices: np.ndarray,
                                     fallback_hw: Tuple[int, int] = (112, 112)) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        H, W = fallback_hw
        z = np.zeros((H, W, 3), dtype=np.uint8)
        return [z.copy() for _ in range(len(indices))]

    fallback = None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, fr = cap.read()
    if ret:
        fallback = fr
    else:
        H, W = fallback_hw
        fallback = np.zeros((H, W, 3), dtype=np.uint8)

    out = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, fr = cap.read()
        if not ret or fr is None:
            fr = fallback
        else:
            fallback = fr
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        out.append(fr.astype(np.uint8))

    cap.release()
    return out


def _read_video_frames_by_fraction_av(video_path: str,
                                      fractions: np.ndarray,
                                      fallback_hw: Tuple[int, int] = (112, 112),
                                      threads: int = 2) -> List[np.ndarray]:
    """
    FAST for sparse sampling:
    - Use container/stream duration and seek by timestamp.
    - Decode a few frames around each seek point.
    Returns RGB uint8 frames.
    """
    H, W = fallback_hw
    z = np.zeros((H, W, 3), dtype=np.uint8)

    if (not _HAS_AV) or (av is None):
        # fallback to cv2 (needs indices; approximate by linspace)
        n = count_video_frames_cv2(video_path)
        idx = sample_indices(max(n, 1), len(fractions), mode="linspace", avoid_ends=True)
        return _read_video_frames_by_indices_cv2(video_path, idx, fallback_hw=fallback_hw)

    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        if threads is not None and int(threads) > 0:
            stream.thread_type = "AUTO"
            stream.thread_count = int(threads)

        # duration in seconds
        dur_sec = None
        try:
            if container.duration is not None and container.duration > 0:
                dur_sec = float(container.duration) / 1e6
        except Exception:
            dur_sec = None
        if (dur_sec is None) or (dur_sec <= 0):
            try:
                if stream.duration is not None and stream.duration > 0:
                    dur_sec = float(stream.duration * stream.time_base)
            except Exception:
                dur_sec = None

        if (dur_sec is None) or (dur_sec <= 0):
            # fallback to cv2
            container.close()
            n = count_video_frames_cv2(video_path)
            idx = sample_indices(max(n, 1), len(fractions), mode="linspace", avoid_ends=True)
            return _read_video_frames_by_indices_cv2(video_path, idx, fallback_hw=fallback_hw)

        tb = float(stream.time_base) if stream.time_base is not None else (1.0 / 90000.0)
        out: List[np.ndarray] = []
        fallback = None

        for frac in fractions:
            frac = float(frac)
            frac = min(max(frac, 0.0), 1.0)
            target_sec = frac * dur_sec
            pts = int(target_sec / max(tb, 1e-12))

            try:
                container.seek(pts, stream=stream, any_frame=False, backward=True)
            except Exception:
                # some formats require any_frame=True
                try:
                    container.seek(pts, stream=stream, any_frame=True, backward=True)
                except Exception:
                    # fallback: decode from start (worst case)
                    container.seek(0, any_frame=True)

            chosen = None
            best = None
            steps = 0
            for frame in container.decode(stream):
                steps += 1
                best = frame
                if frame.pts is None:
                    chosen = frame
                    break
                if frame.pts >= pts:
                    chosen = frame
                    break
                if steps >= 12:
                    break
            if chosen is None:
                chosen = best

            if chosen is None:
                if fallback is not None:
                    out.append(fallback)
                else:
                    out.append(z.copy())
            else:
                arr = chosen.to_ndarray(format="rgb24")
                if arr is None:
                    if fallback is not None:
                        out.append(fallback)
                    else:
                        out.append(z.copy())
                else:
                    arr = arr.astype(np.uint8)
                    fallback = arr
                    out.append(arr)

        container.close()
        return out

    except Exception:
        # final fallback
        n = count_video_frames_cv2(video_path)
        idx = sample_indices(max(n, 1), len(fractions), mode="linspace", avoid_ends=True)
        return _read_video_frames_by_indices_cv2(video_path, idx, fallback_hw=fallback_hw)


def read_framefolder_by_indices(folder: str, indices: np.ndarray, img_exts: Tuple[str, ...],
                               fallback_hw: Tuple[int, int] = (112, 112)) -> List[np.ndarray]:
    names = [x for x in os.listdir(folder) if is_image_file(x, img_exts)]
    names.sort(key=natural_key)
    if len(names) == 0:
        H, W = fallback_hw
        z = np.zeros((H, W, 3), dtype=np.uint8)
        return [z.copy() for _ in range(len(indices))]

    fallback = None
    for fn in names[:50]:
        fp = os.path.join(folder, fn)
        try:
            img = Image.open(fp).convert("RGB")
            fallback = np.asarray(img).astype(np.uint8)
            break
        except Exception:
            pass
    if fallback is None:
        H, W = fallback_hw
        fallback = np.zeros((H, W, 3), dtype=np.uint8)

    out = []
    for i in indices:
        i = int(i)
        i = max(0, min(i, len(names) - 1))
        fp = os.path.join(folder, names[i])
        try:
            img = Image.open(fp).convert("RGB")
            arr = np.asarray(img).astype(np.uint8)
            fallback = arr
        except Exception:
            arr = fallback
        out.append(arr)
    return out


# ---- flow dir list cache ----
@lru_cache(maxsize=20000)
def _cached_sorted_images_in_dir(flow_dir: str, img_exts_key: str) -> Tuple[str, ...]:
    exts = tuple(img_exts_key.split("|"))
    names = [x for x in os.listdir(flow_dir) if is_image_file(x, exts)]
    names.sort(key=natural_key)
    return tuple(names)

def read_flow_dir_by_indices(flow_dir: str, indices: np.ndarray, img_exts: Tuple[str, ...]) -> Tuple[List[np.ndarray], int]:
    img_exts_key = "|".join(img_exts)
    names = _cached_sorted_images_in_dir(flow_dir, img_exts_key)
    if len(names) == 0:
        return [], int(indices.size)

    fallback = None
    for fn in names[:80]:
        tmp = cv2.imread(os.path.join(flow_dir, fn), cv2.IMREAD_COLOR)
        if tmp is not None:
            fallback = tmp
            break
    if fallback is None:
        return [], int(indices.size)

    out = []
    miss = 0
    for i in indices:
        i = int(i)
        i = max(0, min(i, len(names) - 1))
        fp = os.path.join(flow_dir, names[i])

        im = cv2.imread(fp, cv2.IMREAD_COLOR)
        if im is None:
            miss += 1
            im = fallback
        else:
            fallback = im

        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        out.append(im.astype(np.uint8))
    return out, miss


def clip_list_to_tensor(frames: List[np.ndarray],
                        out_hw: Tuple[int,int],
                        mean: np.ndarray,
                        std: np.ndarray,
                        out_dtype: torch.dtype) -> torch.Tensor:
    """
    frames: list of RGB uint8 arrays (H,W,3), possibly different sizes.
    returns torch tensor [C,T,H,W], normalized.
    """
    H, W = out_hw
    T = len(frames)
    if T <= 0:
        x = torch.zeros(3, 0, H, W, dtype=out_dtype)
        return x

    clip = np.empty((T, H, W, 3), dtype=np.float32)
    for t, fr in enumerate(frames):
        if fr is None:
            fr = np.zeros((H, W, 3), dtype=np.uint8)
        if fr.ndim != 3 or fr.shape[2] != 3:
            fr = np.zeros((H, W, 3), dtype=np.uint8)
        if fr.shape[0] != H or fr.shape[1] != W:
            # cv2 resize wants (W,H)
            fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_LINEAR)
        x = fr.astype(np.float32) / 255.0
        x = (x - mean) / (std + 1e-6)
        clip[t] = x

    ten = torch.from_numpy(clip).permute(3, 0, 1, 2).contiguous()  # [C,T,H,W]
    if ten.dtype != out_dtype:
        ten = ten.to(out_dtype)
    return ten


# =========================
# 6) Dataset (returns meta for printing)
# =========================
class PersonalityDataset(Dataset):
    """
    y:
      >=0 : known speaker id (train/val)
      -2  : unseen speaker (external val)
      -1  : broken sample (read fail)

    meta (int32): [backend_id, miss_frames, total_frames]
      backend_id: 0=dir, 1=npy, 2=shard, -1=broken
    """
    def __init__(self, samples: List[Tuple[str, FlowRef, int]], cfg: Config, is_train: bool, print_bad_sample: bool = False):
        self.samples = samples
        self.cfg = cfg
        self.is_train = is_train
        self.print_bad_sample = print_bad_sample

        self.static_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1,1,1,3)
        self.static_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1,1,1,3)

        self.flow_mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1,1,1,3)
        self.flow_std  = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1,1,1,3)

        self.out_dtype = torch.float16 if cfg.return_fp16 else torch.float32

    def __len__(self):
        return len(self.samples)

    def _read_static_frames(self, vpath: str) -> List[np.ndarray]:
        H, W = self.cfg.frame_size
        # framefolder
        if os.path.isdir(vpath):
            # train: random, val: linspace
            mode = "rand" if self.is_train else "linspace"
            names = [x for x in os.listdir(vpath) if is_image_file(x, self.cfg.img_exts)]
            n = len(names)
            sidx = sample_indices(max(n, 1), self.cfg.static_frames, mode=mode, avoid_ends=not self.is_train)
            return read_framefolder_by_indices(vpath, sidx, self.cfg.img_exts, fallback_hw=(H, W))

        # video file
        if self.cfg.use_av and _HAS_AV:
            # sample by fractions of duration (avoid decoding to end)
            if self.cfg.static_frames == 1:
                if self.is_train:
                    fracs = np.array([np.random.uniform(0.05, 0.95)], dtype=np.float32)
                else:
                    fracs = np.array([0.5], dtype=np.float32)
            else:
                base = np.linspace(0.15, 0.85, self.cfg.static_frames, dtype=np.float32)
                if self.is_train:
                    jitter = np.random.uniform(-0.05, 0.05, size=base.shape).astype(np.float32)
                    base = np.clip(base + jitter, 0.05, 0.95)
                fracs = base
            return _read_video_frames_by_fraction_av(vpath, fracs, fallback_hw=(H, W), threads=self.cfg.av_threads)

        # cv2 fallback: use indices, avoid picking last frame
        n = count_video_frames_cv2(vpath)
        if n <= 0:
            raise RuntimeError(f"bad video: {vpath}")
        if self.is_train:
            sidx = sample_indices(n, self.cfg.static_frames, mode="rand", avoid_ends=True)
        else:
            sidx = sample_indices(n, self.cfg.static_frames, mode="linspace", avoid_ends=True)
        return _read_video_frames_by_indices_cv2(vpath, sidx, fallback_hw=(H, W))

    def __getitem__(self, idx):
        vpath, fref, sid = self.samples[idx]
        H, W = self.cfg.frame_size

        try:
            # ---- static ----
            static_frames = self._read_static_frames(vpath)
            if len(static_frames) != self.cfg.static_frames:
                # pad/trim to exact length
                if len(static_frames) == 0:
                    static_frames = [np.zeros((H,W,3), dtype=np.uint8) for _ in range(self.cfg.static_frames)]
                elif len(static_frames) < self.cfg.static_frames:
                    static_frames = static_frames + [static_frames[-1].copy() for _ in range(self.cfg.static_frames - len(static_frames))]
                else:
                    static_frames = static_frames[:self.cfg.static_frames]

            # ---- flow ----
            backend_id = -1
            miss = 0
            total = int(self.cfg.flow_frames)

            if isinstance(fref, dict) and fref.get("type", "") == "shard":
                backend_id = 2
                nflow = int(fref.get("n_frames", 0))
                if nflow <= 0:
                    raise RuntimeError(f"bad shard flow ref: {fref}")
                fidx = sample_indices(max(nflow, 1), self.cfg.flow_frames, mode="linspace", avoid_ends=False)
                flows, miss = read_flow_from_shards(fref, fidx, fallback_hw=(H, W))
                if len(flows) != self.cfg.flow_frames:
                    raise RuntimeError(f"shard flow read mismatch len={len(flows)} need={self.cfg.flow_frames}")

            elif isinstance(fref, str) and os.path.isfile(fref) and fref.lower().endswith(".npy"):
                backend_id = 1
                arr = np.load(fref)
                if arr.ndim != 4:
                    raise RuntimeError(f"bad flow npy shape: {arr.shape} {fref}")
                if arr.dtype != np.uint8:
                    # assume stored as float [0,1] or similar; convert to uint8 for unified pipeline
                    arr_f = arr.astype(np.float32)
                    if arr_f.max() <= 1.5:
                        arr_f = arr_f * 255.0
                    arr = np.clip(arr_f, 0, 255).astype(np.uint8)
                # ensure channels==3
                if arr.shape[-1] != self.cfg.flow_channels:
                    if arr.shape[-1] > self.cfg.flow_channels:
                        arr = arr[..., :self.cfg.flow_channels]
                    else:
                        pad_c = self.cfg.flow_channels - arr.shape[-1]
                        arr = np.concatenate([arr] + [arr[..., :1]] * pad_c, axis=-1)
                nflow = arr.shape[0]
                fidx = sample_indices(max(nflow, 1), self.cfg.flow_frames, mode="linspace", avoid_ends=False)
                flows = [arr[i] for i in fidx.tolist()]
                # NaN/Inf check not needed for uint8

            else:
                backend_id = 0
                if not isinstance(fref, str):
                    raise RuntimeError(f"unknown flow ref type: {type(fref)}")
                names = _cached_sorted_images_in_dir(fref, "|".join(self.cfg.img_exts))
                nflow = len(names)
                if nflow <= 0:
                    raise RuntimeError(f"empty flow dir: {fref}")

                fidx = sample_indices(max(nflow, 1), self.cfg.flow_frames, mode="linspace", avoid_ends=False)
                flows, miss = read_flow_dir_by_indices(fref, fidx, self.cfg.img_exts)
                if len(flows) != self.cfg.flow_frames:
                    if len(flows) == 0:
                        raise RuntimeError(f"flow read fail: {fref}")
                    if len(flows) < self.cfg.flow_frames:
                        pad = [flows[-1].copy() for _ in range(self.cfg.flow_frames - len(flows))]
                        flows = flows + pad
                        miss += len(pad)
                    else:
                        flows = flows[:self.cfg.flow_frames]

            static_t = clip_list_to_tensor(static_frames, (H, W), self.static_mean, self.static_std, self.out_dtype)
            flow_t   = clip_list_to_tensor(flows,        (H, W), self.flow_mean,   self.flow_std,   self.out_dtype)
            meta = torch.tensor([backend_id, int(miss), int(total)], dtype=torch.int32)
            return static_t, flow_t, torch.tensor(int(sid), dtype=torch.long), meta

        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            if self.print_bad_sample:
                print(f"[BAD SAMPLE] v={vpath} f={fref} err={repr(e)}")
            static_t = torch.zeros(3, self.cfg.static_frames, H, W, dtype=self.out_dtype)
            flow_t   = torch.zeros(self.cfg.flow_channels, self.cfg.flow_frames, H, W, dtype=self.out_dtype)
            meta = torch.tensor([-1, int(self.cfg.flow_frames), int(self.cfg.flow_frames)], dtype=torch.int32)
            return static_t, flow_t, torch.tensor(self.cfg.broken_sample_id, dtype=torch.long), meta


# =========================
# 7) safe collate
# =========================
def safe_collate_fn(batch):
    static_list, flow_list, y_list, meta_list = zip(*batch)
    static_b = torch.stack([t.contiguous() for t in static_list], dim=0)
    flow_b   = torch.stack([t.contiguous() for t in flow_list], dim=0)
    y_b      = torch.stack(list(y_list), dim=0)
    meta_b   = torch.stack(list(meta_list), dim=0)
    return static_b, flow_b, y_b, meta_b


# =========================
# 7.5) Shard-aware batch sampler (KEY FIX)
# =========================
class ShardGroupedBatchSampler(Sampler[List[int]]):
    """
    Yield batches where samples in a batch come from the same shard tar.
    This avoids:
      - per-sample opening many tar files
      - LRU thrashing (open/close + scanning headers)
      - huge CPU time spent in tarfile
    """
    def __init__(self,
                 samples: List[Tuple[str, FlowRef, int]],
                 batch_size: int,
                 shuffle: bool,
                 drop_last: bool,
                 seed: int = 42):
        self.samples = samples
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        groups: Dict[str, List[int]] = {}
        for i, (_v, fref, _sid) in enumerate(samples):
            if isinstance(fref, dict) and fref.get("type", "") == "shard":
                k = str(fref.get("tar", ""))
            else:
                k = "__fs__"
            groups.setdefault(k, []).append(i)
        self.groups = groups
        self.keys = list(groups.keys())

        # precompute length
        n_batches = 0
        for k, idxs in groups.items():
            if self.drop_last:
                n_batches += (len(idxs) // self.batch_size)
            else:
                n_batches += int(math.ceil(len(idxs) / max(self.batch_size, 1)))
        self._len = int(n_batches)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterable[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        # build batches
        all_batches: List[List[int]] = []
        keys = self.keys.copy()
        if self.shuffle:
            rng.shuffle(keys)

        for k in keys:
            idxs = self.groups[k].copy()
            if self.shuffle:
                rng.shuffle(idxs)
            bs = self.batch_size
            for i in range(0, len(idxs), bs):
                b = idxs[i:i+bs]
                if len(b) < bs and self.drop_last:
                    continue
                all_batches.append(b)

        if self.shuffle:
            rng.shuffle(all_batches)

        for b in all_batches:
            yield b


# =========================
# 8) Model
# =========================
class StaticExtractor(nn.Module):
    def __init__(self, cfg: Config, debug_shapes: bool = False):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # [B,512,1,1]

        if cfg.freeze_resnet_until is not None:
            freeze = True
            for name, module in resnet.named_children():
                if freeze:
                    for p in module.parameters():
                        p.requires_grad = False
                if name == cfg.freeze_resnet_until:
                    freeze = False

        self.proj = nn.Sequential(
            nn.Linear(512, cfg.feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.debug_shapes = debug_shapes
        self._dbg_printed = False

    def forward(self, x):
        B, C, T, H, W = x.shape
        if self.debug_shapes and (not self._dbg_printed):
            print("\n[DEBUG][StaticExtractor] input:", tuple(x.shape), "dtype=", x.dtype)
        x2 = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        feat = self.backbone(x2).view(B * T, 512)
        feat = feat.view(B, T, 512).mean(dim=1)
        out = self.proj(feat)
        if self.debug_shapes and (not self._dbg_printed):
            print("[DEBUG][StaticExtractor] output:", tuple(out.shape))
            self._dbg_printed = True
        return out


class DynamicExtractor(nn.Module):
    def __init__(self, cfg: Config, debug_shapes: bool = False):
        super().__init__()
        D = cfg.feat_dim
        Dv = D // 2
        De = D - Dv

        def GN(c):
            return nn.GroupNorm(num_groups=8, num_channels=c)

        self.visual_net = nn.Sequential(
            nn.Conv3d(cfg.flow_channels, 64, kernel_size=3, padding=1, bias=False),
            GN(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=3, padding=1, bias=False),
            GN(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            nn.Conv3d(128, 256, kernel_size=3, padding=1, bias=False),
            GN(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool3d((4, 4, 4)),
        )
        self.visual_proj = nn.Sequential(
            nn.Linear(256 * 4 * 4 * 4, Dv),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.energy_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.energy_proj = nn.Sequential(
            nn.Linear(64 + 3, De),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.fuse = nn.Sequential(
            nn.Linear(Dv + De, D),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.debug_shapes = debug_shapes
        self._dbg_printed = False

    def forward(self, x):
        x = x.contiguous()
        B, C, T, H, W = x.shape
        if self.debug_shapes and (not self._dbg_printed):
            print("\n[DEBUG][DynamicExtractor] input:", tuple(x.shape), "dtype=", x.dtype, "contig=", x.is_contiguous())

        v0 = self.visual_net(x)
        v = self.visual_proj(v0.flatten(1))

        if T >= 2:
            diff = x[:, :, 1:] - x[:, :, :-1]
            energy = diff.abs().mean(dim=(1, 3, 4))  # [B,T-1]
        else:
            energy = torch.zeros(B, 1, device=x.device, dtype=x.dtype)

        e_mean = energy.mean(dim=1, keepdim=True)
        e_std  = energy.std(dim=1, keepdim=True) + 1e-6
        e_max  = energy.max(dim=1, keepdim=True).values
        stats = torch.cat([e_mean, e_std, e_max], dim=1)    # [B,3]

        e_seq = energy.unsqueeze(1)                         # [B,1,T-1]
        e_feat = self.energy_conv(e_seq).squeeze(-1)        # [B,64]
        e_feat = torch.cat([e_feat, stats], dim=1)          # [B,67]
        e = self.energy_proj(e_feat)

        out = self.fuse(torch.cat([v, e], dim=1))

        if self.debug_shapes and (not self._dbg_printed):
            print("[DEBUG][DynamicExtractor] v/e/out:", tuple(v.shape), tuple(e.shape), tuple(out.shape))
            self._dbg_printed = True
        return out


class PersonalityNet(nn.Module):
    def __init__(self, cfg: Config, num_speakers: int, debug_shapes: bool = False):
        super().__init__()
        self.static = StaticExtractor(cfg, debug_shapes=debug_shapes)
        self.dynamic = DynamicExtractor(cfg, debug_shapes=debug_shapes)
        self.spk_head_s = nn.Linear(cfg.feat_dim, num_speakers)
        self.spk_head_d = nn.Linear(cfg.feat_dim, num_speakers)
        self.spk_head_f = nn.Linear(cfg.feat_dim * 2, num_speakers)

        self.debug_shapes = debug_shapes
        self._dbg_printed = False

    @torch.no_grad()
    def get_embeddings(self, static_x, flow_x):
        return self.static(static_x), self.dynamic(flow_x)

    def forward(self, static_x, flow_x):
        if self.debug_shapes and (not self._dbg_printed):
            print("\n[DEBUG][PersonalityNet] static_x:", tuple(static_x.shape), "flow_x:", tuple(flow_x.shape))

        ps = self.static(static_x)
        pd = self.dynamic(flow_x)
        ls = self.spk_head_s(ps)
        ld = self.spk_head_d(pd)
        lf = self.spk_head_f(torch.cat([ps, pd], dim=1))

        if self.debug_shapes and (not self._dbg_printed):
            print("[DEBUG][PersonalityNet] ps/pd:", tuple(ps.shape), tuple(pd.shape))
            print("[DEBUG][PersonalityNet] ls/ld/lf:", tuple(ls.shape), tuple(ld.shape), tuple(lf.shape))
            self._dbg_printed = True
        return ps, pd, ls, ld, lf


# =========================
# 9) Losses
# =========================
def indep_correlation_loss(ps: torch.Tensor, pd: torch.Tensor) -> torch.Tensor:
    B, D = ps.shape
    ps = (ps - ps.mean(0)) / (ps.std(0) + 1e-6)
    pd = (pd - pd.mean(0)) / (pd.std(0) + 1e-6)
    c = (ps.T @ pd) / max(B, 1)
    return (c ** 2).mean()

def mi_suppression_margin(ps: torch.Tensor, pd: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """
    NON-NEGATIVE MI suppression (margin form):
      want CE >= log(B) (no better than random matching)
      loss = relu(log(B) - CE) >= 0
    """
    B = ps.size(0)
    if B <= 1:
        return ps.new_tensor(0.0)

    psn = F.normalize(ps, dim=1)
    pdn = F.normalize(pd, dim=1)
    logits = (psn @ pdn.t()) / max(tau, 1e-6)  # [B,B]
    target = torch.arange(B, device=ps.device, dtype=torch.long)
    ce = F.cross_entropy(logits, target, reduction="mean")
    logB = math_log(B, device=ps.device)
    return F.relu(logB - ce)

def correct_count(logits: torch.Tensor, y: torch.Tensor) -> int:
    return int((logits.argmax(dim=1) == y).sum().item())

def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits, dim=1)
    return -(p * (p.clamp_min(1e-9)).log()).sum(dim=1)


class UncertaintyWeighter(nn.Module):
    """
    Kendall et al. uncertainty weighting for multiple losses:
      sum_i exp(-s_i) * L_i + s_i
    """
    def __init__(self, names: List[str]):
        super().__init__()
        self.names = list(names)
        self.log_vars = nn.ParameterDict({n: nn.Parameter(torch.zeros(())) for n in self.names})

    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        any_L = next(iter(losses.values()))
        total = any_L.new_tensor(0.0)
        info = {}
        for n in self.names:
            L = losses.get(n, None)
            if L is None:
                continue
            s = self.log_vars[n]
            w = torch.exp(-s)
            total = total + w * L + s
            info[f"w_{n}"] = float(w.detach().cpu().item())
            info[f"s_{n}"] = float(s.detach().cpu().item())
            info[f"L_{n}"] = float(L.detach().cpu().item())
        return total, info

    @torch.no_grad()
    def export_logvars(self) -> Dict[str, float]:
        return {k: float(v.detach().cpu().item()) for k, v in self.log_vars.items()}


# =========================
# 10) batch selectors (optimized)
# =========================
def select_train_batch(static_x, flow_x, y, meta, device):
    # fast path: no broken samples
    if torch.all(y >= 0):
        return static_x.to(device, non_blocking=True).contiguous(), \
               flow_x.to(device, non_blocking=True).contiguous(), \
               y.to(device, non_blocking=True).contiguous(), \
               meta  # keep on CPU for stats

    idx = torch.nonzero(y >= 0, as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return None
    static_x = static_x.index_select(0, idx).to(device, non_blocking=True).contiguous()
    flow_x   = flow_x.index_select(0, idx).to(device, non_blocking=True).contiguous()
    y        = y.index_select(0, idx).to(device, non_blocking=True).contiguous()
    meta     = meta.index_select(0, idx).contiguous()  # keep on CPU for stats
    return static_x, flow_x, y, meta

def select_nonbroken_batch(static_x, flow_x, y, meta, device):
    if torch.all(y != -1):
        return static_x.to(device, non_blocking=True).contiguous(), \
               flow_x.to(device, non_blocking=True).contiguous(), \
               y.to(device, non_blocking=True).contiguous(), \
               meta
    idx = torch.nonzero(y != -1, as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return None
    static_x = static_x.index_select(0, idx).to(device, non_blocking=True).contiguous()
    flow_x   = flow_x.index_select(0, idx).to(device, non_blocking=True).contiguous()
    y        = y.index_select(0, idx).to(device, non_blocking=True).contiguous()
    meta     = meta.index_select(0, idx).contiguous()
    return static_x, flow_x, y, meta


# =========================
# 11) Train / Val stats printing helpers
# =========================
def data_check_stats(y: torch.Tensor) -> Tuple[int, int]:
    total = int(y.numel())
    bad = int((y == -1).sum().item())
    return total, bad

def _accum_flow_meta(flow_meta: torch.Tensor,
                     backend_stat: Dict[int, Dict[str, int]]):
    """
    flow_meta: [B,3] int32: [backend_id, miss_frames, total_frames]
    backend_id: 0 dir, 1 npy, 2 shard
    """
    if flow_meta is None or flow_meta.numel() == 0:
        return
    b = flow_meta[:, 0].to(torch.int64)
    miss = flow_meta[:, 1].to(torch.int64)
    tot = flow_meta[:, 2].to(torch.int64)
    for bid in (0, 1, 2):
        m = (b == bid)
        if int(m.sum().item()) == 0:
            continue
        backend_stat.setdefault(bid, {"n": 0, "miss": 0, "tot": 0})
        backend_stat[bid]["n"] += int(m.sum().item())
        backend_stat[bid]["miss"] += int(miss[m].sum().item())
        backend_stat[bid]["tot"] += int(tot[m].sum().item())

def _format_backend_stat(backend_stat: Dict[int, Dict[str, int]]) -> str:
    name = {0: "dir", 1: "npy", 2: "shard"}
    parts = []
    for bid in (2, 1, 0):
        s = backend_stat.get(bid, None)
        if s is None:
            continue
        miss = s["miss"]
        tot = max(s["tot"], 1)
        ratio = miss / tot
        parts.append(f"{name[bid]} miss={miss}/{tot}({ratio:.2%}) n={s['n']}")
    return " | ".join(parts) if parts else "n/a"


def train_one_epoch(model, loader, opt, scaler, cfg: Config, device, epoch_idx: int, loss_weighter: UncertaintyWeighter):
    model.train()
    ce = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    loss_sum = 0.0
    corr_sum = 0
    n_sum = 0

    total_batches = 0
    skipped_batches = 0
    total_samples = 0
    bad_samples = 0
    nan_batches = 0

    backend_stat: Dict[int, Dict[str, int]] = {}

    mi_strength = linear_ramp(epoch_idx, cfg.mi_warmup_epochs, cfg.mi_max_strength)

    t_last = time.time()
    t_data_wait_sum = 0.0
    t_step_sum = 0.0
    log_every = max(int(args.log_every), 1)

    comp_sum = {"cls_s": 0.0, "cls_d": 0.0, "cls_f": 0.0, "indep": 0.0, "mi": 0.0}
    comp_n = 0

    pbar = tqdm(loader, desc=f"Train(E{epoch_idx+1})", ncols=140)
    for it, (static_x, flow_x, y, meta) in enumerate(pbar, start=1):
        t_now = time.time()
        t_data_wait = t_now - t_last
        t_data_wait_sum += t_data_wait
        t_step0 = t_now

        total_batches += 1
        t, b = data_check_stats(y)
        total_samples += t
        bad_samples += b

        pack = select_train_batch(static_x, flow_x, y, meta, device)
        if pack is None:
            skipped_batches += 1
            t_last = time.time()
            continue
        static_x, flow_x, y2, meta2 = pack

        _accum_flow_meta(meta2, backend_stat)

        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=cfg.use_amp and device.type == "cuda"):
            ps, pd, ls, ld, lf = model(static_x, flow_x)

            Ls = ce(ls, y2)
            Ld = ce(ld, y2)
            Lf = ce(lf, y2)

            Lind = indep_correlation_loss(ps, pd) * cfg.lambda_indep
            Lmi = mi_suppression_margin(ps, pd, tau=cfg.mi_tau) * (cfg.lambda_mi * mi_strength)

            losses = {"cls_s": Ls, "cls_d": Ld, "cls_f": Lf, "indep": Lind, "mi": Lmi}
            loss, winfo = loss_weighter(losses)

        if not torch.isfinite(loss).all():
            nan_batches += 1
            print(f"[NaN/Inf][Train] epoch={epoch_idx+1} iter={it} loss={float(loss.detach().cpu().item())}")
            skipped_batches += 1
            try:
                scaler.update()
            except Exception:
                pass
            t_last = time.time()
            continue

        scaler.scale(loss).backward()
        if cfg.grad_clip and cfg.grad_clip > 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        scaler.step(opt)
        scaler.update()

        bs = int(y2.size(0))
        loss_sum += float(loss.item()) * bs
        corr_sum += correct_count(lf.detach(), y2)
        n_sum += bs

        comp_sum["cls_s"] += float(Ls.detach().cpu().item()) * bs
        comp_sum["cls_d"] += float(Ld.detach().cpu().item()) * bs
        comp_sum["cls_f"] += float(Lf.detach().cpu().item()) * bs
        comp_sum["indep"] += float(Lind.detach().cpu().item()) * bs
        comp_sum["mi"] += float(Lmi.detach().cpu().item()) * bs
        comp_n += bs

        t_step = time.time() - t_step0
        t_step_sum += t_step
        t_last = time.time()

        if it % log_every == 0:
            avg_wait = t_data_wait_sum / max(it, 1)
            avg_step = t_step_sum / max(it, 1)
            print(f"[Perf][Train] it={it} avg_data_wait={avg_wait:.3f}s avg_step={avg_step:.3f}s "
                  f"(avg_data_wait >> avg_step => GPU starved by data)")
            print(f"[FlowRead][Train][SoFar] { _format_backend_stat(backend_stat) }")
            print(f"[LossWeighter][SoFar] log_vars={loss_weighter.export_logvars()}")

        pbar.set_postfix({
            "loss(avg)": f"{loss_sum/max(n_sum,1):.4f}",
            "acc(fusion,avg)": f"{corr_sum/max(n_sum,1):.4f}",
            "bad_ratio": f"{bad_samples/max(total_samples,1):.2%}",
            "mi_ramp": f"{mi_strength:.2f}",
            "w_cls_f": f"{winfo.get('w_cls_f', 0.0):.2f}",
            "w_indep": f"{winfo.get('w_indep', 0.0):.2f}",
            "w_mi": f"{winfo.get('w_mi', 0.0):.2f}",
        })

    print(
        f"[Train][DataCheck] "
        f"batches={total_batches} | skipped_batches={skipped_batches} | nan_batches={nan_batches} | "
        f"samples={total_samples} | bad_samples={bad_samples} | "
        f"bad_ratio={bad_samples/max(total_samples,1):.2%}"
    )
    print(f"[Train][FlowRead] { _format_backend_stat(backend_stat) }")
    print(f"[Train][LossWeighter] log_vars={loss_weighter.export_logvars()}")

    if comp_n > 0:
        comp_avg = {k: v / comp_n for k, v in comp_sum.items()}
        print(f"[Train][LossComponents(avg)] {comp_avg}")

    loss_avg = loss_sum / max(n_sum, 1)
    acc_avg  = corr_sum / max(n_sum, 1)
    return loss_avg, acc_avg


@torch.no_grad()
def evaluate_in_domain(model, loader, cfg: Config, device, epoch_idx: int, loss_weighter: UncertaintyWeighter):
    model.eval()
    ce = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    loss_sum = 0.0
    corr_sum = 0
    n_sum = 0

    total_batches = 0
    skipped_batches = 0
    total_samples = 0
    bad_samples = 0
    nan_batches = 0

    backend_stat: Dict[int, Dict[str, int]] = {}

    mi_strength = linear_ramp(epoch_idx, cfg.mi_warmup_epochs, cfg.mi_max_strength)

    comp_sum = {"cls_s": 0.0, "cls_d": 0.0, "cls_f": 0.0, "indep": 0.0, "mi": 0.0}
    comp_n = 0

    pbar = tqdm(loader, desc=f"ValIn(E{epoch_idx+1})", ncols=140)
    for static_x, flow_x, y, meta in pbar:
        total_batches += 1
        t, b = data_check_stats(y)
        total_samples += t
        bad_samples += b
        # 在此处检查输入数据是否含有 NaN
        if torch.isnan(static_x).any() or torch.isnan(flow_x).any():
            print(f"NaN detected in input data at batch {total_batches}")
            break
        pack = select_train_batch(static_x, flow_x, y, meta, device)
        if pack is None:
            skipped_batches += 1
            continue
        static_x, flow_x, y2, meta2 = pack

        _accum_flow_meta(meta2, backend_stat)

        with torch.cuda.amp.autocast(enabled=cfg.use_amp and device.type == "cuda"):
            ps, pd, ls, ld, lf = model(static_x, flow_x)

            Ls = ce(ls, y2)
            Ld = ce(ld, y2)
            Lf = ce(lf, y2)
            Lind = indep_correlation_loss(ps, pd) * cfg.lambda_indep
            Lmi = mi_suppression_margin(ps, pd, tau=cfg.mi_tau) * (cfg.lambda_mi * mi_strength)

            losses = {"cls_s": Ls, "cls_d": Ld, "cls_f": Lf, "indep": Lind, "mi": Lmi}
            loss, _ = loss_weighter(losses)

        # if not torch.isfinite(loss).all():
        #     nan_batches += 1
        #     print(f"[NaN/Inf][ValIn] epoch={epoch_idx+1} loss={float(loss.detach().cpu().item())}")
        #     skipped_batches += 1
        #     continue
        # 检查损失是否为 NaN
        if torch.isnan(loss).any():
            print(f"[NaN/Inf][ValIn] epoch={epoch_idx+1} loss={loss.item()}")
            print(f"Loss components: {losses}")
            nan_batches += 1
            skipped_batches += 1
            continue
        bs = int(y2.size(0))
        loss_sum += float(loss.item()) * bs
        corr_sum += correct_count(lf, y2)
        n_sum += bs

        comp_sum["cls_s"] += float(Ls.detach().cpu().item()) * bs
        comp_sum["cls_d"] += float(Ld.detach().cpu().item()) * bs
        comp_sum["cls_f"] += float(Lf.detach().cpu().item()) * bs
        comp_sum["indep"] += float(Lind.detach().cpu().item()) * bs
        comp_sum["mi"] += float(Lmi.detach().cpu().item()) * bs
        comp_n += bs

        pbar.set_postfix({
            "loss(avg)": f"{loss_sum/max(n_sum,1):.4f}",
            "acc(fusion,avg)": f"{corr_sum/max(n_sum,1):.4f}",
            "bad_ratio": f"{bad_samples/max(total_samples,1):.2%}",
        })

    print(
        f"[ValIn][DataCheck] "
        f"batches={total_batches} | skipped_batches={skipped_batches} | nan_batches={nan_batches} | "
        f"samples={total_samples} | bad_samples={bad_samples} | "
        f"bad_ratio={bad_samples/max(total_samples,1):.2%}"
    )
    print(f"[ValIn][FlowRead] { _format_backend_stat(backend_stat) }")

    if comp_n > 0:
        comp_avg = {k: v / comp_n for k, v in comp_sum.items()}
        print(f"[ValIn][LossComponents(avg)] {comp_avg}")

    loss_avg = loss_sum / max(n_sum, 1)
    acc_avg  = corr_sum / max(n_sum, 1)
    return loss_avg, acc_avg


@torch.no_grad()
def evaluate_external_unseen(model, loader, cfg: Config, device, epoch_idx: int):
    model.eval()

    indep_sum = 0.0
    ent_sum = 0.0
    conf_sum = 0.0
    ps_norm_sum = 0.0
    pd_norm_sum = 0.0
    n_sum = 0

    total_batches = 0
    skipped_batches = 0
    total_samples = 0
    bad_samples = 0
    nan_batches = 0

    backend_stat: Dict[int, Dict[str, int]] = {}

    pbar = tqdm(loader, desc=f"ValExt(E{epoch_idx+1})", ncols=140)
    for static_x, flow_x, y, meta in pbar:
        total_batches += 1
        t, b = data_check_stats(y)
        total_samples += t
        bad_samples += b
        # 在此处检查输入数据是否含有 NaN
        if torch.isnan(static_x).any() or torch.isnan(flow_x).any():
            print(f"NaN detected in input data at batch {total_batches}")
            break
        pack = select_nonbroken_batch(static_x, flow_x, y, meta, device)
        if pack is None:
            skipped_batches += 1
            continue
        static_x, flow_x, _y, meta2 = pack

        _accum_flow_meta(meta2, backend_stat)

        with torch.cuda.amp.autocast(enabled=cfg.use_amp and device.type == "cuda"):
            ps, pd, ls, ld, lf = model(static_x, flow_x)
            indep = indep_correlation_loss(ps, pd)
            ent = softmax_entropy(lf)
            conf = torch.softmax(lf, dim=1).max(dim=1).values

        if (not torch.isfinite(indep).all()) or (not torch.isfinite(ent).all()) or (not torch.isfinite(conf).all()):
            nan_batches += 1
            print(f"[NaN/Inf][ValExt] epoch={epoch_idx+1}")
            skipped_batches += 1
            continue

        bs = int(ps.size(0))
        indep_sum += float(indep.item()) * bs
        ent_sum += float(ent.mean().item()) * bs
        conf_sum += float(conf.mean().item()) * bs
        ps_norm_sum += float(ps.norm(dim=1).mean().item()) * bs
        pd_norm_sum += float(pd.norm(dim=1).mean().item()) * bs
        n_sum += bs

        pbar.set_postfix({
            "indep(avg)": f"{indep_sum/max(n_sum,1):.6f}",
            "entropy(avg)": f"{ent_sum/max(n_sum,1):.4f}",
            "conf(avg)": f"{conf_sum/max(n_sum,1):.4f}",
            "bad_ratio": f"{bad_samples/max(total_samples,1):.2%}",
        })

    print(
        f"[ValExt][DataCheck] "
        f"batches={total_batches} | skipped_batches={skipped_batches} | nan_batches={nan_batches} | "
        f"samples={total_samples} | bad_samples={bad_samples} | "
        f"bad_ratio={bad_samples/max(total_samples,1):.2%}"
    )
    print(f"[ValExt][FlowRead] { _format_backend_stat(backend_stat) }")

    denom = max(n_sum, 1)
    return {
        "indep": indep_sum / denom,
        "entropy": ent_sum / denom,
        "conf": conf_sum / denom,
        "ps_norm": ps_norm_sum / denom,
        "pd_norm": pd_norm_sum / denom,
        "n": int(n_sum),
    }


@torch.no_grad()
def compute_neutral_codes(model, loader, cfg: Config, device):
    model.eval()
    sum_ps = torch.zeros(cfg.feat_dim, device=device)
    sum_pd = torch.zeros(cfg.feat_dim, device=device)
    cnt = 0.0

    net = model.module if isinstance(model, nn.DataParallel) else model

    pbar = tqdm(loader, desc="Compute neutral codes", ncols=120)
    for static_x, flow_x, y, meta in pbar:
        pack = select_train_batch(static_x, flow_x, y, meta, device)
        if pack is None:
            continue
        static_x, flow_x, _y, _meta2 = pack

        ps, pd = net.get_embeddings(static_x, flow_x)
        sum_ps += ps.sum(dim=0)
        sum_pd += pd.sum(dim=0)
        cnt += float(ps.size(0))

    ps0 = (sum_ps / max(cnt, 1.0)).detach().cpu()
    pd0 = (sum_pd / max(cnt, 1.0)).detach().cpu()
    return ps0, pd0


# =========================
# 12) Main (DP)
# =========================
def main():
    cfg = build_config(args)

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # set threads in main proc too
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    if device.type == "cuda":
        if args.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
        if args.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

    seed_all(cfg.seed)

    os.makedirs(cfg.save_dir, exist_ok=True)
    cfg_path = os.path.join(cfg.save_dir, f"{cfg.exp_name}.config.dp.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    print(f"[Config] {cfg_path}")
    print(f"[Device] {device} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')} | torch.cuda.device_count()={torch.cuda.device_count()}")
    print(f"[VideoDecode] use_av={cfg.use_av} has_av={_HAS_AV} av_threads={cfg.av_threads}")
    print(f"[TarCache] max_open={args.tar_cache_max_open} (per worker process)")
    print(f"[DatasetDType] return_fp16={cfg.return_fp16} (use_amp={cfg.use_amp})")
    print(f"[PerfArgs] prefetch_factor={args.prefetch_factor} num_workers={cfg.num_workers} log_every={args.log_every}")

    speaker2id, train_samples, val_samples, ext_val_samples = build_index_with_cache(cfg)
    print(f"[Split] train={len(train_samples)} val(in)={len(val_samples)} speakers(train)={len(speaker2id)}")
    if not cfg.disable_external_val:
        print(f"[ExternalVal] samples={len(ext_val_samples)} root={cfg.val_video_root}")

    # sort val/ext by tar to improve locality (fast eval)
    if _is_shard_sample_list(val_samples):
        val_samples = sorted(val_samples, key=lambda x: (x[1].get("tar","") if isinstance(x[1],dict) else "", x[1].get("key","") if isinstance(x[1],dict) else "", x[0]))
    if _is_shard_sample_list(ext_val_samples):
        ext_val_samples = sorted(ext_val_samples, key=lambda x: (x[1].get("tar","") if isinstance(x[1],dict) else "", x[1].get("key","") if isinstance(x[1],dict) else "", x[0]))

    train_ds = PersonalityDataset(train_samples, cfg, is_train=True,  print_bad_sample=args.print_bad_sample)
    val_ds   = PersonalityDataset(val_samples,   cfg, is_train=False, print_bad_sample=args.print_bad_sample)
    ext_val_ds = PersonalityDataset(ext_val_samples, cfg, is_train=False, print_bad_sample=args.print_bad_sample)

    # ---- train loader: shard-aware batching (KEY) ----
    use_shard_backend = _is_shard_sample_list(train_samples)
    # Default ON for shard backend (can disable with --no_batch_by_shard)
    batch_by_shard = (not args.no_batch_by_shard) and use_shard_backend
    # Explicit flag still works as a positive override
    if args.batch_by_shard:
        batch_by_shard = True

    if batch_by_shard:
        train_batch_sampler = ShardGroupedBatchSampler(train_samples, batch_size=cfg.batch_size,
                                                      shuffle=True, drop_last=True, seed=cfg.seed)
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_batch_sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=safe_collate_fn,
            worker_init_fn=seed_worker_fn(cfg.seed),
            persistent_workers=(cfg.num_workers > 0),
            prefetch_factor=int(args.prefetch_factor) if cfg.num_workers > 0 else None,
        )
        print("[DataLoader][Train] ShardGroupedBatchSampler ENABLED (batch_by_shard).")
    else:
        train_batch_sampler = None
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=safe_collate_fn,
            worker_init_fn=seed_worker_fn(cfg.seed),
            persistent_workers=(cfg.num_workers > 0),
            prefetch_factor=int(args.prefetch_factor) if cfg.num_workers > 0 else None,
        )
        print("[DataLoader][Train] standard shuffle batching.")

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=safe_collate_fn,
        worker_init_fn=seed_worker_fn(cfg.seed),
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=int(args.prefetch_factor) if cfg.num_workers > 0 else None,
    )
    ext_val_loader = DataLoader(
        ext_val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=safe_collate_fn,
        worker_init_fn=seed_worker_fn(cfg.seed),
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=int(args.prefetch_factor) if cfg.num_workers > 0 else None,
    )

    debug_shapes = bool(args.debug_shapes)
    model = PersonalityNet(cfg, num_speakers=len(speaker2id), debug_shapes=debug_shapes).to(device)

    if use_cuda and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print("[DP] Enabled nn.DataParallel on visible GPUs.")

    loss_weighter = UncertaintyWeighter(["cls_s", "cls_d", "cls_f", "indep", "mi"]).to(device)

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(loss_weighter.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp and device.type == "cuda")

    scheduler = None
    if cfg.use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.05)
        print("[LR] CosineAnnealingLR enabled.")

    ckpt_best = os.path.join(cfg.save_dir, f"{cfg.exp_name}.best.pth")
    ckpt_last = os.path.join(cfg.save_dir, f"{cfg.exp_name}.last.pth")
    ckpt_compat = os.path.join(cfg.save_dir, f"{cfg.exp_name}.pth")

    start_epoch = 0
    best_val = -1.0

    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"--resume not found: {args.resume}")
        ckpt = load_checkpoint(args.resume)

        ok, err = smart_load_model_state(model, ckpt["model_state"], strict=True)
        if not ok:
            print("[Resume][Warn] strict load failed, try strict=False. reason:", err)
            ok2, err2 = smart_load_model_state(model, ckpt["model_state"], strict=False)
            if not ok2:
                raise RuntimeError(f"[Resume] model_state load failed: {err2}")

        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val = float(ckpt.get("best_val_acc", -1.0))

        if not args.reset_optimizer:
            if ckpt.get("optimizer_state", None) is not None:
                try:
                    opt.load_state_dict(ckpt["optimizer_state"])
                except Exception as e:
                    print("[Resume][Warn] optimizer_state load failed -> restart optimizer:", repr(e))
            if ckpt.get("scaler_state", None) is not None:
                try:
                    scaler.load_state_dict(ckpt["scaler_state"])
                except Exception as e:
                    print("[Resume][Warn] scaler_state load failed -> restart scaler:", repr(e))

        print(f"[Resume] from={args.resume} start_epoch={start_epoch}, best_val_acc={best_val:.4f}")

    elif args.pretrained:
        if not os.path.isfile(args.pretrained):
            raise FileNotFoundError(f"--pretrained not found: {args.pretrained}")
        ckpt = load_checkpoint(args.pretrained)

        ok, err = smart_load_model_state(model, ckpt["model_state"], strict=True)
        if not ok:
            print("[Pretrained][Warn] strict load failed, try strict=False. reason:", err)
            ok2, err2 = smart_load_model_state(model, ckpt["model_state"], strict=False)
            if not ok2:
                raise RuntimeError(f"[Pretrained] model_state load failed: {err2}")

        best_val = float(ckpt.get("best_val_acc", -1.0))
        print(f"[Pretrained] loaded. best_val_acc(in ckpt)={best_val:.4f}")

    for ep in range(start_epoch, cfg.epochs):
        print(f"\n===== Epoch {ep+1}/{cfg.epochs} =====")
        cur_lr = opt.param_groups[0]["lr"]
        print(f"[LR] {cur_lr:.6g}")
        print(f"[LossWeighter][EpochStart] log_vars={loss_weighter.export_logvars()}")

        # update epoch for shard batch sampler
        if train_batch_sampler is not None and hasattr(train_batch_sampler, "set_epoch"):
            train_batch_sampler.set_epoch(ep)

        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, scaler, cfg, device, ep, loss_weighter)
        va_loss, va_acc = evaluate_in_domain(model, val_loader, cfg, device, ep, loss_weighter)

        ext_metrics = None
        if (not cfg.disable_external_val) and (len(ext_val_ds) > 0):
            ext_metrics = evaluate_external_unseen(model, ext_val_loader, cfg, device, ep)

        msg = f"[Epoch {ep+1}] train loss={tr_loss:.4f} acc={tr_acc:.4f} | val(in) loss={va_loss:.4f} acc={va_acc:.4f}"
        if ext_metrics is not None:
            msg += (" | val(ext) "
                    f"indep={ext_metrics['indep']:.6f} "
                    f"entropy={ext_metrics['entropy']:.4f} "
                    f"conf={ext_metrics['conf']:.4f} "
                    f"||ps||={ext_metrics['ps_norm']:.3f} "
                    f"||pd||={ext_metrics['pd_norm']:.3f} "
                    f"n={ext_metrics['n']}")
        print(msg)

        if scheduler is not None:
            scheduler.step()

        save_checkpoint(ckpt_last, cfg, speaker2id, model, opt, scaler, ep, best_val_acc=best_val)
        print(f"[Save] last -> {ckpt_last}")

        if va_acc > best_val:
            best_val = va_acc
            ps0, pd0 = compute_neutral_codes(model, train_loader, cfg, device)
            save_checkpoint(ckpt_best, cfg, speaker2id, model, opt, scaler, ep, best_val_acc=best_val, ps0=ps0, pd0=pd0)
            save_checkpoint(ckpt_compat, cfg, speaker2id, model, opt, scaler, ep, best_val_acc=best_val, ps0=ps0, pd0=pd0)
            print(f"[Save] best -> {ckpt_best} (best_val_acc={best_val:.4f})")
            print(f"[Save] compat -> {ckpt_compat}")
            print(f"[NeutralCodes] ||ps0||={float(ps0.norm().item()):.4f} ||pd0||={float(pd0.norm().item()):.4f}")

    print(f"\nDone. Best in-domain val acc={best_val:.4f}")
    print(f"Best: {ckpt_best}")
    print(f"Last: {ckpt_last}")


if __name__ == "__main__":
    main()
