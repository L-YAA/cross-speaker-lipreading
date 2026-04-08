# feature_ext_dp.py
# -*- coding: utf-8 -*-
"""
Stage1 Personality Feature Training - DataParallel version
(with external val + adaptive loss weighting + MI suppression + data health print)

✅ DP/Single training (CUDA_VISIBLE_DEVICES controls physical GPUs)
✅ in-domain val + external val (unseen speakers)  <-- ext_unseen_speakers uses TRAIN speaker2id
✅ best/last checkpoints + resume (DP/DDP/single compatible "module." prefix handling)
✅ Bad sample statistics printing per epoch (+ optional bad sample path print)
✅ Adaptive loss weighting: Uncertainty weighting (learnable log_vars)
✅ MI suppression (approx): "anti-InfoNCE" pair-matching confusion between ps and pd
✅ Optional cosine LR scheduler

🚀 Speed optimizations (NO EFFECT on model/loss):
- Video decode: prefer PyAV sequential decode (avoid repeated cv2 seek); fallback to cv2 if av not installed.
- Flow dir list cache: LRU cache for os.listdir/sort results to reduce CPU overhead.
"""

import os
import re
import json
import random
import argparse
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

# -------------------------
# 0) parse args & set CUDA_VISIBLE_DEVICES (MUST before import torch)
# -------------------------
def parse_args():
    p = argparse.ArgumentParser("Stage1 Personality Feature Training - DP + ExternalVal + AdaptiveLoss + MI")

    # train paths
    p.add_argument("--video_root", type=str,
                   default="/mnt/netdisk/dataset/lipreading/datasets/CMLR/cmlr/cmlr_video_seg24s/",
                   help="root of original videos (train)")
    p.add_argument("--flow_root", type=str,
                   default="/home/liuyang/Project/flow/flow_sequence/cmlr2/",
                   help="root of flow frames (train)")

    # external val (unseen speakers)
    p.add_argument("--val_video_root", type=str,
                   default="/mnt/netdisk/dataset/lipreading/mpc/preprocess_datasets/CMLR_extra_latest/cmlr_extra/cmlr_extra_video_seg24s/",
                   help="external val video root (unseen speakers)")
    p.add_argument("--val_flow_root", type=str,
                   default="/home/liuyang/Project/flow/flow_sequence/cmlr_extra/",
                   help="external val flow root (unseen speakers)")
    p.add_argument("--disable_external_val", action="store_true", help="do not run external val")

    # debug
    p.add_argument("--debug_shapes", action="store_true", help="print initial shapes once")
    p.add_argument("--print_bad_sample", action="store_true", help="print bad sample path in dataset exception")

    # ---- speed: PyAV controls ----
    p.add_argument("--use_av", action="store_true",
                   help="use PyAV to decode video (faster than cv2 random seek). If av not installed, auto fallback.")
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
    p.add_argument("--exp_name", type=str, default="personality_stage1_dp")

    # index cache (accelerate scanning)
    p.add_argument("--index_cache", type=str, default="",
                   help="path to index cache file (.pt). empty => auto in save_dir")
    p.add_argument("--rebuild_index", action="store_true", help="force rebuild index cache even if exists")

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

    # indep loss weight base (still adaptively weighted, but keep as scaling knob)
    p.add_argument("--lambda_indep", type=float, default=1.0)

    # MI suppression
    p.add_argument("--lambda_mi", type=float, default=0.2, help="scale for MI suppression (anti-InfoNCE)")
    p.add_argument("--mi_tau", type=float, default=0.07, help="temperature for MI suppression")
    p.add_argument("--mi_warmup_epochs", type=int, default=2, help="warmup epochs for MI strength ramp")
    p.add_argument("--mi_max_strength", type=float, default=1.0, help="max ramp multiplier for MI loss")

    # split
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--split_by_speaker", action="store_true")

    # debug scan limit
    p.add_argument("--max_samples", type=int, default=-1, help="limit samples for debug (train scan only)")

    return p.parse_args()


args = parse_args()

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
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from tqdm import tqdm

# optional: PyAV
try:
    import av  # pip install av  (or conda install -c conda-forge av)
    _HAS_AV = True
except Exception:
    av = None
    _HAS_AV = False

from functools import lru_cache


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

    speaker_prefix: str = "s"
    unseen_speaker_id: int = -2
    broken_sample_id: int = -1

    video_exts: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    img_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def build_config(a) -> Config:
    frz = None if str(a.freeze_resnet_until).lower() in ("none", "null", "") else a.freeze_resnet_until
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

def sample_indices(n: int, t: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((t,), dtype=np.int64)
    if n == 1:
        return np.zeros((t,), dtype=np.int64)
    return np.linspace(0, n - 1, t).astype(np.int64)

def cosine_ramp(epoch_idx: int, warmup: int, max_strength: float) -> float:
    if warmup <= 0:
        return max_strength
    if epoch_idx < warmup:
        return max_strength * float(epoch_idx + 1) / float(warmup)
    return max_strength

def math_log(x: int, device: torch.device):
    return torch.log(torch.tensor(float(x), device=device) + 1e-12)


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
    items: List[Tuple[str, str]] = []
    for root, dirs, files in os.walk(speaker_video_dir):
        for fn in files:
            fp = os.path.join(root, fn)
            if is_video_file(fp, cfg.video_exts):
                rel = os.path.relpath(fp, speaker_video_dir)
                rel_no_ext = os.path.splitext(rel)[0]
                items.append((fp, rel_no_ext))

        # optional: frame folders as "video"
        for d in dirs:
            dp = os.path.join(root, d)
            try:
                ims = [x for x in os.listdir(dp) if is_image_file(x, cfg.img_exts)]
                if len(ims) >= 3:
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


def build_samples_from_roots(video_root: str,
                             flow_root: str,
                             cfg: Config,
                             speaker2id: Optional[Dict[str, int]],
                             max_samples: int = -1,
                             unseen_speaker_id: int = -2) -> Tuple[List[Tuple[str, str, int]], Dict[str, int], int, int]:
    """
    returns: samples, speaker2id_used, miss_flow_count, unseen_speaker_count

    IMPORTANT:
      - If speaker2id is provided (train mapping), speakers not in it will be labeled as unseen_speaker_id.
        This is what we want for EXTERNAL VAL.
    """
    speakers = list_speakers(video_root, cfg.speaker_prefix)

    if speaker2id is None:
        speaker2id = {s: i for i, s in enumerate(speakers)}

    samples: List[Tuple[str, str, int]] = []
    miss_flow = 0
    unseen_set = set()

    for spk in speakers:
        v_dir = os.path.join(video_root, spk)
        f_dir = os.path.join(flow_root, spk)
        if not os.path.isdir(v_dir):
            continue

        if spk in speaker2id:
            sid = speaker2id[spk]
        else:
            sid = unseen_speaker_id
            unseen_set.add(spk)

        video_items = discover_video_items_with_relkey(v_dir, cfg)
        for vpath, rel_key in video_items:
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


def split_train_val(samples: List[Tuple[str, str, int]], cfg: Config):
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

    # rebuild train
    all_train, speaker2id, miss_flow_tr, _ = build_samples_from_roots(
        cfg.video_root, cfg.flow_root, cfg,
        speaker2id=None,
        max_samples=args.max_samples,
        unseen_speaker_id=cfg.unseen_speaker_id
    )
    train_samples, val_samples = split_train_val(all_train, cfg)

    # rebuild external val (must use TRAIN speaker2id)
    ext_val_samples = []
    miss_flow_ext = 0
    unseen_cnt = 0
    if (not cfg.disable_external_val) and os.path.isdir(cfg.val_video_root):
        ext_val_samples, _same_map, miss_flow_ext, unseen_cnt = build_samples_from_roots(
            cfg.val_video_root, cfg.val_flow_root, cfg,
            speaker2id=speaker2id,
            max_samples=-1,
            unseen_speaker_id=cfg.unseen_speaker_id
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
        }
    }
    torch.save(pack, cache_path)
    print(f"[IndexCache] saved: {cache_path}")
    print(f"[IndexCache] train={len(train_samples)} val={len(val_samples)} ext={len(ext_val_samples)} info={pack['info']}")
    return speaker2id, train_samples, val_samples, ext_val_samples


# =========================
# 5) I/O helpers (video/flow) + caches
# =========================
def count_video_frames_cv2(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n

def _read_video_frames_by_indices_cv2(video_path: str, indices: np.ndarray,
                                     fallback_hw: Tuple[int, int] = (112, 112)) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        H, W = fallback_hw
        z = np.zeros((H, W, 3), dtype=np.float32)
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
        if not ret:
            fr = fallback
        else:
            fallback = fr
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        fr = fr.astype(np.float32) / 255.0
        out.append(fr)

    cap.release()
    return out

def _read_video_frames_by_indices_av(video_path: str, indices: np.ndarray,
                                    fallback_hw: Tuple[int, int] = (112, 112),
                                    threads: int = 2) -> List[np.ndarray]:
    """
    PyAV sequential decode until max(indices), collect frames at requested indices.
    This avoids repeated seek and is usually faster on network disk.

    Note: Uses cv2 only for frame count (sampling indices), but decoding uses FFmpeg via PyAV.
    """
    H, W = fallback_hw
    z = np.zeros((H, W, 3), dtype=np.float32)

    if (not _HAS_AV) or (av is None):
        return _read_video_frames_by_indices_cv2(video_path, indices, fallback_hw=fallback_hw)

    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        if threads is not None and int(threads) > 0:
            stream.thread_type = "AUTO"
            stream.thread_count = int(threads)

        # indices to collect
        idx_list = [int(x) for x in indices.tolist()]
        need = set(idx_list)
        max_idx = max(idx_list) if len(idx_list) > 0 else 0

        collected: Dict[int, np.ndarray] = {}
        cur = 0
        fallback_rgb = None

        for frame in container.decode(stream):
            if cur > max_idx:
                break
            if cur in need:
                arr = frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0
                collected[cur] = arr
                fallback_rgb = arr
            elif fallback_rgb is None:
                arr = frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0
                fallback_rgb = arr
            cur += 1

        container.close()

        # fill outputs in the original order
        out = []
        for i in idx_list:
            if i in collected:
                out.append(collected[i])
            else:
                if fallback_rgb is not None:
                    out.append(fallback_rgb)
                else:
                    out.append(z.copy())
        return out

    except Exception:
        # fallback to cv2 if PyAV fails for some codec/file
        return _read_video_frames_by_indices_cv2(video_path, indices, fallback_hw=fallback_hw)

def read_video_frames_by_indices(video_path: str, indices: np.ndarray,
                                fallback_hw: Tuple[int, int],
                                use_av: bool,
                                av_threads: int) -> List[np.ndarray]:
    if use_av:
        return _read_video_frames_by_indices_av(video_path, indices, fallback_hw=fallback_hw, threads=av_threads)
    return _read_video_frames_by_indices_cv2(video_path, indices, fallback_hw=fallback_hw)

def read_framefolder_by_indices(folder: str, indices: np.ndarray, img_exts: Tuple[str, ...],
                               fallback_hw: Tuple[int, int] = (112, 112)) -> List[np.ndarray]:
    names = [x for x in os.listdir(folder) if is_image_file(x, img_exts)]
    names.sort(key=natural_key)
    if len(names) == 0:
        H, W = fallback_hw
        z = np.zeros((H, W, 3), dtype=np.float32)
        return [z.copy() for _ in range(len(indices))]

    fallback = None
    for fn in names[:50]:
        fp = os.path.join(folder, fn)
        try:
            img = Image.open(fp).convert("RGB")
            fallback = np.asarray(img)
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
            arr = np.asarray(img)
            fallback = arr
        except Exception:
            arr = fallback
        out.append(arr.astype(np.float32) / 255.0)
    return out

# ---- flow dir list cache (critical for CPU) ----
@lru_cache(maxsize=20000)
def _cached_sorted_images_in_dir(flow_dir: str, img_exts_key: str) -> Tuple[str, ...]:
    # img_exts_key is just to avoid mixing caches if you ever change cfg.img_exts
    names = [x for x in os.listdir(flow_dir) if is_image_file(x, tuple(img_exts_key.split("|")))]
    names.sort(key=natural_key)
    return tuple(names)

def read_flow_dir_by_indices(flow_dir: str, indices: np.ndarray, img_exts: Tuple[str, ...]) -> List[np.ndarray]:
    img_exts_key = "|".join(img_exts)
    names = _cached_sorted_images_in_dir(flow_dir, img_exts_key)
    if len(names) == 0:
        return []

    fallback = None
    # sample a few to find a readable fallback
    for fn in names[:80]:
        tmp = cv2.imread(os.path.join(flow_dir, fn), cv2.IMREAD_COLOR)
        if tmp is not None:
            fallback = tmp
            break
    if fallback is None:
        return []

    out = []
    for i in indices:
        i = int(i)
        i = max(0, min(i, len(names) - 1))
        fp = os.path.join(flow_dir, names[i])

        im = cv2.imread(fp, cv2.IMREAD_COLOR)
        if im is None:
            im = np.zeros_like(fallback)
        else:
            fallback = im

        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.astype(np.float32) / 255.0
        out.append(im)
    return out

def resize_and_normalize_clip(clip: np.ndarray, out_h: int, out_w: int,
                              mean: Tuple[float, ...], std: Tuple[float, ...]) -> torch.Tensor:
    T, H, W, C = clip.shape
    x = torch.from_numpy(clip).permute(0, 3, 1, 2)  # [T,C,H,W]
    x = F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False)
    mean_t = torch.tensor(mean, dtype=x.dtype).view(1, C, 1, 1)
    std_t  = torch.tensor(std,  dtype=x.dtype).view(1, C, 1, 1)
    x = (x - mean_t) / (std_t + 1e-6)
    x = x.permute(1, 0, 2, 3).contiguous().clone()  # [C,T,H,W]
    return x


# =========================
# 6) Dataset
# =========================
class PersonalityDataset(Dataset):
    """
    y:
      >=0 : known speaker id (train/val)
      -2  : unseen speaker (external val)
      -1  : broken sample (read fail)
    """
    def __init__(self, samples: List[Tuple[str, str, int]], cfg: Config, is_train: bool, print_bad_sample: bool = False):
        self.samples = samples
        self.cfg = cfg
        self.is_train = is_train
        self.print_bad_sample = print_bad_sample

        self.static_mean = (0.485, 0.456, 0.406)
        self.static_std  = (0.229, 0.224, 0.225)

        self.flow_mean = (0.5, 0.5, 0.5)
        self.flow_std  = (0.5, 0.5, 0.5)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vpath, fpath, sid = self.samples[idx]
        H, W = self.cfg.frame_size

        try:
            # ---- static ----
            if os.path.isdir(vpath):
                names = [x for x in os.listdir(vpath) if is_image_file(x, self.cfg.img_exts)]
                n = len(names)
                sidx = sample_indices(max(n, 1), self.cfg.static_frames)
                static_frames = read_framefolder_by_indices(vpath, sidx, self.cfg.img_exts, fallback_hw=(H, W))
            else:
                n = count_video_frames_cv2(vpath)  # only for length (sampling indices)
                if n <= 0:
                    raise RuntimeError(f"bad video: {vpath}")
                if self.is_train and self.cfg.static_frames == 1:
                    sidx = np.array([random.randint(0, n - 1)], dtype=np.int64)
                else:
                    sidx = sample_indices(n, self.cfg.static_frames)

                static_frames = read_video_frames_by_indices(
                    vpath, sidx, fallback_hw=(H, W),
                    use_av=self.cfg.use_av,
                    av_threads=self.cfg.av_threads,
                )

            static_clip = np.stack(static_frames, axis=0)  # [T,H,W,3]

            # ---- flow ----
            if os.path.isfile(fpath) and fpath.lower().endswith(".npy"):
                arr = np.load(fpath).astype(np.float32)
                if arr.ndim != 4:
                    raise RuntimeError(f"bad flow npy shape: {arr.shape} {fpath}")
                if arr.shape[-1] != self.cfg.flow_channels:
                    if arr.shape[-1] > self.cfg.flow_channels:
                        arr = arr[..., :self.cfg.flow_channels]
                    else:
                        pad_c = self.cfg.flow_channels - arr.shape[-1]
                        arr = np.concatenate([arr] + [arr[..., :1]] * pad_c, axis=-1)
                nflow = arr.shape[0]
                fidx = sample_indices(max(nflow, 1), self.cfg.flow_frames)
                flow_clip = arr[fidx]
            else:
                # NOTE: we do not listdir here repeatedly; read_flow_dir_by_indices uses cached list.
                flows = read_flow_dir_by_indices(fpath, sample_indices(1, 1), self.cfg.img_exts)  # quick check via cache
                if len(flows) == 0:
                    # now do actual sampling but will still use cached list internally
                    pass

                # determine nflow by cached list length
                names = _cached_sorted_images_in_dir(fpath, "|".join(self.cfg.img_exts))
                nflow = len(names)
                if nflow <= 0:
                    raise RuntimeError(f"empty flow dir: {fpath}")

                fidx = sample_indices(max(nflow, 1), self.cfg.flow_frames)
                flows = read_flow_dir_by_indices(fpath, fidx, self.cfg.img_exts)

                if len(flows) != self.cfg.flow_frames:
                    if len(flows) == 0:
                        raise RuntimeError(f"flow read fail: {fpath}")
                    if len(flows) < self.cfg.flow_frames:
                        pad = [np.zeros_like(flows[-1]) for _ in range(self.cfg.flow_frames - len(flows))]
                        flows = flows + pad
                    else:
                        flows = flows[:self.cfg.flow_frames]
                flow_clip = np.stack(flows, axis=0)  # [T,H,W,3]

            static_t = resize_and_normalize_clip(static_clip, H, W, self.static_mean, self.static_std)  # [3,T,H,W]
            flow_t   = resize_and_normalize_clip(flow_clip,   H, W, self.flow_mean,   self.flow_std)    # [3,T,H,W]
            return static_t, flow_t, torch.tensor(int(sid), dtype=torch.long)

        except Exception as e:
            if self.print_bad_sample:
                print(f"[BAD SAMPLE] v={vpath} f={fpath} err={repr(e)}")
            static_t = torch.zeros(3, self.cfg.static_frames, H, W, dtype=torch.float32)
            flow_t   = torch.zeros(self.cfg.flow_channels, self.cfg.flow_frames, H, W, dtype=torch.float32)
            return static_t, flow_t, torch.tensor(self.cfg.broken_sample_id, dtype=torch.long)


# =========================
# 7) safe collate
# =========================
def safe_collate_fn(batch):
    static_list, flow_list, y_list = zip(*batch)
    static_b = torch.stack([t.contiguous().clone() for t in static_list], dim=0)
    flow_b   = torch.stack([t.contiguous().clone() for t in flow_list], dim=0)
    y_b      = torch.stack(list(y_list), dim=0)
    return static_b, flow_b, y_b


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
            print("\n[DEBUG][StaticExtractor] input:", tuple(x.shape))
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
            print("\n[DEBUG][DynamicExtractor] input:", tuple(x.shape), "contig=", x.is_contiguous())

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
    c = (ps.T @ pd) / B
    return (c ** 2).mean()

def mi_suppression_anti_infonce(ps: torch.Tensor, pd: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    B = ps.size(0)
    if B <= 1:
        return ps.new_tensor(0.0)

    psn = F.normalize(ps, dim=1)
    pdn = F.normalize(pd, dim=1)
    logits = (psn @ pdn.t()) / max(tau, 1e-6)  # [B,B]
    target = torch.arange(B, device=ps.device, dtype=torch.long)
    ce = F.cross_entropy(logits, target, reduction="mean")
    return (math_log(B, device=ps.device) - ce)

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
        total = 0.0
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
        return total, info


# =========================
# 10) batch selectors
# =========================
def select_train_batch(static_x, flow_x, y, device):
    idx = torch.nonzero(y >= 0, as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return None
    static_x = static_x.index_select(0, idx).to(device, non_blocking=True).contiguous()
    flow_x   = flow_x.index_select(0, idx).to(device, non_blocking=True).contiguous()
    y        = y.index_select(0, idx).to(device, non_blocking=True).contiguous()
    return static_x, flow_x, y

def select_nonbroken_batch(static_x, flow_x, y, device):
    idx = torch.nonzero(y != -1, as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return None
    static_x = static_x.index_select(0, idx).to(device, non_blocking=True).contiguous()
    flow_x   = flow_x.index_select(0, idx).to(device, non_blocking=True).contiguous()
    y        = y.index_select(0, idx).to(device, non_blocking=True).contiguous()
    return static_x, flow_x, y


# =========================
# 11) Train / Val / External Val / Neutral
# =========================
def data_check_stats(y: torch.Tensor) -> Tuple[int, int]:
    total = int(y.numel())
    bad = int((y == -1).sum().item())
    return total, bad


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

    mi_strength = cosine_ramp(epoch_idx, cfg.mi_warmup_epochs, cfg.mi_max_strength)

    # throughput timing (helps you confirm GPU-starved)
    t_last = time.time()
    t_data_wait_sum = 0.0
    t_step_sum = 0.0
    log_every = 200  # batches

    pbar = tqdm(loader, desc=f"Train(E{epoch_idx+1})", ncols=140)
    for it, (static_x, flow_x, y) in enumerate(pbar, start=1):
        t_now = time.time()
        t_data_wait = t_now - t_last  # time spent waiting for next batch
        t_data_wait_sum += t_data_wait
        t_step0 = t_now

        total_batches += 1
        t, b = data_check_stats(y)
        total_samples += t
        bad_samples += b

        pack = select_train_batch(static_x, flow_x, y, device)
        if pack is None:
            skipped_batches += 1
            t_last = time.time()
            continue
        static_x, flow_x, y2 = pack

        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=cfg.use_amp and device.type == "cuda"):
            ps, pd, ls, ld, lf = model(static_x, flow_x)

            Ls = ce(ls, y2)
            Ld = ce(ld, y2)
            Lf = ce(lf, y2)

            Lind = indep_correlation_loss(ps, pd) * cfg.lambda_indep
            Lmi = mi_suppression_anti_infonce(ps, pd, tau=cfg.mi_tau) * (cfg.lambda_mi * mi_strength)

            losses = {"cls_s": Ls, "cls_d": Ld, "cls_f": Lf, "indep": Lind, "mi": Lmi}
            loss, winfo = loss_weighter(losses)

        # safety: if NaN/Inf appears, print and skip this batch (keeps run alive; does not change intended objective when stable)
        if not torch.isfinite(loss).all():
            print(f"[NaN/Inf][Train] epoch={epoch_idx+1} iter={it} loss={loss.detach().cpu().item()}")
            skipped_batches += 1
            # reset scaler a bit to avoid permanent NaN
            if scaler is not None:
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

        t_step = time.time() - t_step0
        t_step_sum += t_step
        t_last = time.time()

        if it % log_every == 0:
            avg_wait = t_data_wait_sum / max(it, 1)
            avg_step = t_step_sum / max(it, 1)
            print(f"[Perf][Train] batches={it} avg_data_wait={avg_wait:.3f}s avg_step={avg_step:.3f}s "
                  f"(if avg_data_wait >> avg_step => GPU starved by data)")

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
        f"batches={total_batches} | skipped_batches={skipped_batches} | "
        f"samples={total_samples} | bad_samples={bad_samples} | "
        f"bad_ratio={bad_samples/max(total_samples,1):.2%}"
    )

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

    mi_strength = cosine_ramp(epoch_idx, cfg.mi_warmup_epochs, cfg.mi_max_strength)

    pbar = tqdm(loader, desc=f"ValIn(E{epoch_idx+1})", ncols=140)
    for static_x, flow_x, y in pbar:
        total_batches += 1
        t, b = data_check_stats(y)
        total_samples += t
        bad_samples += b

        pack = select_train_batch(static_x, flow_x, y, device)
        if pack is None:
            skipped_batches += 1
            continue
        static_x, flow_x, y2 = pack

        with torch.cuda.amp.autocast(enabled=cfg.use_amp and device.type == "cuda"):
            ps, pd, ls, ld, lf = model(static_x, flow_x)

            Ls = ce(ls, y2)
            Ld = ce(ld, y2)
            Lf = ce(lf, y2)
            Lind = indep_correlation_loss(ps, pd) * cfg.lambda_indep
            Lmi = mi_suppression_anti_infonce(ps, pd, tau=cfg.mi_tau) * (cfg.lambda_mi * mi_strength)

            losses = {"cls_s": Ls, "cls_d": Ld, "cls_f": Lf, "indep": Lind, "mi": Lmi}
            loss, _ = loss_weighter(losses)

        if not torch.isfinite(loss).all():
            print(f"[NaN/Inf][ValIn] epoch={epoch_idx+1} loss={loss.detach().cpu().item()}")
            skipped_batches += 1
            continue

        bs = int(y2.size(0))
        loss_sum += float(loss.item()) * bs
        corr_sum += correct_count(lf, y2)
        n_sum += bs

        pbar.set_postfix({
            "loss(avg)": f"{loss_sum/max(n_sum,1):.4f}",
            "acc(fusion,avg)": f"{corr_sum/max(n_sum,1):.4f}",
            "bad_ratio": f"{bad_samples/max(total_samples,1):.2%}",
        })

    print(
        f"[ValIn][DataCheck] "
        f"batches={total_batches} | skipped_batches={skipped_batches} | "
        f"samples={total_samples} | bad_samples={bad_samples} | "
        f"bad_ratio={bad_samples/max(total_samples,1):.2%}"
    )

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

    pbar = tqdm(loader, desc=f"ValExt(E{epoch_idx+1})", ncols=140)
    for static_x, flow_x, y in pbar:
        total_batches += 1
        t, b = data_check_stats(y)
        total_samples += t
        bad_samples += b

        pack = select_nonbroken_batch(static_x, flow_x, y, device)
        if pack is None:
            skipped_batches += 1
            continue
        static_x, flow_x, _y = pack

        with torch.cuda.amp.autocast(enabled=cfg.use_amp and device.type == "cuda"):
            ps, pd, ls, ld, lf = model(static_x, flow_x)
            indep = indep_correlation_loss(ps, pd)

            ent = softmax_entropy(lf)
            conf = torch.softmax(lf, dim=1).max(dim=1).values

        if (not torch.isfinite(indep).all()) or (not torch.isfinite(ent).all()) or (not torch.isfinite(conf).all()):
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
        f"batches={total_batches} | skipped_batches={skipped_batches} | "
        f"samples={total_samples} | bad_samples={bad_samples} | "
        f"bad_ratio={bad_samples/max(total_samples,1):.2%}"
    )

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
    for static_x, flow_x, y in pbar:
        pack = select_train_batch(static_x, flow_x, y, device)
        if pack is None:
            continue
        static_x, flow_x, _y = pack

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

    speaker2id, train_samples, val_samples, ext_val_samples = build_index_with_cache(cfg)
    print(f"[Split] train={len(train_samples)} val(in)={len(val_samples)} speakers(train)={len(speaker2id)}")
    if not cfg.disable_external_val:
        print(f"[ExternalVal] samples={len(ext_val_samples)} root={cfg.val_video_root}")

    train_ds = PersonalityDataset(train_samples, cfg, is_train=True,  print_bad_sample=args.print_bad_sample)
    val_ds   = PersonalityDataset(val_samples,   cfg, is_train=False, print_bad_sample=args.print_bad_sample)
    ext_val_ds = PersonalityDataset(ext_val_samples, cfg, is_train=False, print_bad_sample=args.print_bad_sample)

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
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
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
        prefetch_factor=2 if cfg.num_workers > 0 else None,
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
        prefetch_factor=2 if cfg.num_workers > 0 else None,
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

    print(f"\nDone. Best in-domain val acc={best_val:.4f}")
    print(f"Best: {ckpt_best}")
    print(f"Last: {ckpt_last}")


if __name__ == "__main__":
    main()
