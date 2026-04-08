# feature_ext.py
# -*- coding: utf-8 -*-

import os
import re
import json
import random
import argparse
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

# -------------------------
# 0) 先解析参数 & 设置 GPU 可见性（必须在 import torch 前）
# -------------------------
def parse_args():
    p = argparse.ArgumentParser("Stage1 Personality Feature Training (CMLR relpath match)")

    # paths
    p.add_argument("--video_root", type=str,
                   default="/mnt/netdisk/dataset/lipreading/datasets/CMLR/cmlr/cmlr_video_seg24s/",
                   help="root of original videos")
    p.add_argument("--flow_root", type=str,
                   default="/home/liuyang/Project/flow/flow_sequence/cmlr/",
                   help="root of flow frames")
    p.add_argument("--debug_shapes", action="store_true")

    # gpu
    p.add_argument("--gpus", type=str, default="0", help="GPU ids, e.g. '0' or '1,4'")
    p.add_argument("--no_cuda", action="store_true", help="force CPU")

    # output
    p.add_argument("--save_dir", type=str, default="./checkpoints_stage1")
    p.add_argument("--exp_name", type=str, default="personality_stage1_relpath_cli")

    # train
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--use_amp", action="store_true", help="enable AMP")
    p.add_argument("--seed", type=int, default=42)

    # shapes
    p.add_argument("--frame_h", type=int, default=112)
    p.add_argument("--frame_w", type=int, default=112)
    p.add_argument("--static_frames", type=int, default=1)
    p.add_argument("--flow_frames", type=int, default=32)
    p.add_argument("--flow_channels", type=int, default=3, help="RGB flow visualization usually=3")

    # model
    p.add_argument("--feat_dim", type=int, default=128)
    p.add_argument("--freeze_resnet_until", type=str, default="layer3",
                   help="freeze resnet until this layer name; use 'none' to disable")

    # loss
    p.add_argument("--lambda_indep", type=float, default=0.1)
    p.add_argument("--label_smoothing", type=float, default=0.0)

    # split
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--split_by_speaker", action="store_true")

    # debug
    p.add_argument("--max_samples", type=int, default=-1, help="limit samples for debug")
    p.add_argument("--print_bad_sample", action="store_true", help="print bad sample path in dataset exception")

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


# =========================
# 2) Config
# =========================
@dataclass
class Config:
    video_root: str
    flow_root: str
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

    frame_size: Tuple[int, int]
    static_frames: int
    flow_frames: int
    flow_channels: int

    feat_dim: int
    freeze_resnet_until: Optional[str]

    lambda_indep: float
    label_smoothing: float

    val_ratio: float
    split_by_speaker: bool

    speaker_prefix: str = "s"
    video_exts: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    img_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def build_config(a) -> Config:
    frz = None if str(a.freeze_resnet_until).lower() in ("none", "null", "") else a.freeze_resnet_until
    return Config(
        video_root=a.video_root,
        flow_root=a.flow_root,
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

        frame_size=(a.frame_h, a.frame_w),
        static_frames=a.static_frames,
        flow_frames=a.flow_frames,
        flow_channels=a.flow_channels,

        feat_dim=a.feat_dim,
        freeze_resnet_until=frz,

        lambda_indep=a.lambda_indep,
        label_smoothing=a.label_smoothing,

        val_ratio=a.val_ratio,
        split_by_speaker=a.split_by_speaker,
    )


# =========================
# 3) utils
# =========================
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def is_video_file(p: str, exts: Tuple[str, ...]) -> bool:
    return os.path.splitext(p)[1].lower() in exts


def is_image_file(p: str, exts: Tuple[str, ...]) -> bool:
    return os.path.splitext(p)[1].lower() in exts


def list_speakers(root: str, prefix: str) -> List[str]:
    sps = []
    for name in os.listdir(root):
        full = os.path.join(root, name)
        if os.path.isdir(full) and name.startswith(prefix):
            sps.append(name)
    sps.sort(key=natural_key)
    return sps


def sample_indices(n: int, t: int) -> np.ndarray:
    """
    返回长度恒为 t 的索引序列；n < t 时会重复采样（linspace 会产生重复整数）
    """
    if n <= 0:
        return np.zeros((t,), dtype=np.int64)
    if n == 1:
        return np.zeros((t,), dtype=np.int64)
    return np.linspace(0, n - 1, t).astype(np.int64)


# =========================
# 4) scan: rel_key = relative path inside speaker dir w/o ext
#    video: s1/20170903/sec1.mp4 -> rel_key=20170903/sec1
#    flow : s1/20170903/sec1/flow_*.jpg -> rel_key=20170903/sec1
# =========================
def discover_video_items_with_relkey(speaker_video_dir: str, cfg: Config) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for root, dirs, files in os.walk(speaker_video_dir):
        for fn in files:
            fp = os.path.join(root, fn)
            if is_video_file(fp, cfg.video_exts):
                rel = os.path.relpath(fp, speaker_video_dir)      # 20170903/sec1.mp4
                rel_no_ext = os.path.splitext(rel)[0]             # 20170903/sec1
                items.append((fp, rel_no_ext))

        # optional: frame folders as "video"
        for d in dirs:
            dp = os.path.join(root, d)
            try:
                ims = [x for x in os.listdir(dp) if is_image_file(x, cfg.img_exts)]
                if len(ims) >= 3:
                    rel = os.path.relpath(dp, speaker_video_dir)  # 20170903/sec1
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


def build_samples(cfg: Config, max_samples: int = -1) -> Tuple[List[Tuple[str, str, int]], Dict[str, int]]:
    speakers = list_speakers(cfg.video_root, cfg.speaker_prefix)
    speaker2id = {s: i for i, s in enumerate(speakers)}

    samples: List[Tuple[str, str, int]] = []
    miss_flow = 0

    for spk in speakers:
        v_dir = os.path.join(cfg.video_root, spk)
        f_dir = os.path.join(cfg.flow_root, spk)
        if not os.path.isdir(v_dir):
            continue

        video_items = discover_video_items_with_relkey(v_dir, cfg)
        for vpath, rel_key in video_items:
            flow_path = find_flow_by_relkey(f_dir, rel_key) if os.path.isdir(f_dir) else None
            if flow_path is None:
                miss_flow += 1
                continue
            samples.append((vpath, flow_path, speaker2id[spk]))

            if max_samples > 0 and len(samples) >= max_samples:
                break
        if max_samples > 0 and len(samples) >= max_samples:
            break

    print(f"[Scan] speakers={len(speakers)} samples={len(samples)} missing_flow={miss_flow}")
    return samples, speaker2id


def split_train_val(samples: List[Tuple[str, str, int]], cfg: Config):
    if cfg.split_by_speaker:
        spk_ids = sorted(list(set([s[2] for s in samples])))
        random.shuffle(spk_ids)
        n_val = max(1, int(len(spk_ids) * cfg.val_ratio))
        val_spk = set(spk_ids[:n_val])
        train = [x for x in samples if x[2] not in val_spk]
        val = [x for x in samples if x[2] in val_spk]
        return train, val

    by_spk: Dict[int, List] = {}
    for s in samples:
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


# =========================
# 5) I/O helpers (关键：读帧必须“固定长度”，失败帧用 0 填)
# =========================
def count_video_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def read_video_frames_by_indices(video_path: str, indices: np.ndarray,
                                fallback_hw: Tuple[int, int] = (112, 112)) -> List[np.ndarray]:
    """
    返回长度恒为 len(indices) 的 RGB 帧列表，读失败用 0 填。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # 全填 0
        H, W = fallback_hw
        z = np.zeros((H, W, 3), dtype=np.float32)
        return [z.copy() for _ in range(len(indices))]

    # 找一帧做 fallback
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


def read_framefolder_by_indices(folder: str, indices: np.ndarray, img_exts: Tuple[str, ...],
                               fallback_hw: Tuple[int, int] = (112, 112)) -> List[np.ndarray]:
    names = [x for x in os.listdir(folder) if is_image_file(x, img_exts)]
    names.sort(key=natural_key)
    if len(names) == 0:
        H, W = fallback_hw
        z = np.zeros((H, W, 3), dtype=np.float32)
        return [z.copy() for _ in range(len(indices))]

    # 找 fallback
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


def read_flow_dir_by_indices(flow_dir: str, indices: np.ndarray, img_exts: Tuple[str, ...]) -> List[np.ndarray]:
    """
    关键修复点：
    - 返回长度恒为 len(indices)
    - 任意帧读失败，用 0 图补齐（或用 fallback）
    """
    names = [x for x in os.listdir(flow_dir) if is_image_file(x, img_exts)]
    names.sort(key=natural_key)
    if len(names) == 0:
        return []

    # 找一张能读的 fallback
    fallback = None
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
            im = np.zeros_like(fallback)       # 缺帧补 0
        else:
            fallback = im

        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.astype(np.float32) / 255.0
        out.append(im)
    return out


def resize_and_normalize_clip(clip: np.ndarray, out_h: int, out_w: int,
                              mean: Tuple[float, ...], std: Tuple[float, ...]) -> torch.Tensor:
    # clip: [T,H,W,C] float32
    T, H, W, C = clip.shape
    x = torch.from_numpy(clip).permute(0, 3, 1, 2)  # [T,C,H,W]
    x = F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False)
    mean_t = torch.tensor(mean, dtype=x.dtype).view(1, C, 1, 1)
    std_t  = torch.tensor(std,  dtype=x.dtype).view(1, C, 1, 1)
    x = (x - mean_t) / (std_t + 1e-6)
    # clone: 避免 dataloader/pin_memory 下偶发 “storage not resizable”
    x = x.permute(1, 0, 2, 3).contiguous().clone()  # [C,T,H,W]
    return x


# =========================
# 6) Dataset
# =========================
class PersonalityDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, str, int]], cfg: Config, is_train: bool, print_bad_sample: bool = False):
        self.samples = samples
        self.cfg = cfg
        self.is_train = is_train
        self.print_bad_sample = print_bad_sample

        self.static_mean = (0.485, 0.456, 0.406)
        self.static_std  = (0.229, 0.224, 0.225)

        # RGB flow vis: map [0,1] -> [-1,1]
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
                n = count_video_frames(vpath)
                if n <= 0:
                    raise RuntimeError(f"bad video: {vpath}")
                if self.is_train and self.cfg.static_frames == 1:
                    sidx = np.array([random.randint(0, n - 1)], dtype=np.int64)
                else:
                    sidx = sample_indices(n, self.cfg.static_frames)
                static_frames = read_video_frames_by_indices(vpath, sidx, fallback_hw=(H, W))

            static_clip = np.stack(static_frames, axis=0)  # [T,H,W,3]

            # ---- flow ----
            if os.path.isfile(fpath) and fpath.lower().endswith(".npy"):
                arr = np.load(fpath).astype(np.float32)
                if arr.ndim != 4:
                    raise RuntimeError(f"bad flow npy shape: {arr.shape} {fpath}")
                # align channels
                if arr.shape[-1] != self.cfg.flow_channels:
                    if arr.shape[-1] > self.cfg.flow_channels:
                        arr = arr[..., :self.cfg.flow_channels]
                    else:
                        pad_c = self.cfg.flow_channels - arr.shape[-1]
                        arr = np.concatenate([arr] + [arr[..., :1]] * pad_c, axis=-1)
                nflow = arr.shape[0]
                fidx = sample_indices(max(nflow, 1), self.cfg.flow_frames)
                flow_clip = arr[fidx]  # [T,H,W,C]
            else:
                names = [x for x in os.listdir(fpath) if is_image_file(x, self.cfg.img_exts)]
                nflow = len(names)
                if nflow <= 0:
                    raise RuntimeError(f"empty flow dir: {fpath}")
                fidx = sample_indices(max(nflow, 1), self.cfg.flow_frames)
                flows = read_flow_dir_by_indices(fpath, fidx, self.cfg.img_exts)
                if len(flows) != self.cfg.flow_frames:
                    # 双保险：不允许长度不一致
                    if len(flows) == 0:
                        raise RuntimeError(f"flow read fail: {fpath}")
                    if len(flows) < self.cfg.flow_frames:
                        pad = [np.zeros_like(flows[-1]) for _ in range(self.cfg.flow_frames - len(flows))]
                        flows = flows + pad
                    else:
                        flows = flows[:self.cfg.flow_frames]
                flow_clip = np.stack(flows, axis=0)  # [T,H,W,3]

            # ---- resize + normalize ----
            static_t = resize_and_normalize_clip(static_clip, H, W, self.static_mean, self.static_std)  # [3,T,H,W]
            flow_t   = resize_and_normalize_clip(flow_clip,   H, W, self.flow_mean,   self.flow_std)    # [3,T,H,W]

            return static_t, flow_t, torch.tensor(sid, dtype=torch.long)

        except Exception as e:
            if self.print_bad_sample:
                print(f"[BAD SAMPLE] v={vpath} f={fpath} err={repr(e)}")
            static_t = torch.zeros(3, self.cfg.static_frames, H, W, dtype=torch.float32)
            flow_t   = torch.zeros(self.cfg.flow_channels, self.cfg.flow_frames, H, W, dtype=torch.float32)
            return static_t, flow_t, torch.tensor(-1, dtype=torch.long)


# =========================
# 7) safe collate (推荐：彻底避免 shared-memory resize 问题)
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
    def __init__(self, cfg: Config):
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

    def forward(self, x):
        # x: [B,3,T,H,W]
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        feat = self.backbone(x).view(B * T, 512)
        feat = feat.view(B, T, 512).mean(dim=1)
        return self.proj(feat)  # [B,D]


class DynamicExtractor(nn.Module):
    """
    双分支动态个性特征：
    A) RGB flow 可视化序列 -> 3D CNN
    B) 时间差分/能量序列 -> 1D CNN + 统计量
    """
    def __init__(self, cfg: Config):
        super().__init__()
        D = cfg.feat_dim
        Dv = D // 2
        De = D - Dv

        # Branch A: visual
        self.visual_net = nn.Sequential(
            nn.Conv3d(cfg.flow_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool3d((4, 4, 4)),
        )
        self.visual_proj = nn.Sequential(
            nn.Linear(256 * 4 * 4 * 4, Dv),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # Branch B: energy/time-diff
        self.energy_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # [B,64,1]
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

    def forward(self, x):
        # x: [B,C,T,H,W]
        B, C, T, H, W = x.shape

        v = self.visual_net(x).flatten(1)
        v = self.visual_proj(v)

        if T >= 2:
            diff = x[:, :, 1:] - x[:, :, :-1]              # [B,C,T-1,H,W]
            energy = diff.abs().mean(dim=(1, 3, 4))         # [B,T-1]
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
        return out


class PersonalityNet(nn.Module):
    def __init__(self, cfg: Config, num_speakers: int):
        super().__init__()
        self.static = StaticExtractor(cfg)
        self.dynamic = DynamicExtractor(cfg)
        self.spk_head_s = nn.Linear(cfg.feat_dim, num_speakers)
        self.spk_head_d = nn.Linear(cfg.feat_dim, num_speakers)
        self.spk_head_f = nn.Linear(cfg.feat_dim * 2, num_speakers)

    @torch.no_grad()
    def get_embeddings(self, static_x, flow_x):
        return self.static(static_x), self.dynamic(flow_x)

    def forward(self, static_x, flow_x):
        ps = self.static(static_x)
        pd = self.dynamic(flow_x)
        ls = self.spk_head_s(ps)
        ld = self.spk_head_d(pd)
        lf = self.spk_head_f(torch.cat([ps, pd], dim=1))
        return ps, pd, ls, ld, lf


# =========================
# 9) Loss + metrics
# =========================
def indep_correlation_loss(ps: torch.Tensor, pd: torch.Tensor) -> torch.Tensor:
    B, D = ps.shape
    ps = (ps - ps.mean(0)) / (ps.std(0) + 1e-6)
    pd = (pd - pd.mean(0)) / (pd.std(0) + 1e-6)
    c = (ps.T @ pd) / B
    return (c ** 2).mean()


def acc(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()


# =========================
# 10) Train / Val / Neutral codes
# =========================
def train_one_epoch(model, loader, opt, scaler, cfg: Config, device):
    model.train()
    ce = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    loss_sum, acc_sum, n = 0.0, 0.0, 0
    pbar = tqdm(loader, desc="Train", ncols=140)

    for static_x, flow_x, y in pbar:
        valid = (y != -1)
        if not valid.any():
            continue
        if args.debug_shapes and not printed:
            print("[DEBUG] static_x", static_x.shape)
            print("[DEBUG] flow_x  ", flow_x.shape)
            with torch.no_grad():
                ps, pd, ls, ld, lf = model(static_x[:2], flow_x[:2])  # 只取2个样本
            print("[DEBUG] ps", ps.shape, "pd", pd.shape)
            print("[DEBUG] ls", ls.shape, "ld", ld.shape, "lf", lf.shape)
            printed = True
        static_x = static_x[valid].to(device, non_blocking=True)
        flow_x   = flow_x[valid].to(device, non_blocking=True)
        y        = y[valid].to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            ps, pd, ls, ld, lf = model(static_x, flow_x)
            loss_cls = ce(ls, y) + ce(ld, y) + ce(lf, y)
            loss_ind = indep_correlation_loss(ps, pd) * cfg.lambda_indep
            loss = loss_cls + loss_ind

        scaler.scale(loss).backward()
        if cfg.grad_clip and cfg.grad_clip > 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        scaler.step(opt)
        scaler.update()

        bs = y.size(0)
        loss_sum += loss.item() * bs
        acc_sum  += acc(lf.detach(), y) * bs
        n += bs

        pbar.set_postfix({
            "loss(avg)": f"{loss_sum/max(n,1):.4f}",
            "cls": f"{loss_cls.item():.4f}",
            "ind": f"{loss_ind.item():.4f}",
            "acc(fusion,avg)": f"{acc_sum/max(n,1):.4f}",
        })

    return loss_sum / max(n, 1), acc_sum / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, cfg: Config, device):
    model.eval()
    ce = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    loss_sum, acc_sum, n = 0.0, 0.0, 0
    pbar = tqdm(loader, desc="Val", ncols=140)

    for static_x, flow_x, y in pbar:
        valid = (y != -1)
        if not valid.any():
            continue

        static_x = static_x[valid].to(device, non_blocking=True)
        flow_x   = flow_x[valid].to(device, non_blocking=True)
        y        = y[valid].to(device, non_blocking=True)

        ps, pd, ls, ld, lf = model(static_x, flow_x)
        loss_cls = ce(ls, y) + ce(ld, y) + ce(lf, y)
        loss_ind = indep_correlation_loss(ps, pd) * cfg.lambda_indep
        loss = loss_cls + loss_ind

        bs = y.size(0)
        loss_sum += loss.item() * bs
        acc_sum  += acc(lf, y) * bs
        n += bs

        pbar.set_postfix({
            "loss(avg)": f"{loss_sum/max(n,1):.4f}",
            "acc(fusion,avg)": f"{acc_sum/max(n,1):.4f}",
        })

    return loss_sum / max(n, 1), acc_sum / max(n, 1)


@torch.no_grad()
def compute_neutral_codes(model, loader, cfg: Config, device):
    model.eval()
    sum_ps = torch.zeros(cfg.feat_dim, device=device)
    sum_pd = torch.zeros(cfg.feat_dim, device=device)
    cnt = 0

    for static_x, flow_x, y in tqdm(loader, desc="Compute neutral codes", ncols=120):
        valid = (y != -1)
        if not valid.any():
            continue

        static_x = static_x[valid].to(device, non_blocking=True)
        flow_x   = flow_x[valid].to(device, non_blocking=True)

        ps, pd = model.get_embeddings(static_x, flow_x)
        sum_ps += ps.sum(dim=0)
        sum_pd += pd.sum(dim=0)
        cnt += ps.size(0)

    ps0 = (sum_ps / max(cnt, 1)).detach().cpu()
    pd0 = (sum_pd / max(cnt, 1)).detach().cpu()
    return ps0, pd0


# =========================
# 11) Main
# =========================
def main():
    cfg = build_config(args)
    seed_all(cfg.seed)

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    os.makedirs(cfg.save_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.save_dir, f"{cfg.exp_name}.pth")
    cfg_path  = os.path.join(cfg.save_dir, f"{cfg.exp_name}.config.json")

    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    print(f"[Config] {cfg_path}")
    print(f"[Device] {device} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")

    samples, speaker2id = build_samples(cfg, max_samples=args.max_samples)
    if len(samples) == 0:
        raise RuntimeError("No samples matched. Check video_root/flow_root structure.")

    train_samples, val_samples = split_train_val(samples, cfg)
    print(f"[Split] train={len(train_samples)} val={len(val_samples)} speakers={len(speaker2id)}")

    train_ds = PersonalityDataset(train_samples, cfg, is_train=True,  print_bad_sample=args.print_bad_sample)
    val_ds   = PersonalityDataset(val_samples,   cfg, is_train=False, print_bad_sample=args.print_bad_sample)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        collate_fn=safe_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
        collate_fn=safe_collate_fn,
    )

    model = PersonalityNet(cfg, num_speakers=len(speaker2id)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp and use_cuda)

    best_val = -1.0
    for ep in range(cfg.epochs):
        print(f"\n===== Epoch {ep+1}/{cfg.epochs} =====")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, scaler, cfg, device)
        va_loss, va_acc = evaluate(model, val_loader, cfg, device)
        print(f"[Epoch {ep+1}] train loss={tr_loss:.4f} acc={tr_acc:.4f} | val loss={va_loss:.4f} acc={va_acc:.4f}")

        if va_acc > best_val:
            best_val = va_acc
            ps0, pd0 = compute_neutral_codes(model, train_loader, cfg, device)

            torch.save({
                "cfg": asdict(cfg),
                "speaker2id": speaker2id,
                "model_state": model.state_dict(),
                "best_val_acc": best_val,
                "ps0": ps0,
                "pd0": pd0,
            }, ckpt_path)
            print(f"[Save] {ckpt_path} (best_val_acc={best_val:.4f})")

    print(f"\nDone. Best val acc={best_val:.4f}")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
