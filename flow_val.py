#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resume RAFT optical flow generation based on per-folder progress.json
(Worker auto-respawn, task retry N times, OOM microbatch auto-down, illegal-access safe,
chunked ffmpeg decode, CPU-only flow_to_image).

Rules:
- If <flow_dir>/progress.json exists AND <flow_dir>/DONE does NOT exist -> unfinished -> (re)generate missing flows.
- If DONE exists -> finished -> skip.
- Do NOT scan video_root. Only rely on progress.json to find video_path.
- Keep original params consistency:
    BATCH_SIZE = 48
    TARGET_SIZE = (520, 960)   # must be multiple of 8
    SKIP_FRAMES = 1            # consecutive frames (n and n+1)
    RESUME = True
    RECHECK_BACK = 128
"""

import os
import re
import json
import time
import argparse
import subprocess
import tempfile
import shutil
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from PIL import Image


FLOW_NAME_RE = re.compile(r"^flow_(\d{6})\.jpg$", re.IGNORECASE)
PROGRESS_FILE = "progress.json"
DONE_FILE = "DONE"


# ======================== 0) Args ========================
def parse_args():
    p = argparse.ArgumentParser("Resume RAFT flow from progress.json (multi-GPU, respawn, retry, OOM-safe)")

    p.add_argument("--flow_root", type=str, required=True,
                   help="flow root that contains many sample dirs (each has progress.json)")
    p.add_argument("--gpus", type=str, default="0",
                   help="physical GPU ids, e.g. '0' or '0,2' or '7'")

    # keep same defaults as your original
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--target_h", type=int, default=520)
    p.add_argument("--target_w", type=int, default=960)
    p.add_argument("--recheck_back", type=int, default=128)

    # behavior
    p.add_argument("--overwrite_done", action="store_true",
                   help="if set, also re-run dirs even if DONE exists (DANGEROUS). default False")
    p.add_argument("--max_tasks", type=int, default=-1,
                   help="limit tasks for debugging")
    p.add_argument("--print_every", type=int, default=50,
                   help="print progress every N done tasks (per worker)")
    p.add_argument("--sleep_ms", type=int, default=0,
                   help="sleep ms between tasks (reduce IO pressure)")

    # robustness / perf
    p.add_argument("--amp", action="store_true",
                   help="enable AMP inference (if illegal access appears, run WITHOUT --amp).")
    p.add_argument("--ffprobe", type=str, default="ffprobe", help="ffprobe binary")
    p.add_argument("--ffmpeg", type=str, default="ffmpeg", help="ffmpeg binary")
    p.add_argument("--tmp_dir", type=str, default="", help="optional temp dir base (e.g. /dev/shm)")
    p.add_argument("--sync_each_step", action="store_true",
                   help="torch.cuda.synchronize() after each micro-batch (slower but helps debug/robust)")

    # decoding chunk limit (protect RAM/tmp usage)
    p.add_argument("--max_decode_frames", type=int, default=512,
                   help="max frames decoded in one ffmpeg call (run will be further split if needed).")

    # retry / respawn
    p.add_argument("--task_retries", type=int, default=5,
                   help="retry each dir up to N times if failed (OOM/illegal/etc.).")
    p.add_argument("--exit_on_illegal", action="store_true",
                   help="if illegal memory access happens, kill the worker immediately (recommended).")
    p.add_argument("--max_respawn", type=int, default=1000000,
                   help="max respawn times per gpu (safety).")

    return p.parse_args()


# ======================== 1) Small helpers ========================
def flow_path(flow_dir: str, idx: int) -> str:
    return os.path.join(flow_dir, f"flow_{idx:06d}.jpg")

def progress_path(flow_dir: str) -> str:
    return os.path.join(flow_dir, PROGRESS_FILE)

def done_path(flow_dir: str) -> str:
    return os.path.join(flow_dir, DONE_FILE)

def atomic_write_text(path: str, text: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def save_progress(flow_dir: str, next_idx: int, video_path: str):
    data = {
        "next_idx": int(next_idx),
        "video_path": video_path,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_write_text(progress_path(flow_dir), json.dumps(data, ensure_ascii=False))

def load_progress(flow_dir: str) -> Optional[Dict]:
    p = progress_path(flow_dir)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return None
        if "video_path" not in obj:
            return None
        if "next_idx" not in obj:
            obj["next_idx"] = 0
        return obj
    except Exception:
        return None

def quick_max_existing_idx(flow_dir: str) -> int:
    max_idx = -1
    try:
        with os.scandir(flow_dir) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                m = FLOW_NAME_RE.match(entry.name)
                if m:
                    idx = int(m.group(1))
                    if idx > max_idx:
                        max_idx = idx
    except FileNotFoundError:
        return -1
    return max_idx

def cleanup_tmp_jpg(flow_dir: str):
    try:
        with os.scandir(flow_dir) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(".jpg.tmp"):
                    try:
                        os.remove(entry.path)
                    except Exception:
                        pass
    except FileNotFoundError:
        pass

def get_resume_start_idx(flow_dir: str, recheck_back: int) -> int:
    if os.path.exists(done_path(flow_dir)):
        return -1

    cleanup_tmp_jpg(flow_dir)

    prog = load_progress(flow_dir)
    if prog and "next_idx" in prog:
        next_idx = max(0, int(prog["next_idx"]))
    else:
        max_idx = quick_max_existing_idx(flow_dir)
        next_idx = max_idx + 1

    start = max(0, next_idx - int(recheck_back))
    for i in range(start, next_idx):
        if not os.path.exists(flow_path(flow_dir, i)):
            return i
    return next_idx


# ======================== 2) ffprobe / ffmpeg chunk decode ========================
def count_frames_ffprobe(ffprobe_bin: str, video_path: str) -> int:
    cmd = [
        ffprobe_bin, "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=nokey=1:noprint_wrappers=1",
        video_path
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8", errors="ignore").strip()
        n = int(out) if out else 0
        if n > 0:
            return n
    except Exception:
        pass

    cmd2 = [
        ffprobe_bin, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames",
        "-of", "default=nokey=1:noprint_wrappers=1",
        video_path
    ]
    try:
        out = subprocess.check_output(cmd2, stderr=subprocess.STDOUT).decode("utf-8", errors="ignore").strip()
        n = int(out) if out else 0
        if n > 0:
            return n
    except Exception:
        pass
    return 0

def decode_frames_range_ffmpeg(ffmpeg_bin: str,
                              video_path: str,
                              start_frame: int,
                              end_frame: int,
                              tmp_base: str = "") -> Tuple[str, List[str]]:
    if end_frame < start_frame:
        return "", []

    if tmp_base:
        os.makedirs(tmp_base, exist_ok=True)
        temp_dir = tempfile.mkdtemp(prefix="ffmpeg_frames_", dir=tmp_base)
    else:
        temp_dir = tempfile.mkdtemp(prefix="ffmpeg_frames_")

    out_pattern = os.path.join(temp_dir, "frame_%06d.jpg")
    vf = f"select='between(n\\,{start_frame}\\,{end_frame})'"

    cmd = [
        ffmpeg_bin,
        "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-vf", vf,
        "-vsync", "0",
        "-q:v", "1",
        "-f", "image2",
        out_pattern
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        files = [f for f in os.listdir(temp_dir) if f.startswith("frame_") and f.endswith(".jpg")]
        files.sort()
        return temp_dir, files
    except subprocess.CalledProcessError as e:
        err = (e.stderr.decode("utf-8", errors="ignore") if e.stderr else str(e))
        print(f"❌ ffmpeg decode failed: {video_path} range=[{start_frame},{end_frame}] err={err}", flush=True)
        shutil.rmtree(temp_dir, ignore_errors=True)
        return "", []


# ======================== 3) Atomic JPEG write ========================
def atomic_write_jpeg_tensor(flow_img_chw_uint8, dst_path: str):
    from torchvision.io import write_jpeg
    tmp = dst_path + ".tmp"
    write_jpeg(flow_img_chw_uint8, tmp)
    os.replace(tmp, dst_path)


# ======================== 4) Scan tasks ========================
def scan_unfinished(flow_root: Path, overwrite_done: bool, max_tasks: int) -> List[str]:
    tasks: List[str] = []
    for prog in flow_root.rglob(PROGRESS_FILE):
        flow_dir = str(prog.parent)
        if (not overwrite_done) and os.path.exists(os.path.join(flow_dir, DONE_FILE)):
            continue
        obj = load_progress(flow_dir)
        if obj is None:
            continue
        tasks.append(flow_dir)
        if max_tasks > 0 and len(tasks) >= max_tasks:
            break
    return tasks


# ======================== 5) Core generation ========================
def build_missing_list(flow_dir: str, total_pairs: int) -> List[int]:
    miss = []
    for i in range(total_pairs):
        if not os.path.exists(flow_path(flow_dir, i)):
            miss.append(i)
    return miss

def group_into_runs(idxs: List[int]) -> List[Tuple[int, int]]:
    if not idxs:
        return []
    runs = []
    a = b = idxs[0]
    for x in idxs[1:]:
        if x == b + 1:
            b = x
        else:
            runs.append((a, b))
            a = b = x
    runs.append((a, b))
    return runs

def split_run_if_too_long(a: int, b: int, max_decode_frames: int) -> List[Tuple[int, int]]:
    # run [a,b] needs frames [a, b+1] => len=(b-a+2) <= max_decode_frames
    max_flows = max(1, int(max_decode_frames) - 1)
    out = []
    cur = a
    while cur <= b:
        end = min(b, cur + max_flows - 1)
        out.append((cur, end))
        cur = end + 1
    return out

def _assert_finite_cpu(x, name: str):
    import torch
    if not torch.isfinite(x).all():
        raise RuntimeError(f"{name} has NaN/Inf (decode issue)")

def infer_flow_microbatch_cpu_image(model, img1_cpu, img2_cpu, device, use_amp: bool, sync_each_step: bool):
    """
    img1_cpu/img2_cpu: CPU float tensor [B,3,H,W] in [0,1]
    return: CPU uint8 [B,3,H,W]
    """
    import torch
    from torchvision.utils import flow_to_image

    _assert_finite_cpu(img1_cpu, "img1_cpu")
    _assert_finite_cpu(img2_cpu, "img2_cpu")

    # ensure contiguous BEFORE to(device)
    img1_cpu = img1_cpu.contiguous()
    img2_cpu = img2_cpu.contiguous()

    img1 = img1_cpu.to(device, non_blocking=True).contiguous()
    img2 = img2_cpu.to(device, non_blocking=True).contiguous()

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
            flow_list = model(img1, img2)
            pred = flow_list[-1]  # [B,2,H,W] on GPU/CPU

    pred_cpu = pred.detach().float().cpu()
    flow_img_cpu = flow_to_image(pred_cpu)  # CPU uint8

    del img1, img2, flow_list, pred
    if device.type == "cuda" and sync_each_step:
        torch.cuda.synchronize()

    return flow_img_cpu

def infer_with_auto_microbatch(model, img1_cpu, img2_cpu, device,
                               use_amp: bool, start_bs: int, sync_each_step: bool):
    """
    Auto-decrease micro-batch on OOM: start_bs -> 24 -> 12 -> 6 -> 3 -> 1
    Return: CPU uint8 [B,3,H,W]
    """
    import torch

    start_bs = max(1, int(start_bs))
    candidates = [start_bs]
    for v in [24, 12, 6, 3, 1]:
        if v < candidates[-1]:
            candidates.append(v)
    if candidates[-1] != 1:
        candidates.append(1)

    N = int(img1_cpu.size(0))
    outs = []
    s = 0
    while s < N:
        used = None
        last_err = None
        for mb in candidates:
            mb2 = min(mb, N - s)
            try:
                out = infer_flow_microbatch_cpu_image(
                    model=model,
                    img1_cpu=img1_cpu[s:s+mb2].contiguous(),
                    img2_cpu=img2_cpu[s:s+mb2].contiguous(),
                    device=device,
                    use_amp=use_amp,
                    sync_each_step=sync_each_step,
                )
                outs.append(out)
                used = mb2
                last_err = None
                break
            except RuntimeError as e:
                last_err = e
                msg = str(e).lower()
                if "out of memory" in msg and device.type == "cuda":
                    torch.cuda.empty_cache()
                    continue
                raise

        if used is None:
            raise RuntimeError(f"OOM even at micro-batch=1, last_err={repr(last_err)}")
        s += used

    return torch.cat(outs, dim=0)

def generate_flow_frames_for_dir(flow_dir: str,
                                 model,
                                 preprocess_transform,
                                 device,
                                 batch_size: int,
                                 target_size_hw: Tuple[int, int],
                                 recheck_back: int,
                                 ffprobe_bin: str,
                                 ffmpeg_bin: str,
                                 tmp_base: str,
                                 use_amp: bool,
                                 sync_each_step: bool,
                                 max_decode_frames: int) -> bool:
    prog = load_progress(flow_dir)
    if prog is None:
        print(f"⚠️ progress.json unreadable, skip: {flow_dir}", flush=True)
        return False

    video_path = str(prog["video_path"])
    if not os.path.exists(video_path):
        print(f"⚠️ video missing, skip: {video_path}", flush=True)
        return False

    if os.path.exists(done_path(flow_dir)):
        return True

    n_frames = count_frames_ffprobe(ffprobe_bin, video_path)
    if n_frames <= 1:
        print(f"⚠️ ffprobe frames<=1, skip: {video_path}", flush=True)
        return False
    total_pairs = n_frames - 1

    start_idx = get_resume_start_idx(flow_dir, recheck_back=recheck_back)
    if start_idx < 0:
        return True

    missing = build_missing_list(flow_dir, total_pairs)
    if not missing:
        atomic_write_text(done_path(flow_dir), "done\n")
        save_progress(flow_dir, total_pairs, video_path)
        print(f"✅ already complete: {flow_dir}", flush=True)
        return True

    first_missing = missing[0]
    save_progress(flow_dir, first_missing, video_path)

    runs0 = group_into_runs(missing)
    runs: List[Tuple[int, int]] = []
    for (a, b) in runs0:
        runs.extend(split_run_if_too_long(a, b, max_decode_frames=max_decode_frames))

    print(f"▶️ {Path(video_path).name} total_frames={n_frames} total_pairs={total_pairs} missing={len(missing)} runs={len(runs)} dir={flow_dir}", flush=True)

    import torch
    import torchvision.transforms.functional as TF

    def preprocess_frames(img1_batch, img2_batch):
        # resize then transforms (CPU)
        img1_batch = TF.resize(img1_batch, size=target_size_hw, antialias=False).contiguous()
        img2_batch = TF.resize(img2_batch, size=target_size_hw, antialias=False).contiguous()
        img1_batch, img2_batch = preprocess_transform(img1_batch, img2_batch)
        # IMPORTANT: ensure contiguous after transforms too
        return img1_batch.contiguous(), img2_batch.contiguous()

    for (a, b) in runs:
        f_start = a
        f_end = b + 1

        temp_dir, files = decode_frames_range_ffmpeg(ffmpeg_bin, video_path, f_start, f_end, tmp_base=tmp_base)
        if not temp_dir or len(files) < 2:
            print(f"⚠️ decode produced <2 frames, run=[{a},{b}] video={video_path}", flush=True)
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
            return False

        frames = []
        for fn in files:
            fp = os.path.join(temp_dir, fn)
            try:
                with Image.open(fp) as img:
                    img = img.convert("RGB")
                    frames.append(TF.to_tensor(img))
            except Exception as e:
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"❌ PIL open failed: {fp} err={repr(e)}", flush=True)
                return False

        frames = torch.stack(frames, dim=0).contiguous()  # CPU float

        local_flows = list(range(a, b + 1))
        i = 0
        while i < len(local_flows):
            cur = local_flows[i:i + int(batch_size)]
            idx0 = torch.tensor([k - f_start for k in cur], dtype=torch.long)
            img1 = frames.index_select(0, idx0).contiguous()
            img2 = frames.index_select(0, idx0 + 1).contiguous()

            img1, img2 = preprocess_frames(img1, img2)

            flow_imgs_cpu = infer_with_auto_microbatch(
                model=model,
                img1_cpu=img1,
                img2_cpu=img2,
                device=device,
                use_amp=use_amp,
                start_bs=int(batch_size),
                sync_each_step=sync_each_step,
            )

            for k, flow_img in zip(cur, flow_imgs_cpu):
                outp = flow_path(flow_dir, int(k))
                if not os.path.exists(outp):
                    atomic_write_jpeg_tensor(flow_img, outp)

            last_written = int(cur[-1])
            save_progress(flow_dir, last_written + 1, video_path)

            i += len(cur)

        shutil.rmtree(temp_dir, ignore_errors=True)

    missing2 = build_missing_list(flow_dir, total_pairs)
    if missing2:
        save_progress(flow_dir, missing2[0], video_path)
        print(f"⚠️ still missing after run, missing={len(missing2)} first={missing2[0]} dir={flow_dir}", flush=True)
        return False

    atomic_write_text(done_path(flow_dir), "done\n")
    save_progress(flow_dir, total_pairs, video_path)
    print(f"✅ DONE: {flow_dir}", flush=True)
    return True


# ======================== 6) Worker loop (queue-based) ========================
def worker_loop(rank: int,
                gpu_id: int,
                task_q,
                result_q,
                inflight_dict,
                args):
    """
    Worker continuously pulls tasks from task_q: (flow_dir, attempt)
    Updates inflight_dict[gpu_id] = (flow_dir, attempt) while processing.
    On illegal memory access (and --exit_on_illegal), process exits immediately.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")

    import torch
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

    # Make cudnn safer (benchmark can cause weird edge cases on some shapes)
    try:
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    print(f"[Worker{rank}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

    if not torch.cuda.is_available():
        print(f"[Worker{rank}] ❌ CUDA not available -> CPU (VERY slow)", flush=True)
        device = torch.device("cpu")
    else:
        print(f"[Worker{rank}] ✅ cuda available, device_count={torch.cuda.device_count()} (expect 1)", flush=True)
        device = torch.device("cuda:0")

    weights = Raft_Large_Weights.DEFAULT
    preprocess_transform = weights.transforms()
    model = raft_large(weights=weights, progress=False).to(device).eval()

    # TF32 optional
    if device.type == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    ok_cnt = 0
    fail_cnt = 0
    skip_cnt = 0
    done_cnt = 0

    while True:
        item = task_q.get()
        if item is None:
            # sentinel
            inflight_dict[str(gpu_id)] = None
            task_q.task_done()
            break

        flow_dir, attempt = item
        inflight_dict[str(gpu_id)] = (flow_dir, int(attempt), time.time())
        task_q.task_done()  # we mark "taken"; requeue handled by main if needed

        # skip if DONE and not overwrite
        if (not args.overwrite_done) and os.path.exists(done_path(flow_dir)):
            skip_cnt += 1
            done_cnt += 1
            inflight_dict[str(gpu_id)] = None
            continue

        try:
            ok = generate_flow_frames_for_dir(
                flow_dir=flow_dir,
                model=model,
                preprocess_transform=preprocess_transform,
                device=device,
                batch_size=int(args.batch_size),
                target_size_hw=(int(args.target_h), int(args.target_w)),
                recheck_back=int(args.recheck_back),
                ffprobe_bin=str(args.ffprobe),
                ffmpeg_bin=str(args.ffmpeg),
                tmp_base=str(args.tmp_dir).strip(),
                use_amp=bool(args.amp),
                sync_each_step=bool(args.sync_each_step),
                max_decode_frames=int(args.max_decode_frames),
            )
        except Exception as e:
            ok = False
            emsg = str(e).lower()
            print(f"[Worker{rank}|GPU{gpu_id}] ❌ EXCEPTION attempt={attempt} dir={flow_dir} err={repr(e)}", flush=True)

            if args.exit_on_illegal and ("illegal memory access" in emsg):
                # report fatal so main can requeue inflight and respawn
                result_q.put(("FATAL_ILLEGAL", gpu_id, flow_dir, int(attempt), repr(e)))
                print(f"[Worker{rank}|GPU{gpu_id}] 🔥 illegal memory access => exiting worker now", flush=True)
                os._exit(3)

        if ok:
            ok_cnt += 1
            result_q.put(("OK", gpu_id, flow_dir, int(attempt), ""))
        else:
            fail_cnt += 1
            result_q.put(("FAIL", gpu_id, flow_dir, int(attempt), ""))

        done_cnt += 1
        inflight_dict[str(gpu_id)] = None

        if (done_cnt % max(1, int(args.print_every))) == 0:
            print(f"[Worker{rank}|GPU{gpu_id}] done={done_cnt} ok={ok_cnt} fail={fail_cnt} skip={skip_cnt}", flush=True)

        if args.sleep_ms and args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    # graceful end stats
    result_q.put(("WORKER_DONE", gpu_id, "", -1, f"ok={ok_cnt} fail={fail_cnt} skip={skip_cnt}"))


# ======================== 7) Main: respawn workers + retry tasks ========================
def main():
    args = parse_args()
    flow_root = Path(args.flow_root)

    gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip() != ""]
    if len(gpus) == 0:
        raise ValueError("--gpus is empty")

    tasks = scan_unfinished(flow_root, overwrite_done=bool(args.overwrite_done), max_tasks=int(args.max_tasks))
    print(f"[Main] flow_root={flow_root}", flush=True)
    print(f"[Main] gpus={gpus} workers={len(gpus)} overwrite_done={bool(args.overwrite_done)} recheck_back={args.recheck_back}", flush=True)
    print(f"[Main] unfinished (progress.json exists && DONE missing) = {len(tasks)}", flush=True)

    if len(tasks) == 0:
        print("[Main] No tasks to process. Exit.", flush=True)
        return

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()

    task_q = ctx.JoinableQueue(maxsize=4096)
    result_q = ctx.Queue()
    inflight = manager.dict()  # key=str(gpu_id) -> (flow_dir, attempt, ts) or None

    # init inflight
    for gid in gpus:
        inflight[str(gid)] = None

    # push initial tasks (attempt=0)
    for d in tasks:
        task_q.put((d, 0))

    # worker process handles
    procs: Dict[int, mp.Process] = {}
    respawn_cnt: Dict[int, int] = {gid: 0 for gid in gpus}

    def spawn_worker(rank: int, gid: int):
        p = ctx.Process(target=worker_loop, args=(rank, gid, task_q, result_q, inflight, args))
        p.daemon = False
        p.start()
        procs[gid] = p
        respawn_cnt[gid] += 1
        print(f"[Main] spawn worker rank={rank} gpu={gid} respawn={respawn_cnt[gid]}", flush=True)

    # start workers
    for rank, gid in enumerate(gpus):
        spawn_worker(rank, gid)

    # bookkeeping
    total_tasks = len(tasks)
    done_ok = 0
    done_fail_final = 0
    done_skip = 0

    # Track final-state per flow_dir
    final_state: Dict[str, str] = {}  # "OK" | "FAIL" | "SKIP"
    # Track attempts
    attempts: Dict[str, int] = {}

    # helper: requeue with attempt+1 if allowed
    def requeue(flow_dir: str, prev_attempt: int, reason: str):
        nxt = prev_attempt + 1
        attempts[flow_dir] = nxt
        if nxt <= int(args.task_retries):
            print(f"[Main] 🔁 requeue attempt={nxt}/{args.task_retries} reason={reason} dir={flow_dir}", flush=True)
            task_q.put((flow_dir, nxt))
            return True
        else:
            print(f"[Main] ❌ give up after attempt={prev_attempt} dir={flow_dir}", flush=True)
            final_state[flow_dir] = "FAIL"
            return False

    # main loop until all tasks reach terminal state
    # Terminal if: OK or (SKIP) or (FAIL after retries exceeded)
    while True:
        # 1) respawn dead workers (and requeue inflight if any)
        for gid in list(procs.keys()):
            p = procs[gid]
            if not p.is_alive():
                exitcode = p.exitcode
                infl = inflight.get(str(gid), None)
                if infl is not None:
                    # worker died while handling a task => requeue it
                    flow_dir, att, ts = infl
                    print(f"[Main] ⚠️ worker gpu={gid} died exitcode={exitcode} inflight dir={flow_dir} attempt={att} => requeue", flush=True)
                    inflight[str(gid)] = None
                    requeue(flow_dir, int(att), reason=f"worker_died_exitcode={exitcode}")

                # respawn if under limit
                if respawn_cnt[gid] < int(args.max_respawn):
                    # keep same rank index by gid order
                    rank = gpus.index(gid)
                    spawn_worker(rank, gid)
                else:
                    print(f"[Main] ❌ reached max_respawn for gpu={gid}, will not respawn", flush=True)
                    # if no worker, tasks may stall; user can stop here
                    procs.pop(gid, None)

        # 2) process results if any
        drained = False
        while not result_q.empty():
            drained = True
            tag, gid, flow_dir, att, msg = result_q.get()

            if tag == "OK":
                if flow_dir not in final_state:
                    final_state[flow_dir] = "OK"
                    done_ok += 1
            elif tag == "FAIL":
                # task finished with failure in this attempt
                if flow_dir in final_state:
                    continue
                prev = attempts.get(flow_dir, int(att))
                # ensure attempts dict consistent
                attempts[flow_dir] = max(prev, int(att))
                # requeue or finalize fail
                ok_re = requeue(flow_dir, int(att), reason="task_fail")
                if not ok_re:
                    done_fail_final += 1
            elif tag == "FATAL_ILLEGAL":
                # worker will exit; inflight requeue also handled by death check
                # But we can requeue immediately too (safe, might duplicate once; duplicates are harmless due to DONE check)
                if flow_dir not in final_state:
                    requeue(flow_dir, int(att), reason="fatal_illegal")
            elif tag == "WORKER_DONE":
                # ignore
                pass

        # 3) check terminal condition
        terminal = len(final_state)
        if terminal >= total_tasks:
            break

        # 4) avoid busy loop
        if not drained:
            time.sleep(0.2)

    # stop workers: send sentinels
    for _ in procs.values():
        task_q.put(None)

    # join workers
    for gid, p in procs.items():
        try:
            p.join(timeout=30)
        except Exception:
            pass

    # summary
    print("\n" + "=" * 70, flush=True)
    print(f"[Done] total={total_tasks} ok={done_ok} fail_final={done_fail_final} (retries={args.task_retries})", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
