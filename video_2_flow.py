import os
import re
import json
import time
import numpy as np

# ======================== 0. GPU设备配置（多卡，必须在import torch前） ========================
GPU_IDS = [7]  # 物理GPU编号（nvidia-smi看到的编号）
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, GPU_IDS))

import torch
import subprocess
import tempfile
import shutil
import torchvision.transforms.functional as F
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.utils import flow_to_image
from torchvision.io import write_jpeg

# ======================== 1. 核心配置 ========================
ORIG_VIDEO_ROOT = "/mnt/netdisk/dataset/lipreading/mpc/preprocess_datasets/CMLR_extra_latest/cmlr_extra/cmlr_extra_video_seg24s/"
FLOW_FRAMES_ROOT = "/home/liuyang/Project/flow/flow_sequence/cmlr_extra/"

BATCH_SIZE = 48
TARGET_SIZE = (520, 960)   # 需为8的倍数
SKIP_FRAMES = 1            # 必须是1，连续帧
RESUME = True              # 断点续跑
RECHECK_BACK = 128         # resume时向后回看多少帧找“第一个缺失帧”（越大越稳，越小越快）

# 不再使用 ffprobe 的 count_frames（慢）
FAST_SKIP_BY_FFPROBE = False

FLOW_NAME_RE = re.compile(r"^flow_(\d{6})\.jpg$")
PROGRESS_FILE = "progress.json"
DONE_FILE = "DONE"


# ======================== 2. GPU/设备初始化 ========================
def setup_device():
    print(f"[GPU] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"[GPU] torch.version.cuda={torch.version.cuda}", flush=True)

    if not torch.cuda.is_available():
        print("❌ torch.cuda.is_available() = False，将使用CPU。", flush=True)
        return torch.device("cpu"), 0

    n = torch.cuda.device_count()
    print(f"✅ CUDA可用，可见GPU数量：{n}", flush=True)
    for i in range(n):
        print(f"   - cuda:{i} = {torch.cuda.get_device_name(i)}", flush=True)

    return torch.device("cuda:0"), n


DEVICE, NUM_GPUS = setup_device()


# ======================== 3. RAFT模型初始化（支持多GPU） ========================
def init_raft_model(device, num_gpus):
    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()

    model = raft_large(weights=weights, progress=False)

    if device.type == "cuda" and num_gpus >= 2:
        print(f"✅ 启用 DataParallel，多卡数量={num_gpus}（程序内GPU编号 0..{num_gpus-1}）", flush=True)
        model = torch.nn.DataParallel(model)

    model = model.to(device)
    model.eval()
    return model, transforms


model, preprocess_transform = init_raft_model(DEVICE, NUM_GPUS)


def preprocess_frames(img1_batch, img2_batch, target_size):
    img1_batch = F.resize(img1_batch, size=target_size, antialias=False)
    img2_batch = F.resize(img2_batch, size=target_size, antialias=False)
    return preprocess_transform(img1_batch, img2_batch)


# ======================== 4. 路径工具 ========================
def get_all_mp4_files(root_dir):
    mp4_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".mp4"):
                mp4_files.append(os.path.join(root, file))
    mp4_files.sort()
    print(f"✅ 共找到 {len(mp4_files)} 个原始mp4视频", flush=True)
    return mp4_files


def get_flow_frames_dir(orig_video_path, flow_frames_root):
    rel_path = os.path.relpath(orig_video_path, ORIG_VIDEO_ROOT)
    rel_dir, video_name = os.path.split(rel_path)
    video_name_no_ext = Path(video_name).stem
    flow_frames_dir = os.path.join(flow_frames_root, rel_dir, video_name_no_ext)
    os.makedirs(flow_frames_dir, exist_ok=True)
    return flow_frames_dir


def flow_path(flow_dir, idx: int) -> str:
    return os.path.join(flow_dir, f"flow_{idx:06d}.jpg")


def progress_path(flow_dir) -> str:
    return os.path.join(flow_dir, PROGRESS_FILE)


def done_path(flow_dir) -> str:
    return os.path.join(flow_dir, DONE_FILE)


# ======================== 5. 进度文件（快速断点续跑关键） ========================
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


def load_progress(flow_dir: str):
    p = progress_path(flow_dir)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def quick_max_existing_idx(flow_dir: str) -> int:
    """
    只在“没有progress.json”的老目录里用一次：扫描目录找到最大的flow_xxxxxx.jpg索引。
    这是一次性的，后续都走 progress.json，非常快。
    """
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


def get_resume_start_idx(flow_dir: str) -> int:
    """
    断点续跑：优先读 progress.json 的 next_idx
    为了防止上次崩在写文件中间，回看 RECHECK_BACK 帧找到“第一个缺失idx”
    """
    # 如果存在 DONE 标记，直接认为完成
    if os.path.exists(done_path(flow_dir)):
        return -1  # 表示已完成

    # 清理残留 tmp（如果有）
    # 防止上次崩在写 tmp 的途中，下次误判“已存在”
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

    prog = load_progress(flow_dir)
    if prog and "next_idx" in prog:
        next_idx = max(0, int(prog["next_idx"]))
    else:
        # 老目录：没有progress.json，退化为扫描一次最大idx
        max_idx = quick_max_existing_idx(flow_dir)
        next_idx = max_idx + 1

    # 回看找第一个缺失（快：最多检查 RECHECK_BACK 次）
    start = max(0, next_idx - RECHECK_BACK)
    i = start
    while i < next_idx:
        if not os.path.exists(flow_path(flow_dir, i)):
            return i
        i += 1
    return next_idx


# ======================== 6. 原子写JPEG（避免半截坏文件） ========================
def atomic_write_jpeg_tensor(flow_img_chw_uint8, dst_path: str):
    tmp = dst_path + ".tmp"
    write_jpeg(flow_img_chw_uint8.to("cpu"), tmp)
    os.replace(tmp, dst_path)


# ======================== 7. ffmpeg读取视频帧（支持从指定起始帧开始导出） ========================
def read_video_frames_with_ffmpeg(video_path: str, start_frame: int = 0):
    """
    用ffmpeg解码出帧到临时目录，再读回为 TCHW Tensor
    start_frame>0 时只导出从该帧开始的帧，减少恢复时的临时数据量/内存压力
    注意：ffmpeg可能仍需解码到该位置（编码原因），但至少不会把前面所有帧写出来/读进内存
    """
    temp_dir = tempfile.mkdtemp(prefix="ffmpeg_frames_")
    try:
        ffmpeg_cmd = ["ffmpeg", "-i", video_path]

        if start_frame > 0:
            # 选择从第start_frame帧开始输出
            # n 从0开始计数；逗号需要转义为 \,
            vf = f"select='gte(n\\,{start_frame})'"
            ffmpeg_cmd += ["-vf", vf, "-vsync", "0"]

        ffmpeg_cmd += [
            "-q:v", "1",
            "-f", "image2",
            "-hide_banner", "-loglevel", "error",
            os.path.join(temp_dir, "frame_%06d.jpg"),
        ]

        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        frame_files = sorted([f for f in os.listdir(temp_dir) if f.startswith("frame_") and f.endswith(".jpg")])
        if len(frame_files) < 2:
            return None

        frames = []
        for f in frame_files:
            fp = os.path.join(temp_dir, f)
            with Image.open(fp) as img:
                img = img.convert("RGB")
                frames.append(F.to_tensor(img))

        return torch.stack(frames)  # (T,3,H,W)

    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", errors="ignore") if e.stderr else str(e)
        print(f"❌ ffmpeg读取失败：{video_path}\n{err}", flush=True)
        return None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ======================== 8. 核心：生成光流帧（快速断点续跑） ========================
def generate_flow_frames(orig_video_path: str):
    flow_dir = get_flow_frames_dir(orig_video_path, FLOW_FRAMES_ROOT)

    if RESUME:
        start_idx = get_resume_start_idx(flow_dir)
        if start_idx < 0:
            print(f"⏭️  DONE已存在，跳过：{flow_dir}", flush=True)
            return True
    else:
        start_idx = 0

    # 为了稳妥：如果 start_idx 位置文件存在，则顺延到第一个不存在的位置（避免重复）
    while RESUME and os.path.exists(flow_path(flow_dir, start_idx)):
        start_idx += 1

    print(f"▶️  {Path(orig_video_path).name} 从 idx={start_idx} 继续（dir={flow_dir}）", flush=True)
    save_progress(flow_dir, start_idx, orig_video_path)

    # 只从 start_idx 开始读取帧，减少恢复时内存与临时文件
    print(f"🎞️  ffmpeg读取帧（start_frame={start_idx}）：{orig_video_path}", flush=True)
    frames = read_video_frames_with_ffmpeg(orig_video_path, start_frame=start_idx)
    if frames is None:
        print(f"⚠️  帧数不足或读取失败，跳过：{orig_video_path}", flush=True)
        return False

    # frames[0] 对应原始帧 start_idx
    remain_frames = frames.shape[0]
    remain_pairs = remain_frames - 1
    if remain_pairs <= 0:
        # 没有可生成的帧对，认为完成
        atomic_write_text(done_path(flow_dir), "done\n")
        return True

    # 仅对“缺失的flow文件”做推理（逐个exists检查，比扫描全目录快很多）
    missing_local = []
    for local_idx in range(remain_pairs):
        orig_idx = start_idx + local_idx
        if not (RESUME and os.path.exists(flow_path(flow_dir, orig_idx))):
            missing_local.append(local_idx)

    if not missing_local:
        # 从 start_idx 开始后面都已经存在：直接DONE
        atomic_write_text(done_path(flow_dir), "done\n")
        save_progress(flow_dir, start_idx + remain_pairs, orig_video_path)
        print(f"✅ 已补齐/已存在：{flow_dir}", flush=True)
        return True

    total_batches = (len(missing_local) + BATCH_SIZE - 1) // BATCH_SIZE
    print(
        f"📌 待补帧对：{len(missing_local)}（从idx={start_idx}起）| remain_pairs={remain_pairs} | batches={total_batches}",
        flush=True
    )

    with tqdm(total=total_batches, desc=f"Flow {Path(orig_video_path).stem}") as pbar:
        for b in range(total_batches):
            s = b * BATCH_SIZE
            e = min(s + BATCH_SIZE, len(missing_local))
            batch_local = missing_local[s:e]

            # 构建帧对（local）
            img1_batch = frames[batch_local]
            img2_batch = frames[np.array(batch_local) + 1]

            img1_batch, img2_batch = preprocess_frames(img1_batch, img2_batch, TARGET_SIZE)

            with torch.no_grad():
                flow_list = model(img1_batch.to(DEVICE), img2_batch.to(DEVICE))
                pred_flows = flow_list[-1]
                flow_imgs = flow_to_image(pred_flows)  # uint8 (B,3,H,W)

            # 保存 + 更新progress（按当前批次推进）
            max_orig_in_batch = None
            for local_idx, flow_img in zip(batch_local, flow_imgs):
                orig_idx = start_idx + int(local_idx)
                out_path = flow_path(flow_dir, orig_idx)

                # 断点续跑双保险
                if RESUME and os.path.exists(out_path):
                    max_orig_in_batch = orig_idx
                    continue

                # 原子写
                atomic_write_jpeg_tensor(flow_img, out_path)
                max_orig_in_batch = orig_idx

            # 每个batch更新一次进度：下次直接从这里继续
            if max_orig_in_batch is not None:
                save_progress(flow_dir, max_orig_in_batch + 1, orig_video_path)

            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

            pbar.update(1)

    # 处理完 start_idx 之后的部分，如果理论上已经到末尾，则写DONE
    # 这里我们把“从start_idx到视频末尾”补齐就认为完成（DONE用于快速跳过）
    atomic_write_text(done_path(flow_dir), "done\n")
    save_progress(flow_dir, start_idx + remain_pairs, orig_video_path)
    print(f"✅ 完成：{flow_dir}", flush=True)
    return True


# ======================== 9. 批处理所有视频 ========================
def batch_process_all_videos():
    if not os.path.exists(ORIG_VIDEO_ROOT):
        print(f"❌ 原始视频根目录不存在：{ORIG_VIDEO_ROOT}", flush=True)
        return

    all_mp4_files = get_all_mp4_files(ORIG_VIDEO_ROOT)
    if not all_mp4_files:
        print("❌ 未找到任何mp4视频文件", flush=True)
        return

    success_count = 0
    fail_count = 0

    for video_path in all_mp4_files:
        try:
            ok = generate_flow_frames(video_path)
        except Exception as e:
            print(f"❌ 处理异常：{video_path}\n{repr(e)}", flush=True)
            ok = False

        if ok:
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 60, flush=True)
    print(f"📊 批量处理完成 | 成功：{success_count} | 失败：{fail_count}", flush=True)
    print(f"📁 光流帧根目录：{FLOW_FRAMES_ROOT}", flush=True)
    print("=" * 60, flush=True)


# ======================== 10. 执行入口 ========================
if __name__ == "__main__":
    print(f"🚀 开始生成光流帧 | DEVICE={DEVICE} | 可见GPU数={NUM_GPUS}", flush=True)
    print(f"📌 原始视频根目录：{ORIG_VIDEO_ROOT}", flush=True)
    print(f"📌 光流帧输出根目录：{FLOW_FRAMES_ROOT}", flush=True)
    print(f"📌 RESUME={RESUME} | RECHECK_BACK={RECHECK_BACK}", flush=True)
    batch_process_all_videos()
