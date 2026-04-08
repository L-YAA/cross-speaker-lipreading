"""
=====================================================
Optical Flow: Predicting movement with the RAFT model
=====================================================

.. note::
    Try on `Colab <https://colab.research.google.com/github/pytorch/vision/blob/gh-pages/main/_generated_ipynb_notebooks/plot_optical_flow.ipynb>`_
    or :ref:`go to the end <sphx_glr_download_auto_examples_others_plot_optical_flow.py>` to download the full example code.

Optical flow is the task of predicting movement between two images, usually two
consecutive frames of a video. Optical flow models take two images as input, and
predict a flow: the flow indicates the displacement of every single pixel in the
first image, and maps it to its corresponding pixel in the second image. Flows
are (2, H, W)-dimensional tensors, where the first axis corresponds to the
predicted horizontal and vertical displacements.

The following example illustrates how torchvision can be used to predict flows
using our implementation of the RAFT model. We will also see how to convert the
predicted flows to RGB images for visualization.
"""

import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import torchvision.transforms.functional as F
#
#
# plt.rcParams["savefig.bbox"] = "tight"
# # sphinx_gallery_thumbnail_number = 2
#
#
# def plot(imgs, **imshow_kwargs):
#     if not isinstance(imgs[0], list):
#         # Make a 2d grid even if there's just 1 row
#         imgs = [imgs]
#
#     num_rows = len(imgs)
#     num_cols = len(imgs[0])
#     _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
#     for row_idx, row in enumerate(imgs):
#         for col_idx, img in enumerate(row):
#             ax = axs[row_idx, col_idx]
#             img = F.to_pil_image(img.to("cpu"))
#             ax.imshow(np.asarray(img), **imshow_kwargs)
#             ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
#
#     plt.tight_layout()
#
# # %%
# # Reading Videos Using Torchvision
# # --------------------------------
# # We will first read a video using :func:`~torchvision.io.read_video`.
# # Alternatively one can use the new :class:`~torchvision.io.VideoReader` API (if
# # torchvision is built from source).
# # The video we will use here is free of use from `pexels.com
# # <https://www.pexels.com/video/a-man-playing-a-game-of-basketball-5192157/>`_,
# # credits go to `Pavel Danilyuk <https://www.pexels.com/@pavel-danilyuk>`_.
#
#
# import tempfile
# from pathlib import Path
# from urllib.request import urlretrieve
#
#
# video_url = "https://download.pytorch.org/tutorial/pexelscom_pavel_danilyuk_basketball_hd.mp4"
# video_path = Path(tempfile.mkdtemp()) / "basketball.mp4"
# _ = urlretrieve(video_url, video_path)
#
# # %%
# # :func:`~torchvision.io.read_video` returns the video frames, audio frames and
# # the metadata associated with the video. In our case, we only need the video
# # frames.
# #
# # Here we will just make 2 predictions between 2 pre-selected pairs of frames,
# # namely frames (100, 101) and (150, 151). Each of these pairs corresponds to a
# # single model input.
#
# from torchvision.io import read_video
# frames, _, _ = read_video(str(video_path), output_format="TCHW")
#
# img1_batch = torch.stack([frames[100], frames[150]])
# img2_batch = torch.stack([frames[101], frames[151]])
#
# plot(img1_batch)
#
# # %%
# # The RAFT model accepts RGB images. We first get the frames from
# # :func:`~torchvision.io.read_video` and resize them to ensure their dimensions
# # are divisible by 8. Note that we explicitly use ``antialias=False``, because
# # this is how those models were trained. Then we use the transforms bundled into
# # the weights in order to preprocess the input and rescale its values to the
# # required ``[-1, 1]`` interval.
#
# from torchvision.models.optical_flow import Raft_Large_Weights
#
# weights = Raft_Large_Weights.DEFAULT
# transforms = weights.transforms()
#
#
# def preprocess(img1_batch, img2_batch):
#     img1_batch = F.resize(img1_batch, size=[520, 960], antialias=False)
#     img2_batch = F.resize(img2_batch, size=[520, 960], antialias=False)
#     return transforms(img1_batch, img2_batch)
#
#
#
# img1_batch, img2_batch = preprocess(img1_batch, img2_batch)
#
# print(f"shape = {img1_batch.shape}, dtype = {img1_batch.dtype}")
#
#
# # %%
# # Estimating Optical flow using RAFT
# # ----------------------------------
# # We will use our RAFT implementation from
# # :func:`~torchvision.models.optical_flow.raft_large`, which follows the same
# # architecture as the one described in the `original paper <https://arxiv.org/abs/2003.12039>`_.
# # We also provide the :func:`~torchvision.models.optical_flow.raft_small` model
# # builder, which is smaller and faster to run, sacrificing a bit of accuracy.
#
# from torchvision.models.optical_flow import raft_large
#
# # If you can, run this example on a GPU, it will be a lot faster.
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
# model = model.eval()
#
# list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
# print(f"type = {type(list_of_flows)}")
# print(f"length = {len(list_of_flows)} = number of iterations of the model")
#
# # %%
# # The RAFT model outputs lists of predicted flows where each entry is a
# # (N, 2, H, W) batch of predicted flows that corresponds to a given "iteration"
# # in the model. For more details on the iterative nature of the model, please
# # refer to the `original paper <https://arxiv.org/abs/2003.12039>`_. Here, we
# # are only interested in the final predicted flows (they are the most accurate
# # ones), so we will just retrieve the last item in the list.
# #
# # As described above, a flow is a tensor with dimensions (2, H, W) (or (N, 2, H,
# # W) for batches of flows) where each entry corresponds to the horizontal and
# # vertical displacement of each pixel from the first image to the second image.
# # Note that the predicted flows are in "pixel" unit, they are not normalized
# # w.r.t. the dimensions of the images.
# predicted_flows = list_of_flows[-1]
# print(f"dtype = {predicted_flows.dtype}")
# print(f"shape = {predicted_flows.shape} = (N, 2, H, W)")
# print(f"min = {predicted_flows.min()}, max = {predicted_flows.max()}")
#
#
# # %%
# # Visualizing predicted flows
# # ---------------------------
# # Torchvision provides the :func:`~torchvision.utils.flow_to_image` utility to
# # convert a flow into an RGB image. It also supports batches of flows.
# # each "direction" in the flow will be mapped to a given RGB color. In the
# # images below, pixels with similar colors are assumed by the model to be moving
# # in similar directions. The model is properly able to predict the movement of
# # the ball and the player. Note in particular the different predicted direction
# # of the ball in the first image (going to the left) and in the second image
# # (going up).
#
# from torchvision.utils import flow_to_image
#
# flow_imgs = flow_to_image(predicted_flows)
#
# # The images have been mapped into [-1, 1] but for plotting we want them in [0, 1]
# img1_batch = [(img1 + 1) / 2 for img1 in img1_batch]
#
# grid = [[img1, flow_img] for (img1, flow_img) in zip(img1_batch, flow_imgs)]
# plot(grid)
# plt.show()
#
# from torchvision.io import write_jpeg
# import os
#
# # 创建保存光流图的文件夹
# output_dir = "/home/ly/flow/flow_frames"
# os.makedirs(output_dir, exist_ok=True)
#
# # 遍历视频所有连续帧（逐帧推理，也可按batch_size=8/16批量推理）
# for i in range(len(frames) - 1):
#     # 取第i和i+1帧
#     img1 = frames[i].unsqueeze(0)  # 增加batch维度 (1, 3, H, W)
#     img2 = frames[i + 1].unsqueeze(0)
#
#     # 预处理
#     img1, img2 = preprocess(img1, img2)
#     img1, img2 = preprocess(img1, img2)
#
#     # 模型推理（评估模式，禁用梯度）
#     with torch.no_grad():
#         list_of_flows = model(img1.to(device), img2.to(device))
#         predicted_flow = list_of_flows[-1][0]  # 取最后一次迭代的光流（去掉batch维度）
#
#     # 转换为可视化图像并保存
#     flow_img = flow_to_image(predicted_flow).to("cpu")
#     write_jpeg(flow_img, f"{output_dir}/flow_{i}.jpg")
# %%
# Bonus: Creating GIFs of predicted flows
# ---------------------------------------
# In the example above we have only shown the predicted flows of 2 pairs of
# frames. A fun way to apply the Optical Flow models is to run the model on an
# entire video, and create a new video from all the predicted flows. Below is a
# snippet that can get you started with this. We comment out the code, because
# this example is being rendered on a machine without a GPU, and it would take
# too long to run it.

# from torchvision.io import write_jpeg
# for i, (img1, img2) in enumerate(zip(frames, frames[1:])):
#     # Note: it would be faster to predict batches of flows instead of individual flows
#     img1, img2 = preprocess(img1, img2)

#     list_of_flows = model(img1.to(device), img2.to(device))
#     predicted_flow = list_of_flows[-1][0]
#     flow_img = flow_to_image(predicted_flow).to("cpu")
#     output_folder = "/tmp/"  # Update this to the folder of your choice
#     write_jpeg(flow_img, output_folder + f"predicted_flow_{i}.jpg")

# %%
# Once the .jpg flow images are saved, you can convert them into a video or a
# GIF using ffmpeg with e.g.:
#
# ffmpeg -f image2 -framerate 30 -i predicted_flow_%d.jpg -loop -1 flow.gif



#cmlr 生成光流视频
import os
import numpy as np
import torch
import subprocess
import torchvision.transforms.functional as F
from tqdm import tqdm
from pathlib import Path
from torchvision.io import read_video, write_jpeg
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.utils import flow_to_image

# ======================== 1. 核心配置（按需调整） ========================
# 原始视频根目录（嵌套结构：sxx/日期/xxx.mp4）
ORIG_VIDEO_ROOT = "/mnt/dataset/CMLR/cmlr/cmlr_video_seg24s/"
# 光流视频输出根目录（复刻原视频嵌套结构）
FLOW_VIDEO_ROOT = "/home/ly/flow/flow_video/cmlr"
# 中间光流帧保存目录（默认在视频输出目录下的frames子文件夹）
KEEP_FLOW_FRAMES = True  # True=保留帧，False=合成视频后删除帧
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4  # GPU显存不足则减小（如2/1），CPU建议1
TARGET_SIZE = (520, 960)  # 光流模型输入尺寸（需为8的倍数）
SKIP_FRAMES = 1  # 1=处理所有连续帧，>1可跳过（如2=每2帧取一对）
VIDEO_FPS = 30  # 光流视频帧率（设为0则自动匹配原视频帧率）
VIDEO_CODEC = "libx264"  # 视频编码（H.264兼容性最好）
PIX_FMT = "yuv420p"  # 像素格式（避免播放器黑屏）


# ======================== 2. 模型与预处理初始化 ========================
def init_raft_model():
    """初始化RAFT光流模型（评估模式）"""
    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()
    model = raft_large(weights=weights, progress=False).to(DEVICE)
    model.eval()
    return model, transforms


# 初始化模型（全局仅初始化一次，避免重复加载）
model, preprocess_transform = init_raft_model()


def preprocess_frames(img1_batch, img2_batch, target_size):
    """预处理帧对：调整尺寸 + 归一化到[-1,1]"""
    img1_batch = F.resize(img1_batch, size=target_size, antialias=False)
    img2_batch = F.resize(img2_batch, size=target_size, antialias=False)
    return preprocess_transform(img1_batch, img2_batch)


# ======================== 3. 路径工具函数 ========================
def get_all_mp4_files(root_dir):
    """递归遍历根目录，返回所有mp4视频的绝对路径列表"""
    mp4_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".mp4"):
                mp4_files.append(os.path.join(root, file))
    print(f"✅ 共找到 {len(mp4_files)} 个原始mp4视频")
    return mp4_files


def get_output_paths(orig_video_path, flow_video_root):
    """
    生成光流帧目录和光流视频路径（复刻原视频嵌套结构）
    示例：
    原视频：/mnt/.../s11/20180529/xxx.mp4
    光流帧目录：/home/ly/flow/flow_video/cmlr/s11/20180529/xxx_frames/
    光流视频：/home/ly/flow/flow_video/cmlr/s11/20180529/xxx.mp4
    """
    # 提取原视频相对路径（如s11/20180529/xxx.mp4）
    rel_path = os.path.relpath(orig_video_path, ORIG_VIDEO_ROOT)
    # 拆分目录和文件名（如s11/20180529/ + xxx.mp4）
    rel_dir, video_name = os.path.split(rel_path)
    video_name_no_ext = Path(video_name).stem  # 去掉.mp4后缀

    # 光流视频路径
    flow_video_path = os.path.join(flow_video_root, rel_dir, f"{video_name_no_ext}.mp4")
    # 光流帧目录
    flow_frame_dir = os.path.join(flow_video_root, rel_dir, f"{video_name_no_ext}_frames")

    # 创建目录（自动创建嵌套结构）
    os.makedirs(os.path.dirname(flow_video_path), exist_ok=True)
    os.makedirs(flow_frame_dir, exist_ok=True)

    return flow_frame_dir, flow_video_path


def get_video_fps(video_path):
    """读取原视频的实际帧率（失败则返回默认值）"""
    if VIDEO_FPS == 0:
        try:
            cmd = [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            fps_str = subprocess.check_output(cmd).decode("utf-8").strip()
            num, den = fps_str.split('/')
            return int(int(num) / int(den))
        except:
            print(f"⚠️  读取 {video_path} 帧率失败，使用默认帧率30")
            return 30
    return VIDEO_FPS


# ======================== 4. 光流帧生成函数 ========================
def generate_flow_frames(orig_video_path, flow_frame_dir):
    """读取原始视频，生成光流帧并保存到指定目录"""
    # 1. 读取原始视频帧
    try:
        frames, _, _ = read_video(str(orig_video_path), output_format="TCHW", pts_unit="sec")
        if frames.shape[0] < 2:
            print(f"⚠️  {orig_video_path} 帧数不足2，跳过")
            return False
    except Exception as e:
        print(f"❌ 读取 {orig_video_path} 失败：{e}")
        return False

    total_frames = frames.shape[0]
    frame_indices = list(range(0, total_frames - 1, SKIP_FRAMES))
    total_batches = (len(frame_indices) + BATCH_SIZE - 1) // BATCH_SIZE

    # 2. 批量生成光流帧
    with tqdm(total=total_batches, desc=f"生成光流帧 {Path(orig_video_path).stem}") as pbar:
        for batch_idx in range(total_batches):
            # 取当前批次帧索引
            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(frame_indices))
            batch_indices = frame_indices[start:end]

            # 构建帧对批量
            img1_batch = frames[batch_indices]
            img2_batch = frames[np.array(batch_indices) + 1]

            # 预处理
            img1_batch, img2_batch = preprocess_frames(img1_batch, img2_batch, TARGET_SIZE)

            # 模型推理（禁用梯度，节省内存）
            with torch.no_grad():
                flow_list = model(img1_batch.to(DEVICE), img2_batch.to(DEVICE))
                pred_flows = flow_list[-1]  # 取最后一次迭代的光流（精度最高）
                flow_imgs = flow_to_image(pred_flows)  # 转为RGB可视化帧

            # 保存光流帧（6位数字命名，适配ffmpeg）
            for idx, flow_img in zip(batch_indices, flow_imgs):
                frame_path = os.path.join(flow_frame_dir, f"flow_{idx:06d}.jpg")
                write_jpeg(flow_img.to("cpu"), frame_path)

            # 清理GPU缓存（避免显存泄漏）
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            pbar.update(1)

    return True


# ======================== 5. 光流帧合成视频函数 ========================
def convert_frames_to_video(flow_frame_dir, flow_video_path):
    """调用ffmpeg将光流帧合成视频"""
    fps = get_video_fps(flow_video_path.replace("_frames", "").replace(".mp4", "") + ".mp4")
    # ffmpeg命令（覆盖已有文件，仅输出错误信息）
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(flow_frame_dir, "flow_%06d.jpg"),
        "-c:v", VIDEO_CODEC,
        "-pix_fmt", PIX_FMT,
        "-hide_banner", "-loglevel", "error",
        flow_video_path
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✅ 合成光流视频：{flow_video_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 合成 {flow_video_path} 失败：{e.stderr.decode('utf-8')}")
        return False
    except Exception as e:
        print(f"❌ 合成 {flow_video_path} 失败：{str(e)}")
        return False


def delete_flow_frames(flow_frame_dir):
    """删除光流帧目录（清理中间文件）"""
    try:
        for root, _, files in os.walk(flow_frame_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            os.rmdir(root)
        print(f"🗑️  已删除中间光流帧：{flow_frame_dir}")
    except Exception as e:
        print(f"⚠️  删除 {flow_frame_dir} 失败：{e}")


# ======================== 6. 主流程：原始视频→光流视频 ========================
def process_single_video(orig_video_path):
    """处理单个原始视频：生成光流帧→合成光流视频→可选删除帧"""
    # 1. 生成输出路径
    flow_frame_dir, flow_video_path = get_output_paths(orig_video_path, FLOW_VIDEO_ROOT)

    # 2. 生成光流帧
    if not generate_flow_frames(orig_video_path, flow_frame_dir):
        return False

    # 3. 合成光流视频
    if not convert_frames_to_video(flow_frame_dir, flow_video_path):
        return False

    # 4. 可选删除中间帧
    if not KEEP_FLOW_FRAMES:
        delete_flow_frames(flow_frame_dir)

    return True


def batch_process_all_videos():
    """批量处理所有原始视频"""
    # 检查原始视频目录
    if not os.path.exists(ORIG_VIDEO_ROOT):
        print(f"❌ 原始视频根目录不存在：{ORIG_VIDEO_ROOT}")
        return

    # 获取所有mp4视频
    all_mp4 = get_all_mp4_files(ORIG_VIDEO_ROOT)
    if not all_mp4:
        print("❌ 未找到任何mp4视频文件")
        return

    # 逐个处理视频
    success_count = 0
    fail_count = 0
    for video_path in all_mp4:
        if process_single_video(video_path):
            success_count += 1
        else:
            fail_count += 1

    # 输出统计结果
    print("\n" + "=" * 50)
    print(f"处理完成！成功：{success_count} 个，失败：{fail_count} 个")
    print(f"光流视频根目录：{FLOW_VIDEO_ROOT}")
    if KEEP_FLOW_FRAMES:
        print(f"光流帧保留路径：{FLOW_VIDEO_ROOT}/*_frames/")


# ======================== 7. 执行主流程 ========================
if __name__ == "__main__":
    print(f"🚀 开始处理（设备：{DEVICE}）")
    batch_process_all_videos()