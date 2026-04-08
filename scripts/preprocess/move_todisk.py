import os
import sys
import hashlib
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# ====================== 配置参数（请根据你的需求修改）======================
# 源目录（要移动的目录）
SOURCE_DIR = "/home/liuyang/Project/flow/flow_sequence"
# 目标目录（移动到的位置）
DEST_DIR = "/mnt/netdisk/usr/liuyang/flow/flow_sequence"
# 并发线程数（网络磁盘建议 8-16，根据服务器性能调整）
THREADS_NUM = 10
# 小文件阈值（小于该值的文件批量处理，单位：字节，1MB=1024*1024）
SMALL_FILE_THRESHOLD = 1024 * 1024 * 2  # 2MB
# 断点续传记录文件（自动生成，无需修改）
PROGRESS_FILE = "./move_progress.log"
# ==========================================================================

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_completed_files():
    """加载已传输完成的文件列表（断点续传）"""
    completed = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            completed = set(line.strip() for line in f if line.strip())
    return completed


def save_completed_file(file_path):
    """保存已完成的文件路径到进度文件"""
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(f"{file_path}\n")


def calculate_file_hash(file_path, hash_algorithm="sha256"):
    """计算文件的哈希值（用于完整性校验）"""
    hash_obj = hashlib.new(hash_algorithm)
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(4096 * 1024):  # 4MB 块读取，减少内存占用
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    except Exception as e:
        logger.error(f"计算文件 {file_path} 哈希失败：{e}")
        return None


def transfer_file(rel_path, source_root, dest_root, completed_files):
    """
    传输单个文件并校验完整性
    :param rel_path: 文件相对于源目录的路径
    :param source_root: 源目录根路径
    :param dest_root: 目标目录根路径
    :param completed_files: 已完成的文件集合
    """
    source_file = os.path.join(source_root, rel_path)
    dest_file = os.path.join(dest_root, rel_path)

    # 跳过已完成的文件
    if rel_path in completed_files:
        logger.debug(f"跳过已完成文件：{rel_path}")
        return True, rel_path, "已完成"

    # 创建目标目录
    dest_dir = os.path.dirname(dest_file)
    os.makedirs(dest_dir, exist_ok=True)

    try:
        # 传输文件（分块写入，适配大文件）
        with open(source_file, "rb") as f_src, open(dest_file, "wb") as f_dst:
            while chunk := f_src.read(4096 * 1024):
                f_dst.write(chunk)

        # 完整性校验
        src_hash = calculate_file_hash(source_file)
        dest_hash = calculate_file_hash(dest_file)

        if src_hash and dest_hash and src_hash == dest_hash:
            save_completed_file(rel_path)
            logger.info(f"传输并校验成功：{rel_path}")
            return True, rel_path, "成功"
        else:
            # 校验失败，删除目标文件
            if os.path.exists(dest_file):
                os.remove(dest_file)
            logger.error(f"文件 {rel_path} 校验失败：源哈希 {src_hash}，目标哈希 {dest_hash}")
            return False, rel_path, "校验失败"

    except Exception as e:
        # 传输失败，清理目标文件
        if os.path.exists(dest_file):
            os.remove(dest_file)
        logger.error(f"传输文件 {rel_path} 失败：{e}")
        return False, rel_path, f"传输失败：{e}"


def get_all_files(source_dir):
    """获取源目录下所有文件的相对路径（递归遍历）"""
    all_files = []
    source_dir = Path(source_dir)
    for file_path in source_dir.rglob("*"):
        if file_path.is_file():
            rel_path = str(file_path.relative_to(source_dir))
            all_files.append(rel_path)
    return all_files


def main():
    # 检查源目录是否存在
    if not os.path.isdir(SOURCE_DIR):
        logger.error(f"源目录不存在：{SOURCE_DIR}")
        sys.exit(1)

    # 创建目标目录
    os.makedirs(DEST_DIR, exist_ok=True)

    # 加载已完成的文件
    completed_files = load_completed_files()
    logger.info(f"已完成传输的文件数：{len(completed_files)}")

    # 获取所有需要传输的文件
    all_files = get_all_files(SOURCE_DIR)
    total_files = len(all_files)
    pending_files = [f for f in all_files if f not in completed_files]
    logger.info(f"待传输文件总数：{len(pending_files)}（总文件数：{total_files}）")

    if not pending_files:
        logger.info("所有文件已传输完成！")
        # 清理源目录
        logger.info("开始删除源目录...")
        shutil.rmtree(SOURCE_DIR, ignore_errors=True)
        logger.info("源目录删除完成！")
        return

    # 多线程传输文件
    success_count = 0
    fail_count = 0
    failed_files = []

    with ThreadPoolExecutor(max_workers=THREADS_NUM) as executor:
        # 提交任务
        futures = [
            executor.submit(transfer_file, rel_path, SOURCE_DIR, DEST_DIR, completed_files)
            for rel_path in pending_files
        ]

        # 处理结果
        for future in as_completed(futures):
            try:
                success, rel_path, msg = future.result()
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    failed_files.append((rel_path, msg))
                # 输出进度
                progress = (success_count + fail_count) / len(pending_files) * 100
                logger.info(
                    f"进度：{success_count + fail_count}/{len(pending_files)} ({progress:.1f}%) | 成功：{success_count} | 失败：{fail_count}")
            except Exception as e:
                logger.error(f"处理任务失败：{e}")
                fail_count += 1

    # 输出最终结果
    logger.info("=" * 50)
    logger.info(
        f"传输完成！总计：{total_files} | 已完成：{len(completed_files) + success_count} | 成功：{success_count} | 失败：{fail_count}")

    if failed_files:
        logger.error(f"失败文件列表：{failed_files}")
        logger.error("请检查失败文件，修复后重新运行程序即可断点续传！")
    else:
        logger.info("所有文件传输并校验成功！")
        # 校验通过后删除源目录
        logger.info("开始删除源目录...")
        shutil.rmtree(SOURCE_DIR, ignore_errors=True)
        logger.info("源目录删除完成！")
        # 删除进度文件
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
            logger.info("进度文件已清理！")


if __name__ == "__main__":
    main()