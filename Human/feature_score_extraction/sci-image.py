import os
import cv2
import numpy as np
import pandas as pd

from skimage import feature, filters, morphology, util


def compute_frame_metrics_skimage(frame):
    # 1) 转灰度
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    area = h * w

    # 转成 uint8，方便 scikit-image 处理
    gray_u8 = util.img_as_ubyte(gray)

    # 2) 边缘密度：Canny
    edges = feature.canny(gray_u8, sigma=1.0)
    edge_density = np.count_nonzero(edges) / area

    # 3) 关键点密度：ORB
    orb = feature.ORB(n_keypoints=500, fast_threshold=0.08)
    try:
        orb.detect(gray_u8)
        keypoints = orb.keypoints
        num_keypoints = len(keypoints)
    except RuntimeError:
        num_keypoints = 0

    keypoint_density = num_keypoints / area

    # 4) 信息复杂度：局部熵
    entropy_img = filters.rank.entropy(gray_u8, morphology.disk(5))
    mean_entropy = float(np.mean(entropy_img))

    # 5) 合成每帧视觉复杂度
    visual_density = (
        0.4 * edge_density +
        0.3 * keypoint_density * 1000 +
        0.3 * (mean_entropy / 8.0)
    )

    return visual_density


def analyze_video(video_path, sample_every=10):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    visual_density_list = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % sample_every == 0:
            visual_density = compute_frame_metrics_skimage(frame)
            visual_density_list.append(visual_density)

        frame_id += 1

    cap.release()

    if len(visual_density_list) == 0:
        raise ValueError(f"没有成功抽取到任何帧，请检查视频文件: {video_path}")

    summary = {
        "video_name": os.path.basename(video_path),
        "video_path": video_path,
        "fps": fps,
        "total_frames": total_frames,
        "sample_every": sample_every,
        "sampled_frames": len(visual_density_list),
        "mean_visual_density": float(np.mean(visual_density_list)),
        "std_visual_density": float(np.std(visual_density_list, ddof=1)) if len(visual_density_list) > 1 else 0.0,
        "max_visual_density": float(np.max(visual_density_list)),
    }

    return summary


def get_video_files(folder_path):
    video_exts = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
    video_files = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in video_exts:
                video_files.append(file_path)

    video_files.sort()
    return video_files


if __name__ == "__main__":
    video_folder = "video"   # 读取 video 文件夹里所有视频
    sample_every = 10
    output_csv = "video_summary_sci_all.csv"

    if not os.path.exists(video_folder):
        raise ValueError(f"找不到文件夹: {video_folder}")

    video_files = get_video_files(video_folder)

    if len(video_files) == 0:
        raise ValueError(f"文件夹里没有找到视频文件: {video_folder}")

    all_results = []
    failed_videos = []

    print(f"共找到 {len(video_files)} 个视频，开始处理...\n")

    for i, video_path in enumerate(video_files, start=1):
        print(f"[{i}/{len(video_files)}] 正在处理: {os.path.basename(video_path)}")

        try:
            summary = analyze_video(video_path, sample_every=sample_every)
            all_results.append(summary)

            print(
                f"  平均复杂度={summary['mean_visual_density']:.6f}, "
                f"波动程度={summary['std_visual_density']:.6f}, "
                f"峰值复杂度={summary['max_visual_density']:.6f}"
            )
        except Exception as e:
            print(f"  处理失败: {e}")
            failed_videos.append({
                "video_path": video_path,
                "error": str(e)
            })

    if len(all_results) > 0:
        df_summary = pd.DataFrame(all_results)
        df_summary.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"\n所有成功结果已保存到: {output_csv}")
    else:
        print("\n没有任何视频处理成功。")

    if len(failed_videos) > 0:
        df_failed = pd.DataFrame(failed_videos)
        df_failed.to_csv("failed_videos_sci.csv", index=False, encoding="utf-8-sig")
        print("失败视频列表已保存到: failed_videos_sci.csv")

    print("\n处理完成。")