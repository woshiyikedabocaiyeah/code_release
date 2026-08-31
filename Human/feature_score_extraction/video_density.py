import os
import cv2
import numpy as np
import pandas as pd


def compute_frame_metrics(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    area = h * w

    # 1. Edge density
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.count_nonzero(edges) / area

    # 2. Keypoint density
    orb = cv2.ORB_create(nfeatures=500)
    keypoints = orb.detect(gray, None)
    keypoint_density = len(keypoints) / area

    # 3. Texture complexity
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 4. Combine per-frame visual complexity metrics
    visual_density = (
        0.4 * edge_density +
        0.4 * keypoint_density * 1000 +
        0.2 * np.log1p(laplacian_var)
    )

    return visual_density


def analyze_video(video_path, sample_every=10):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    visual_density_list = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % sample_every == 0:
            visual_density = compute_frame_metrics(frame)
            visual_density_list.append(visual_density)

        frame_id += 1

    cap.release()

    if len(visual_density_list) == 0:
        raise ValueError(f"No frames were extracted successfully: {video_path}")

    summary = {
        "video_name": os.path.basename(video_path),
        "video_path": video_path,
        "fps": fps,
        "total_frames": total_frames,
        "sample_every": sample_every,
        "sampled_frames": len(visual_density_list),
        "mean_visual_density": float(np.mean(visual_density_list)),   # Mean complexity
        "std_visual_density": float(np.std(visual_density_list, ddof=1)) if len(visual_density_list) > 1 else 0.0,  # Variability
        "max_visual_density": float(np.max(visual_density_list)),     # Peak complexity
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
    video_folder = "video"   # your video directory
    sample_every = 10
    output_csv = "video_summary_all.csv"

    if not os.path.exists(video_folder):
        raise ValueError(f"Directory not found: {video_folder}")

    video_files = get_video_files(video_folder)

    if len(video_files) == 0:
        raise ValueError(f"No video files were found in the directory: {video_folder}")

    all_results = []
    failed_videos = []

    print(f"Found {len(video_files)} videos. Starting processing...\n")

    for i, video_path in enumerate(video_files, start=1):
        print(f"[{i}/{len(video_files)}] Processing: {os.path.basename(video_path)}")

        try:
            summary = analyze_video(video_path, sample_every=sample_every)
            all_results.append(summary)

            print(
                f"  Mean complexity={summary['mean_visual_density']:.6f}, "
                f"Variability={summary['std_visual_density']:.6f}, "
                f"Peak complexity={summary['max_visual_density']:.6f}"
            )
        except Exception as e:
            print(f"  Processing failed: {e}")
            failed_videos.append({
                "video_path": video_path,
                "error": str(e)
            })

    if len(all_results) > 0:
        df_summary = pd.DataFrame(all_results)
        df_summary.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"\nAll successful results were saved to: {output_csv}")
    else:
        print("\nNo videos were processed successfully.")

    if len(failed_videos) > 0:
        df_failed = pd.DataFrame(failed_videos)
        df_failed.to_csv("failed_videos.csv", index=False, encoding="utf-8-sig")
        print("The failed-video list was saved to: failed_videos.csv")

    print("\nProcessing complete.")
