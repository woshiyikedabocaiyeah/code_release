from pathlib import Path

from intphys2_dataset import IntPhys2Dataset

video_dir = Path(__file__).resolve().parent / "video"

ds = IntPhys2Dataset(
    data_path=video_dir,
    frame_step=10,
    transform=None
)

print("视频数量:", len(ds))
print("前5个路径:")
for p in ds.videopaths[:5]:
    print(p)

frames, idx = ds[0]
print("第一条视频 shape:", frames.shape)
print("索引:", idx)
