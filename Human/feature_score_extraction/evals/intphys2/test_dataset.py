from pathlib import Path

from intphys2_dataset import IntPhys2Dataset

video_dir = Path(__file__).resolve().parent / "video"

ds = IntPhys2Dataset(
    data_path=video_dir,
    frame_step=10,
    transform=None
)

print("Number of videos:", len(ds))
print("First five paths:")
for p in ds.videopaths[:5]:
    print(p)

frames, idx = ds[0]
print("Shape of the first video:", frames.shape)
print("Index:", idx)
