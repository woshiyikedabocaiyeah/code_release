from pathlib import Path

from eval import main

PROJECT_DIR = Path(__file__).resolve().parent

args_eval = {
    "folder": str(PROJECT_DIR / "outputs"),
    "tag": "myvideos",
    "resume_checkpoint": False,

    "model_kwargs": {
        "checkpoint": str(PROJECT_DIR / "checkpoints" / "checkpoint.pth"),
        "module_name": "PATH.TO.YOUR.MODULE",
        "pretrain_kwargs": {
            # 从官方现成配置里复制
        },
        "wrapper_kwargs": {
            # 从官方现成配置里复制
        },
    },

    "experiment": {
        "mode": "losses",
        "max_context_mode": True,
        "data": {
            "resolution": 224,
            "batch_size": 1,
            "stride_sliding_window": 2,
            "use_bfloat16": False,
            "frames_per_clip": 16,
            "context_lengths": [4],
            "num_frames_to_pred": -1,
            "frame_steps": [1],
            "dataset": "intphys2-main"
        }
    }
}

main(args_eval)
