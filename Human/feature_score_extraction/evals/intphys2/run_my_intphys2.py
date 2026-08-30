from pathlib import Path

from evals.intphys2.eval import main

PROJECT_DIR = Path(__file__).resolve().parents[2]

args_eval = {
    "folder": str(PROJECT_DIR / "outputs"),
    "tag": "myvideos",
    "resume_checkpoint": False,

    "experiment": {
        "data": {
            "batch_size": 1,
            "resolution": 256,
            "stride_sliding_window": 2,
            "use_bfloat16": False,
            "frames_per_clip": 48,
            "context_lengths": [12, 18, 24, 30, 36, 42],
            "frame_steps": 10,
            "num_frames_to_pred": -1,
            "dataset": "intphys2-main"
        },
        "max_context_mode": True
    },

    "model_kwargs": {
        "checkpoint": str(PROJECT_DIR / "checkpoints" / "vith.pt"),
        "module_name": "app.vjepa.modelcustom.default_wrapper",
        "wrapper_kwargs": {
            "no_predictor": False
        },
        "pretrain_kwargs": {
            "resolution": 256,
            "predictor": {
                "model_name": "vit_predictor",
                "checkpoint_key": "predictor",
                "depth": 12
            },
            "encoder": {
                "model_name": "vit_huge",
                "checkpoint_key": "encoder",
                "is_causal": False,
                "local_window": [-1, -1, -1],
                "uniform_power": True,
                "use_activation_checkpointing": True,
                "use_mask_tokens": True,
                "use_rope": True,
                "zero_init_mask_tokens": True
            },
            "target_encoder": {
                "checkpoint_key": "target_encoder"
            }
        }
    }
}

if __name__ == "__main__":
    main(args_eval)
