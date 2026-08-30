#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file).get("results", [])


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine semantic repeat non-embodied and embodied outputs into one results.json.")
    parser.add_argument("--repeat-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    repeat_dir = Path(args.repeat_dir)
    paths = [
        repeat_dir / "non_embodied" / "semantic_yesno_non_embodied_results.json",
        repeat_dir / "embodied" / "semantic_yesno_embodied_results.json",
    ]
    rows: list[dict] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        rows.extend(load_rows(path))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "experiment_info": {
                    "combined_at": datetime.now().isoformat(),
                    "source_files": [str(path) for path in paths],
                    "total_results": len(rows),
                },
                "results": rows,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Wrote {len(rows)} combined rows to {output_path}")


if __name__ == "__main__":
    main()
