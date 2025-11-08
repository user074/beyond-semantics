#!/usr/bin/env python3

import os
import sys


def get_repo_root() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # evaluation/data/download_data_scripts -> go up 3 levels to repo root
    return os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))


def combine_jsonl_files(cv_bench_dir: str) -> None:
    test_2d = os.path.join(cv_bench_dir, "test_2d.jsonl")
    test_3d = os.path.join(cv_bench_dir, "test_3d.jsonl")
    out_path = os.path.join(cv_bench_dir, "test.jsonl")

    if os.path.exists(out_path):
        print(f"Already exists: {out_path}")
        return

    if not os.path.isfile(test_2d) or not os.path.isfile(test_3d):
        raise FileNotFoundError(
            f"Missing required files. Expected both: {test_2d} and {test_3d}"
        )

    os.makedirs(cv_bench_dir, exist_ok=True)
    total_lines = 0
    with open(out_path, "w") as out_f:
        for src in (test_2d, test_3d):
            with open(src, "r") as in_f:
                for line in in_f:
                    # Write raw lines to preserve JSONL formatting
                    out_f.write(line.rstrip("\n") + "\n")
                    total_lines += 1

    print(f"Wrote {total_lines} lines to {out_path}")


def main():
    # Optional arg: custom CV-Bench dir. Default to <repo>/Data/CV-Bench
    if len(sys.argv) > 1:
        cv_bench_dir = sys.argv[1]
    else:
        repo_root = get_repo_root()
        cv_bench_dir = os.path.join(repo_root, "Data", "CV-Bench")

    combine_jsonl_files(cv_bench_dir)


if __name__ == "__main__":
    main()


