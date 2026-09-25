"""Small CPU-only check of profiler event accounting around the installed Adam."""

import argparse
import cProfile
import hashlib
import json
import pstats
from pathlib import Path

import torch

COUNT = 0


def one_fit():
    global COUNT
    COUNT += 1
    value = torch.ones(27, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([value], lr=.25)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        value.square().sum().backward()
        optimizer.step()


def outer():
    for _ in range(5):
        one_fit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    profiler = cProfile.Profile()
    profiler.runcall(outer)
    stats = pstats.Stats(profiler)
    rows = [dict(file=key[0], line=key[1], name=key[2], calls=value[1],
                 primitive_calls=value[0], self_seconds=value[2], cumulative_seconds=value[3])
            for key, value in stats.stats.items() if key[2] in ("outer", "one_fit")]
    report = dict(device="cpu", cuda_operations=False, actual_one_fit_count=COUNT,
                  expected_outer_calls=1, expected_one_fit_calls=5, recorded_functions=rows,
                  accounting_matches=(len(rows) == 2 and all(
                      row["calls"] == (1 if row["name"] == "outer" else 5) for row in rows)),
                  torch=torch.__version__, script_sha256=hashlib.sha256(
                      Path(__file__).read_bytes()).hexdigest())
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
