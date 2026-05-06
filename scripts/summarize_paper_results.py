from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
RAW_RUNS = {
    "s42": "20260505_130308",
    "s123": "20260505_163304",
    "s7": "20260505_200221",
    "s999": "20260505_234842",
    "s2048": "20260506_032147",
}


def read_auc_by_variant(path: Path) -> dict[tuple[str, str], float]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    result: dict[tuple[str, str], float] = {}
    for row in rows:
        result[(row["split"], row["variant"])] = float(row["auc"])
    return result


def main() -> None:
    rows = []
    for seed, timestamp in RAW_RUNS.items():
        output = EXPERIMENTS / timestamp / "output"
        angle = read_auc_by_variant(output / "angle_length_metrics.csv")
        length = read_auc_by_variant(output / "length_only_metrics.csv")
        for split in ["ffpp", "cdf", "dfd"]:
            rows.append({
                "dataset": split.upper(),
                "seed": seed,
                "length": length[(split, "feature_length")],
                "angle": angle[(split, "feature_norm")],
                "raw": angle[(split, "raw")],
            })

    print("dataset,seed,length,angle,raw")
    for row in rows:
        print(f"{row['dataset']},{row['seed']},{row['length']:.4f},{row['angle']:.4f},{row['raw']:.4f}")

    print("\nmean_auc")
    print("dataset,length,angle,raw")
    for dataset in ["FFPP", "CDF", "DFD"]:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        print(
            f"{dataset},"
            f"{mean(row['length'] for row in dataset_rows):.4f},"
            f"{mean(row['angle'] for row in dataset_rows):.4f},"
            f"{mean(row['raw'] for row in dataset_rows):.4f}"
        )


if __name__ == "__main__":
    main()
