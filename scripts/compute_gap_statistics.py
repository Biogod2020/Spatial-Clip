#!/usr/bin/env python
"""Compute nearest-neighbor gap statistics directly from raw HEST AnnData files."""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

import rootutils

PROJECT_ROOT = Path(rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True))

DEFAULT_METADATA_COLUMNS: Sequence[str] = (
    "st_technology",
    "organ",
    "tissue",
    "species",
    "disease_state",
    "preservation_method",
    "spots_under_tissue",
    "inter_spot_dist",
    "spot_diameter",
    "pixel_size_um_estimated",
    "pixel_size_um_embedded",
    "fullres_px_width",
    "fullres_px_height",
)

from src.data.preprocessing.utils import get_spot_coordinates


@dataclass(frozen=True)
class SampleTask:
    sample_id: str
    st_path: Path
    metadata: Dict[str, float | str | int | None]

    def exists(self) -> bool:
        return self.st_path.exists()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=PROJECT_ROOT / "data/raw/hest_v1",
        help="Root folder containing metadata CSV plus 'st/' H5AD files.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional explicit path to the metadata CSV (defaults to <raw>/HEST_v1_1_0.csv).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "outputs/gap_report",
        help="Output directory for CSV/JSON summaries.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of parallel workers (per sample).",
    )
    parser.add_argument(
        "--executor",
        choices=("process", "thread"),
        default="process",
        help="Executor backend to use for per-sample processing.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional cap on the number of samples (for smoke tests).",
    )
    parser.add_argument(
        "--sample-ids",
        type=str,
        default=None,
        help="Comma-separated whitelist of sample IDs to process.",
    )
    parser.add_argument(
        "--patch-size",
        type=float,
        default=224.0,
        help="Patch edge length in the same pixel units as the AnnData coordinates.",
    )
    parser.add_argument(
        "--metadata-columns",
        type=str,
        default=",".join(DEFAULT_METADATA_COLUMNS),
        help="Comma-separated metadata columns to copy into the per-sample CSV.",
    )
    return parser.parse_args()


def load_metadata(metadata_csv: Path) -> pd.DataFrame:
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found at {metadata_csv}")
    df = pd.read_csv(metadata_csv)
    if "id" not in df.columns:
        raise ValueError("Metadata CSV must contain an 'id' column.")
    df["id"] = df["id"].astype(str)
    return df


def build_tasks(raw_root: Path, meta_df: pd.DataFrame, include_ids: Optional[set[str]] = None,
                sample_limit: Optional[int] = None) -> List[SampleTask]:
    st_dir = raw_root / "st"
    if not st_dir.exists():
        raise FileNotFoundError(f"Expected AnnData directory at {st_dir}")

    tasks: List[SampleTask] = []
    for _, row in meta_df.iterrows():
        sample_id = row["id"]
        if include_ids and sample_id not in include_ids:
            continue
        st_path = st_dir / f"{sample_id}.h5ad"
        if not st_path.exists():
            # Attempt fuzzy match
            matches = sorted(st_dir.glob(f"*{sample_id}*.h5ad"))
            if matches:
                st_path = matches[0]
        metadata = row.to_dict()
        tasks.append(SampleTask(sample_id=sample_id, st_path=st_path, metadata=metadata))
        if sample_limit and len(tasks) >= sample_limit:
            break
    return tasks


def compute_sample_metrics(
    task: SampleTask,
    patch_size_px: float,
    metadata_columns: Sequence[str],
) -> Dict[str, object]:
    if not task.exists():
        return {"sample_id": task.sample_id, "error": f"Missing AnnData at {task.st_path}"}

    try:
        adata = sc.read_h5ad(task.st_path, backed="r")
        coords = get_spot_coordinates(adata)
        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[0] < 2:
            return {"sample_id": task.sample_id, "error": "Insufficient coordinate rows"}

        tree = cKDTree(coords)
        distances, _ = tree.query(coords, k=2)
        nn = distances[:, 1]
        metrics = {
            "sample_id": task.sample_id,
            "spot_count": int(coords.shape[0]),
            "nn_mean": float(nn.mean()),
            "nn_median": float(np.median(nn)),
            "nn_std": float(np.std(nn)),
            "nn_min": float(nn.min()),
            "nn_max": float(nn.max()),
            "nn_q25": float(np.quantile(nn, 0.25)),
            "nn_q75": float(np.quantile(nn, 0.75)),
        }

        for column in metadata_columns:
            if column in metrics:
                continue
            metrics[column] = task.metadata.get(column)

        if "st_technology" not in metrics:
            metrics["st_technology"] = task.metadata.get("st_technology", "unknown")

        gaps = nn - patch_size_px
        metrics.update(
            {
                "patch_size_px": float(patch_size_px),
                "gap_mean": float(gaps.mean()),
                "gap_median": float(np.median(gaps)),
                "gap_std": float(np.std(gaps)),
                "gap_q25": float(np.quantile(gaps, 0.25)),
                "gap_q75": float(np.quantile(gaps, 0.75)),
            }
        )
        return metrics
    except Exception as exc:  # pragma: no cover - defensive path
        return {"sample_id": task.sample_id, "error": str(exc)}
    finally:
        try:
            adata.file.close()  # type: ignore[attr-defined]
        except Exception:
            pass


def run_tasks(
    tasks: List[SampleTask],
    workers: int,
    executor_type: str,
    patch_size_px: float,
    metadata_columns: Sequence[str],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if not tasks:
        return [], []

    executor_cls = ProcessPoolExecutor if executor_type == "process" else ThreadPoolExecutor
    results: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []

    with executor_cls(max_workers=workers) as executor:
        future_map = {
            executor.submit(compute_sample_metrics, task, patch_size_px, metadata_columns): task.sample_id
            for task in tasks
        }
        for future in tqdm(as_completed(future_map), total=len(future_map), desc="Gap stats"):
            sample_id = future_map[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - defensive
                failures.append({"sample_id": sample_id, "error": str(exc)})
                continue
            if "error" in result:
                failures.append(result)
            else:
                results.append(result)
    return results, failures


def write_outputs(out_dir: Path, records: List[Dict[str, object]], failures: List[Dict[str, object]]):
    out_dir.mkdir(parents=True, exist_ok=True)

    if records:
        df = pd.DataFrame(records)
        df.sort_values("sample_id", inplace=True)
        df.to_csv(out_dir / "per_sample_gap_stats.csv", index=False)

        summary = {
            "sample_count": int(len(df)),
            "spot_count_total": int(df["spot_count"].sum()),
            "global_nn_mean": float(df["nn_mean"].mean()),
            "global_nn_median": float(df["nn_median"].median()),
            "global_gap_mean": float(df["gap_mean"].dropna().mean()) if df["gap_mean"].notna().any() else None,
        }

        tech_summary = {}
        for tech, tech_df in df.groupby("st_technology"):
            tech_summary[tech] = {
                "samples": int(len(tech_df)),
                "avg_spots": float(tech_df["spot_count"].mean()),
                "nn_mean": float(tech_df["nn_mean"].mean()),
                "gap_mean": float(tech_df["gap_mean"].dropna().mean()) if tech_df["gap_mean"].notna().any() else None,
            }
        summary["technology"] = tech_summary

        with (out_dir / "gap_summary.json").open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
    else:
        (out_dir / "per_sample_gap_stats.csv").write_text("sample_id,error\n", encoding="utf-8")

    if failures:
        with (out_dir / "failures.json").open("w", encoding="utf-8") as fp:
            json.dump(failures, fp, indent=2)


def main():
    args = parse_args()
    metadata_csv = args.metadata or args.raw / "HEST_v1_1_0.csv"
    meta_df = load_metadata(metadata_csv)
    include_ids = None
    if args.sample_ids:
        include_ids = {token.strip() for token in args.sample_ids.split(",") if token.strip()}
    tasks = build_tasks(args.raw, meta_df, include_ids=include_ids, sample_limit=args.sample_limit)
    if not tasks:
        raise SystemExit("No samples matched the provided filters.")

    if args.patch_size <= 0:
        raise SystemExit("--patch-size must be positive")
    metadata_columns = [token.strip() for token in (args.metadata_columns or "").split(",") if token.strip()]
    if "st_technology" not in metadata_columns:
        metadata_columns.append("st_technology")

    records, failures = run_tasks(
        tasks,
        args.workers,
        args.executor,
        args.patch_size,
        metadata_columns,
    )
    write_outputs(args.out, records, failures)

    print(f"Processed {len(records)} samples. Failures: {len(failures)}. Outputs saved to {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
