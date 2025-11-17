#!/usr/bin/env python
"""Validate sharded preprocessing outputs for coordinate and sentence correctness."""
from __future__ import annotations

import json
import sys
import tarfile
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, cast

import anndata as ad
import numpy as np
import pandas as pd
import typer
from scipy.sparse import issparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.preprocessing.utils import get_spot_coordinates, get_top_k_genes  # noqa: E402

app = typer.Typer(help="Validate sharded dataset artifacts (coordinates + gene ranking).")


@dataclass
class SampleSummary:
    sample_id: str
    technology: Optional[str]
    total_spots_in_adata: int
    spots_requested: int
    spots_evaluated: int
    coordinate_mismatches: int
    missing_reference_coords: int
    gene_failures: int
    missing_payloads: int
    notes: List[Dict[str, object]]


@dataclass
class ValidationReport:
    dataset_key: str
    dataset_dir: str
    intermediate_adata: str
    total_samples: int
    evaluated_samples: int
    total_spots_in_adata: int
    spots_requested: int
    spots_evaluated: int
    coordinate_tolerance: float
    top_k_genes: int
    coordinate_mismatches: int
    missing_reference_coords: int
    gene_failures: int
    missing_payloads: int
    per_sample: List[SampleSummary]


def _load_manifest(dataset_dir: Path) -> dict:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise typer.BadParameter(f"No manifest.json under {dataset_dir}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _extract_coord(row: pd.Series, columns: Sequence[str]) -> Optional[float]:
    for col in columns:
        if col in row and pd.notna(row[col]):
            return float(row[col])
    return None


def _coerce_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(result):
        return None
    return result


def _get_expression_vector(matrix, idx: int) -> np.ndarray:
    row = matrix[idx]
    if issparse(row):
        return row.toarray().ravel()
    return np.asarray(row).ravel()


def _clean_tokens(sentence: str) -> List[str]:
    return sentence.strip().split()


def _choose_spots(indices: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if limit <= 0 or limit >= indices.size:
        return indices
    return rng.choice(indices, size=limit, replace=False)


def _read_payloads_for_sample(sample_dir: Path, targets: set[str]) -> Dict[str, Dict[str, Any]]:
    """Return mapping of spot -> {"json": {...}, "text": "..."}."""
    payloads: Dict[str, Dict[str, Any]] = defaultdict(dict)
    remaining: set[str] = set(targets)
    tar_paths = sorted(sample_dir.glob("*.tar"))
    for tar_path in tar_paths:
        if not remaining:
            break
        with tarfile.open(tar_path, "r") as tar_obj:
            for member in tar_obj:
                if not member.isfile():
                    continue
                base = Path(member.name).stem
                if base not in remaining:
                    continue
                extracted = tar_obj.extractfile(member)
                if extracted is None:
                    continue
                if member.name.endswith(".json"):
                    payloads[base]["json"] = json.loads(extracted.read().decode("utf-8"))
                elif member.name.endswith(".txt"):
                    payloads[base]["text"] = extracted.read().decode("utf-8")
                extracted.close()
                if "json" in payloads[base] and "text" in payloads[base]:
                    remaining.remove(base)
    # Any targets that were never seen should still appear in payloads so downstream logic records them.
    for missing in remaining:
        payloads.setdefault(missing, {})
    return payloads


def _summarize_sample(
    sample_id: str,
    payloads: Dict[str, Dict[str, Any]],
    obs_df: pd.DataFrame,
    obs_index: Dict[str, int],
    matrix,
    gene_names: np.ndarray,
    top_k: int,
    coord_tol: float,
    coord_matrix: Optional[np.ndarray],
) -> SampleSummary:
    sample_mask = obs_df["sample_id"] == sample_id
    total_spots = int(sample_mask.sum())
    evaluated = 0
    coord_mismatches = 0
    missing_coord_refs = 0
    gene_failures = 0
    missing_payloads = 0
    notes: List[Dict[str, object]] = []
    coord_cols = ("pxl_col_in_fullres", "pxl_col_in_fullres_old", "array_col")
    row_cols = ("pxl_row_in_fullres", "pxl_row_in_fullres_old", "array_row")

    for spot_name, bundle in payloads.items():
        # Consider only spots that belong to the current sample
        if not spot_name.startswith(f"{sample_id}_") and not spot_name.startswith(f"{sample_id}-") and spot_name != sample_id:
            continue
        if spot_name not in obs_index:
            missing_payloads += 1
            if len(notes) < 20:
                notes.append({"spot": spot_name, "error": "spot missing from AnnData"})
            continue
        obs_row = cast(pd.Series, obs_df.loc[spot_name])
        idx = obs_index[spot_name]

        spot_notes: Dict[str, object] = {"spot": spot_name}

        json_payload = cast(Optional[Dict[str, Any]], bundle.get("json"))
        text_payload_obj = bundle.get("text")
        if not json_payload or not isinstance(text_payload_obj, str):
            missing_payloads += 1
            spot_notes["error"] = "missing json or text payload"
            if len(notes) < 20:
                notes.append(spot_notes)
            continue
        text_payload = text_payload_obj
        evaluated += 1

        # Coordinate validation
        coord_error: Optional[object] = None
        coord_x = _extract_coord(obs_row, coord_cols)
        coord_y = _extract_coord(obs_row, row_cols)
        if (coord_x is None or coord_y is None) and coord_matrix is not None:
            coord_x = float(coord_matrix[idx, 0])
            coord_y = float(coord_matrix[idx, 1])
        payload_x = _coerce_float(json_payload.get("x"))
        payload_y = _coerce_float(json_payload.get("y"))
        if coord_x is None or coord_y is None or np.isnan(coord_x) or np.isnan(coord_y):
            missing_coord_refs += 1
            coord_error = "missing coordinate columns"
        elif payload_x is None or payload_y is None:
            coord_mismatches += 1
            coord_error = "missing coordinate(s) in payload"
        else:
            dx = abs(coord_x - payload_x)
            dy = abs(coord_y - payload_y)
            if max(dx, dy) > coord_tol:
                coord_mismatches += 1
                coord_error = {"dx": dx, "dy": dy}

        if coord_error is not None:
            spot_notes["coord_error"] = coord_error

        # Sentence validation
        expr_vec = _get_expression_vector(matrix, idx)
        expected_sentence = get_top_k_genes(expr_vec, gene_names, top_k)
        expected_tokens = _clean_tokens(expected_sentence)
        actual_tokens = _clean_tokens(text_payload)
        if expected_tokens != actual_tokens:
            gene_failures += 1
            spot_notes["gene_error"] = {
                "expected": expected_tokens[:10],
                "actual": actual_tokens[:10],
            }

        if ("coord_error" in spot_notes or "gene_error" in spot_notes) and len(notes) < 20:
            notes.append(spot_notes)

    technology = None
    if total_spots:
        first_idx = np.where(sample_mask)[0]
        if first_idx.size:
            technology = str(obs_df.iloc[first_idx[0]].get("st_technology"))
    return SampleSummary(
        sample_id=sample_id,
        technology=technology,
        total_spots_in_adata=total_spots,
        spots_requested=len(payloads),
        spots_evaluated=evaluated,
        coordinate_mismatches=coord_mismatches,
        missing_reference_coords=missing_coord_refs,
        gene_failures=gene_failures,
        missing_payloads=missing_payloads,
        notes=notes,
    )


@app.command()
def validate(
    dataset_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, help="Processed dataset directory."),
    intermediate_dir: Optional[Path] = typer.Option(
        None,
        "--intermediate-dir",
        help="Override path to processed_intermediate/<dataset>/ directory containing adata_final_for_sharding.h5ad.",
    ),
    max_spots_per_sample: int = typer.Option(
        200,
        "--max-spots-per-sample",
        help="Maximum number of spots to evaluate per sample (<=0 means all spots).",
    ),
    coordinate_tolerance: float = typer.Option(
        1.5,
        "--coord-tol",
        help="Allowable absolute pixel difference when comparing JSON vs AnnData coordinates.",
    ),
    samples: Optional[List[str]] = typer.Option(
        None,
        "--sample",
        help="Optional sample IDs to restrict validation to (can be provided multiple times).",
    ),
    seed: int = typer.Option(17, help="RNG seed for spot sampling."),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output",
        help="Optional path to write the validation report JSON.",
    ),
):
    manifest = _load_manifest(dataset_dir)
    dataset_key = manifest.get("dataset", {}).get("key", dataset_dir.name)
    typer.echo(f"📦 Dataset: {dataset_key}")

    inter_dir = intermediate_dir or Path(manifest.get("paths", {}).get("intermediate_dir", ""))
    if not inter_dir:
        raise typer.BadParameter("Could not determine intermediate directory; pass --intermediate-dir.")
    adata_path = Path(inter_dir) / "adata_final_for_sharding.h5ad"
    if not adata_path.exists():
        raise typer.BadParameter(f"Missing reference AnnData at {adata_path}")
    typer.echo(f"🔍 Loading AnnData: {adata_path}")
    reference = ad.read_h5ad(adata_path)
    obs_df = reference.obs.copy()
    obs_index = {name: idx for idx, name in enumerate(reference.obs_names)}
    gene_names = reference.var_names.to_numpy()
    matrix = reference.X
    coord_matrix = None
    try:
        coord_matrix = get_spot_coordinates(reference)
    except Exception as exc:  # pragma: no cover - coordinate fallback guard
        typer.secho(f"Failed to compute canonical coordinates via get_spot_coordinates: {exc}", fg=typer.colors.RED)

    top_k = int(manifest.get("hydra_config", {}).get("resolved", {}).get("params", {}).get("sentence_generation", {}).get("n_top_genes", 50))
    sample_dirs = manifest.get("outputs", {}).get("sample_dirs", []) or [p.name for p in dataset_dir.iterdir() if p.is_dir()]
    sample_dirs = sorted(sample_dirs)
    if samples:
        sample_dirs = [s for s in sample_dirs if s in samples]
        if not sample_dirs:
            raise typer.BadParameter("None of the requested --sample entries match the dataset.")

    rng = np.random.default_rng(seed)
    per_sample_reports: List[SampleSummary] = []
    total_requested = 0
    total_evaluated = 0
    for sample_id in sample_dirs:
        sample_path = dataset_dir / sample_id
        if not sample_path.exists():
            typer.echo(f"⚠️  Missing sample directory {sample_path}")
            continue
        sample_mask = obs_df["sample_id"] == sample_id
        if not sample_mask.any():
            typer.echo(f"⚠️  Sample {sample_id} missing from AnnData; skipping")
            continue
        sample_indices = np.where(sample_mask)[0]
        chosen_indices = _choose_spots(sample_indices, max_spots_per_sample, rng)
        target_spots = {str(reference.obs_names[idx]) for idx in chosen_indices.tolist()}
        total_requested += len(target_spots)
        payloads = _read_payloads_for_sample(sample_path, target_spots)
        summary = _summarize_sample(
            sample_id=sample_id,
            payloads=payloads,
            obs_df=obs_df,
            obs_index=obs_index,
            matrix=matrix,
            gene_names=gene_names,
            top_k=top_k,
            coord_tol=coordinate_tolerance,
            coord_matrix=coord_matrix,
        )
        total_evaluated += summary.spots_evaluated
        per_sample_reports.append(summary)
        typer.echo(
            "✅ {}: checked {} spots (coord_mismatch={}, coord_missing_ref={}, gene_fail={})".format(
                sample_id,
                summary.spots_evaluated,
                summary.coordinate_mismatches,
                summary.missing_reference_coords,
                summary.gene_failures,
            )
        )

    total_coord_mismatches = sum(s.coordinate_mismatches for s in per_sample_reports)
    total_coord_missing = sum(s.missing_reference_coords for s in per_sample_reports)
    total_gene_fail = sum(s.gene_failures for s in per_sample_reports)
    total_missing = sum(s.missing_payloads for s in per_sample_reports)

    if total_evaluated and total_gene_fail / total_evaluated > 0.9:
        typer.secho(
            "⚠️  Gene validation failure rate exceeds 90%; reference AnnData preprocessing may not match sharding pipeline.",
            fg=typer.colors.YELLOW,
        )

    manifest_samples = manifest.get("stats", {}).get("samples")
    manifest_sample_count = len(manifest_samples) if isinstance(manifest_samples, dict) else len(sample_dirs)

    report = ValidationReport(
        dataset_key=dataset_key,
        dataset_dir=str(dataset_dir),
        intermediate_adata=str(adata_path),
        total_samples=manifest_sample_count,
        evaluated_samples=len(per_sample_reports),
        total_spots_in_adata=reference.n_obs,
        spots_requested=total_requested,
        spots_evaluated=total_evaluated,
        coordinate_tolerance=coordinate_tolerance,
        top_k_genes=top_k,
        coordinate_mismatches=total_coord_mismatches,
        missing_reference_coords=total_coord_missing,
        gene_failures=total_gene_fail,
        missing_payloads=total_missing,
        per_sample=per_sample_reports,
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(
            {
                **asdict(report),
                "per_sample": [asdict(s) for s in per_sample_reports],
            },
            output_path.open("w", encoding="utf-8"),
            indent=2,
        )
        typer.echo(f"📝 Wrote report to {output_path}")

    if total_coord_mismatches or total_coord_missing or total_gene_fail or total_missing:
        typer.secho(
            "Validation completed with issues (coord_mismatch={}, coord_missing_ref={}, gene_fail={}, missing={}).".format(
                total_coord_mismatches, total_coord_missing, total_gene_fail, total_missing
            ),
            fg=typer.colors.YELLOW,
        )
    else:
        typer.secho(
            f"Validation successful across {total_evaluated} spots (no discrepancies detected).",
            fg=typer.colors.GREEN,
        )


if __name__ == "__main__":
    app()
