"""Grid sweep over training / label / decision knobs, scored by walk-forward CV.

One :class:`TuningSpec` -- a fixed base config plus a ``grid`` of fields to vary
-- fans out to the cartesian product, runs each combination through
:func:`trader.service.training.run_cv`, and ranks the results by **median
out-of-sample Sharpe** (tie-break: lower median turnover). A per-config failure
(too few samples for a big window, say) is recorded, not fatal, so the rest of
the sweep still finishes.

The report is a plain JSON-safe dict in the spirit of
:func:`trader.data.depth.save_report`: :func:`build_report` assembles it,
:func:`render` prints a table, :func:`save_report` writes it under ``state/``.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from trader.config import Settings
from trader.service.errors import InvalidRequest
from trader.service.training import CVConfig, TrainingSpec, run_cv

__all__ = ["TuningSpec", "build_report", "render", "run_tuning", "save_report"]

# grid keys are matched against these two models' fields
_TRAIN_FIELDS = set(TrainingSpec.model_fields)
_CV_FIELDS = set(CVConfig.model_fields)


class TuningSpec(BaseModel):
    """A fixed base config, a walk-forward setup, and the grid to sweep."""

    base: TrainingSpec
    cv: CVConfig
    grid: dict[str, list[Any]] = Field(min_length=1)
    top_k: int = Field(default=5, ge=1)

    @field_validator("grid")
    @classmethod
    def _grid_keys_are_known(cls, grid: dict[str, list[Any]]) -> dict[str, list[Any]]:
        for key, values in grid.items():
            if not values:
                raise ValueError(f"grid[{key!r}] is empty")
            if key not in _TRAIN_FIELDS and key not in _CV_FIELDS:
                raise ValueError(f"grid key {key!r} is not a TrainingSpec or CVConfig field")
        return grid

    @model_validator(mode="after")
    def _base_has_no_split_dates(self) -> TuningSpec:
        # run_cv derives a split per fold; a fixed train_end/val_end would be ignored
        if self.base.train_end is not None or self.base.val_end is not None:
            raise ValueError("drop base.train_end / base.val_end -- run_cv derives folds")
        return self

    def combinations(self) -> list[dict[str, Any]]:
        keys = list(self.grid)
        return [
            dict(zip(keys, combo, strict=True)) for combo in itertools.product(*self.grid.values())
        ]

    def specs_for(self, overrides: dict[str, Any]) -> tuple[TrainingSpec, CVConfig]:
        train_over = {k: v for k, v in overrides.items() if k in _TRAIN_FIELDS}
        cv_over = {k: v for k, v in overrides.items() if k in _CV_FIELDS}
        # rebuild (not model_copy) so field validators run on the swept values
        train = TrainingSpec(**{**self.base.model_dump(), **train_over})
        cv = CVConfig(**{**self.cv.model_dump(), **cv_over})
        return train, cv


async def run_tuning(
    settings: Settings,
    spec: TuningSpec,
    *,
    progress: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Run every grid combination through walk-forward CV; return the report dict."""
    started_at = datetime.now(UTC)
    combos = spec.combinations()
    n = len(combos)

    results: list[dict[str, Any]] = []
    for i, overrides in enumerate(combos):

        def _sub(frac: float, i: int = i) -> None:
            if progress is not None:
                progress(max(0.0, min(1.0, (i + frac) / n)))

        train_spec, cv_config = spec.specs_for(overrides)
        row: dict[str, Any] = {"config": overrides}
        try:
            cv_out = await run_cv(settings, train_spec, cv_config, progress=_sub)
            row["aggregate"] = cv_out["aggregate"]
            row["folds"] = cv_out["folds"]
        except InvalidRequest as exc:
            row["error"] = str(exc)
        results.append(row)
    if progress is not None:
        progress(1.0)

    return build_report(spec, results, started_at=started_at)


def _rank_key(row: dict[str, Any]) -> tuple[float, float]:
    agg = row.get("aggregate")
    if not agg:
        return (float("inf"), float("inf"))  # errors sort last
    median_sharpe = agg["sharpe"].get("median")
    median_turnover = agg["turnover"].get("median")
    sharpe = median_sharpe if isinstance(median_sharpe, (int, float)) else float("-inf")
    turnover = median_turnover if isinstance(median_turnover, (int, float)) else float("inf")
    return (-sharpe, turnover)


def build_report(
    spec: TuningSpec, results: list[dict[str, Any]], *, started_at: datetime
) -> dict[str, Any]:
    """Assemble the JSON-safe sweep report (ranked, with the grid and CV setup)."""
    from trader.backtest.result import _json_safe

    order = sorted(range(len(results)), key=lambda j: _rank_key(results[j]))
    ranked = [results[j] for j in order]
    for pos, row in enumerate(ranked, start=1):
        row["rank"] = pos if "aggregate" in row else None

    report = {
        "measured_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "symbols": spec.base.symbols,
        "horizon": spec.base.horizon,
        "feature_set": spec.base.feature_set,
        "grid": spec.grid,
        "n_configs": len(results),
        "n_errored": sum(1 for r in results if "error" in r),
        "cv": spec.cv.model_dump(),
        "results": ranked,
        "top": [r for r in ranked if "aggregate" in r][: spec.top_k],
    }
    return _json_safe(report)


def _fmt_config(config: dict[str, Any]) -> str:
    return " ".join(f"{k}={v}" for k, v in config.items())


def render(report: dict[str, Any]) -> str:
    """Format a sweep report as a table for the terminal."""
    cv = report["cv"]
    header = (
        f"Tuning sweep: {report['n_configs']} configs, {cv['folds']} {cv['mode']} folds each, "
        f"costs fee/spread/slippage = {cv['fee_bps']}/{cv['spread_bps']}/{cv['slippage_bps']} bps"
    )
    grid = "Grid: " + "  ".join(f"{k}={v}" for k, v in report["grid"].items())
    lines = [header, grid, "-" * max(len(header), len(grid))]
    cols = f"{'#':>3}  {'config':<38}  {'folds>.5':>9}  {'Sharpe':>7}  {'ret%':>7}  {'turn':>7}"
    lines.append(cols)

    for row in report["results"]:
        cfg = _fmt_config(row["config"])
        agg = row.get("aggregate")
        if not agg:
            lines.append(f"{'--':>3}  {cfg:<38}  {'ERR':>9}  {row.get('error', '')}")
            continue
        folds = f"{agg['folds_sharpe_gt_0_5']}/{agg['n_folds']}"
        lines.append(
            f"{row['rank']:>3}  {cfg:<38}  {folds:>9}  "
            f"{_num(agg['sharpe']['median'], '{:+.2f}'):>7}  "
            f"{_num(agg['total_return']['median'], '{:+.1%}'):>7}  "
            f"{_num(agg['turnover']['median'], '{:.1f}x'):>7}"
        )
    return "\n".join(lines)


def _num(value: Any, fmt: str) -> str:
    return fmt.format(value) if isinstance(value, (int, float)) else "n/a"


def save_report(report: dict[str, Any], path: Path) -> Path:
    """Write a report as JSON, creating parent directories."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path
