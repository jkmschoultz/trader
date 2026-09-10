"""Grid sweep over any strategy's knobs, scored by one walk-forward CV.

One :class:`TuningSpec` -- a fixed base config plus a ``grid`` of fields to vary
-- fans out to the cartesian product, runs each combination through the right
CV path (:func:`trader.service.training.run_cv` for ``lstm`` / ``gbm``,
:func:`trader.service.evaluation.run_strategy_cv` for a classical strategy like
``ma_cross`` / ``orb``), and ranks the results by **median out-of-sample
Sharpe** (tie-break: lower median turnover). A per-config failure (too few
samples for a big window, say) is recorded, not fatal, so the rest of the sweep
still finishes.

The report is a plain JSON-safe dict in the spirit of
:func:`trader.data.depth.save_report`: :func:`build_report` assembles it,
:func:`render` prints a table, :func:`save_report` writes it under ``state/``.
"""

from __future__ import annotations

import asyncio
import itertools
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from trader.config import Settings
from trader.service.errors import InvalidRequest
from trader.service.evaluation import MODEL_STRATEGIES, StrategyCVSpec, run_strategy_cv
from trader.service.training import CVConfig, TrainingSpec, run_cv

__all__ = ["TuningSpec", "build_report", "render", "run_tuning", "save_report"]

# grid keys for a model sweep are matched against these two models' fields
_TRAIN_FIELDS = set(TrainingSpec.model_fields)
_CV_FIELDS = set(CVConfig.model_fields)
# extra grid keys a classical sweep may vary (everything else must be a ctor arg)
_SCORING_FIELDS = {"allocator", "leverage", "fee_bps", "spread_bps", "slippage_bps"}

# either (TrainingSpec, CVConfig) for a model sweep or a StrategyCVSpec for a classical one
JobSpec = tuple[TrainingSpec, CVConfig] | StrategyCVSpec


class TuningSpec(BaseModel):
    """A fixed base config, a walk-forward setup, and the grid to sweep.

    ``strategy`` selects the CV path: ``"lstm"`` / ``"gbm"`` sweep
    ``TrainingSpec`` / ``CVConfig`` fields and train a model per fold; any other
    registered name (``"ma_cross"``, ``"orb"``, …) sweeps that strategy's
    constructor args plus the scoring knobs and runs no training. ``params``
    holds fixed strategy args for the classical case.
    """

    base: TrainingSpec
    cv: CVConfig
    strategy: str = "lstm"
    params: dict[str, Any] = Field(default_factory=dict)
    grid: dict[str, list[Any]] = Field(min_length=1)
    top_k: int = Field(default=5, ge=1)
    #: parallel workers for the sweep. 0 = auto (min(4, configs, cpu//2)); 1 = in-process.
    max_workers: int = Field(default=0, ge=0)

    def resolved_workers(self, n_configs: int) -> int:
        if self.max_workers:
            return max(1, min(self.max_workers, n_configs))
        return max(1, min(4, n_configs, (os.cpu_count() or 2) // 2))

    @property
    def is_model_sweep(self) -> bool:
        return self.strategy in MODEL_STRATEGIES

    @field_validator("grid")
    @classmethod
    def _grid_values_nonempty(cls, grid: dict[str, list[Any]]) -> dict[str, list[Any]]:
        for key, values in grid.items():
            if not values:
                raise ValueError(f"grid[{key!r}] is empty")
        return grid

    @model_validator(mode="after")
    def _validate_shape(self) -> TuningSpec:
        # run_cv / run_strategy_cv derive a split per fold; fixed dates are ignored
        if self.base.train_end is not None or self.base.val_end is not None:
            raise ValueError("drop base.train_end / base.val_end -- folds are derived")

        if self.is_model_sweep:
            valid = _TRAIN_FIELDS | _CV_FIELDS
        else:
            import inspect

            from trader.strategies import UnknownStrategy, get_strategy

            try:
                strategy_cls = get_strategy(self.strategy)
            except UnknownStrategy as exc:
                raise ValueError(str(exc)) from exc
            ctor = set(inspect.signature(strategy_cls.__init__).parameters) - {"self", "models_dir"}
            valid = ctor | _SCORING_FIELDS
        unknown = [k for k in self.grid if k not in valid]
        if unknown:
            raise ValueError(
                f"grid key(s) {unknown} not valid for strategy {self.strategy!r} "
                f"(allowed: {', '.join(sorted(valid))})"
            )
        return self

    def combinations(self) -> list[dict[str, Any]]:
        keys = list(self.grid)
        return [
            dict(zip(keys, combo, strict=True)) for combo in itertools.product(*self.grid.values())
        ]

    def specs_for(self, overrides: dict[str, Any]) -> JobSpec:
        if self.is_model_sweep:
            train_over = {k: v for k, v in overrides.items() if k in _TRAIN_FIELDS}
            cv_over = {k: v for k, v in overrides.items() if k in _CV_FIELDS}
            # rebuild (not model_copy) so field validators run on the swept values
            # strategy is "lstm" or "gbm" here -- which is exactly the model_type
            train = TrainingSpec(
                **{**self.base.model_dump(), "model_type": self.strategy, **train_over}
            )
            cv = CVConfig(**{**self.cv.model_dump(), **cv_over})
            return train, cv

        param_over = {k: v for k, v in overrides.items() if k not in _SCORING_FIELDS}
        scoring = {
            "fee_bps": self.cv.fee_bps,
            "spread_bps": self.cv.spread_bps,
            "slippage_bps": self.cv.slippage_bps,
            "allocator": self.cv.allocator,
            "leverage": self.cv.leverage,
        }
        scoring.update({k: v for k, v in overrides.items() if k in _SCORING_FIELDS})
        return StrategyCVSpec(
            symbols=self.base.symbols,
            uics=self.base.uics,
            asset_type=self.base.asset_type,
            exchange=self.base.exchange,
            strategy=self.strategy,
            params={**self.params, **param_over},
            horizon=self.base.horizon,
            since=self.base.since,
            folds=self.cv.folds,
            mode=self.cv.mode,
            train_days=self.cv.train_days,
            val_days=self.cv.val_days,
            test_days=self.cv.test_days,
            step_days=self.cv.step_days,
            **scoring,
        )


async def run_tuning(
    settings: Settings,
    spec: TuningSpec,
    *,
    progress: Callable[[float], None] | None = None,
    on_message: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run every grid combination through walk-forward CV; return the report dict.

    ``progress`` gets a 0..1 fraction over the whole sweep; ``on_message`` gets a
    short human line as each config starts and finishes (config i/n, best Sharpe
    so far) -- surfaced on the job so the UI can show which combo is running.
    """
    started_at = datetime.now(UTC)
    combos = spec.combinations()
    n = len(combos)

    def _say(text: str) -> None:
        if on_message is not None:
            on_message(text)

    def _best_median(rows: list[dict[str, Any]]) -> float | None:
        vals = [
            r["aggregate"]["sharpe"]["median"]
            for r in rows
            if r.get("aggregate") and isinstance(r["aggregate"]["sharpe"]["median"], (int, float))
        ]
        return max(vals) if vals else None

    jobs = [(settings, spec.specs_for(o)) for o in combos]
    workers = spec.resolved_workers(n)
    results: list[dict[str, Any] | None] = [None] * n
    done = 0

    def _finish(i: int, outcome: dict[str, Any]) -> None:
        nonlocal done
        results[i] = {"config": combos[i], **outcome}
        done += 1
        if progress is not None:
            progress(done / n)
        best = _best_median([r for r in results if r])
        best_txt = f"best median Sharpe {best:+.2f}" if best is not None else "no config scored yet"
        _say(f"config {done}/{n} done — {best_txt}")

    _say(f"sweeping {n} configs on {workers} worker{'s' if workers != 1 else ''}")

    if workers == 1:
        for i, (s, job_spec) in enumerate(jobs):

            def _sub(frac: float, i: int = i) -> None:
                if progress is not None:
                    progress(max(0.0, min(1.0, (i + frac) / n)))

            _say(f"config {i + 1}/{n}: {_fmt_config(combos[i]) or 'base'} — {spec.cv.folds} folds")
            _finish(i, await _cv_outcome(s, job_spec, progress=_sub))
    else:
        loop = asyncio.get_running_loop()
        ctx = get_context("spawn")  # torch is already loaded in the parent; fork would be unsafe
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:

            async def _one(i: int) -> tuple[int, dict[str, Any]]:
                return i, await loop.run_in_executor(pool, _run_config_sync, jobs[i])

            for coro in asyncio.as_completed([_one(i) for i in range(n)]):
                i, outcome = await coro
                _finish(i, outcome)

    if progress is not None:
        progress(1.0)
    return build_report(spec, [r for r in results if r], started_at=started_at)


async def _cv_outcome(
    settings: Settings,
    job_spec: JobSpec,
    *,
    progress: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """One config's CV result as a report row body: ``aggregate`` + ``folds``, or ``error``."""
    try:
        if isinstance(job_spec, tuple):
            train_spec, cv_config = job_spec
            out = await run_cv(settings, train_spec, cv_config, progress=progress)
        else:
            out = await run_strategy_cv(settings, job_spec, progress=progress)
    except InvalidRequest as exc:
        return {"error": str(exc)}
    return {"aggregate": out["aggregate"], "folds": out["folds"]}


def _run_config_sync(job: tuple[Settings, JobSpec]) -> dict[str, Any]:
    """Process-pool entry point: run one config's CV in a fresh event loop."""
    settings, job_spec = job
    return asyncio.run(_cv_outcome(settings, job_spec))


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
        "strategy": spec.strategy,
        "symbols": spec.base.symbols,
        "horizon": spec.base.horizon,
        "feature_set": spec.base.feature_set if spec.is_model_sweep else None,
        "params": spec.params if not spec.is_model_sweep else None,
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
        f"Tuning sweep [{report.get('strategy', 'lstm')}]: {report['n_configs']} configs, "
        f"{cv['folds']} {cv['mode']} folds each, "
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
