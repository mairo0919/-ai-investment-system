"""Load and merge simulation configuration."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.config.settings import PROJECT_ROOT
from src.simulation.execution_policy import ExecutionPolicy

CandidateMode = Literal["top_percentile", "top_n"]
RebalanceFrequency = Literal["daily", "weekly", "biweekly"]


@dataclass(frozen=True)
class SimulationConfig:
    raw: dict[str, Any]
    initial_capital: float
    base_currency: str
    top_percentile: float
    top_n: int
    candidate_mode: CandidateMode
    max_positions: int
    max_position_weight: float
    max_country_weight: float
    holding_period_days: int
    stop_loss_pct: float | None
    take_profit_pct: float | None
    ranking_exit_enabled: bool
    ranking_exit_percentile: float
    commission_rate: float
    slippage_rate: float
    label_scheme: str
    gain_name: str
    feature_set: str
    param_preset: str
    horizon_days: int
    fx_tickers: dict[str, str]
    benchmarks: dict[str, str]
    strategies: dict[str, dict[str, Any]]
    rebalance_frequency: RebalanceFrequency = "daily"
    weekly_weekday: int = 0
    block_low_resolution_entries: bool = False
    low_resolution_tie_threshold: float = 0.5
    min_unique_score_ratio: float | None = None
    min_score_std: float | None = None
    min_top_median_gap: float | None = None
    min_cutoff_median_gap: float | None = None
    use_train_dispersion_filter: bool = False
    use_score_separation_filter: bool = False
    train_threshold_quantile: float = 0.25
    trailing_stop_pct: float | None = None
    cooldown_days: int = 0
    cost_presets: dict[str, dict[str, float]] | None = None
    # Opt-in; default preserves legacy fractional / equal-weight behavior.
    execution_policy: ExecutionPolicy = ExecutionPolicy()

    def with_strategy(self, name: str) -> SimulationConfig:
        if name not in self.strategies:
            raise KeyError(f"Unknown strategy: {name}")
        return self.with_overrides(**self.strategies[name])

    def with_cost_preset(self, name: str) -> SimulationConfig:
        presets = self.cost_presets or {}
        if name not in presets:
            raise KeyError(f"Unknown cost preset: {name}")
        return self.with_overrides(**presets[name])

    def with_overrides(self, **overlay: Any) -> SimulationConfig:
        data = deepcopy(self.raw)
        cand = data.setdefault("candidate", {})
        port = data.setdefault("portfolio", {})
        exits = data.setdefault("exit_rules", {})
        costs = data.setdefault("costs", {})
        reb = data.setdefault("rebalance", {})
        qf = data.setdefault("quality_filters", {})

        mapping = {
            "top_percentile": ("candidate", "top_percentile"),
            "top_n": ("candidate", "top_n"),
            "candidate_mode": ("candidate", "mode"),
            "max_positions": ("portfolio", "max_positions"),
            "max_position_weight": ("portfolio", "max_position_weight"),
            "max_country_weight": ("portfolio", "max_country_weight"),
            "holding_period_days": ("exit_rules", "holding_period_days"),
            "stop_loss_pct": ("exit_rules", "stop_loss_pct"),
            "take_profit_pct": ("exit_rules", "take_profit_pct"),
            "ranking_exit_enabled": ("exit_rules", "ranking_exit_enabled"),
            "ranking_exit_percentile": ("exit_rules", "ranking_exit_percentile"),
            "commission_rate": ("costs", "commission_rate"),
            "slippage_rate": ("costs", "slippage_rate"),
            "rebalance_frequency": ("rebalance", "frequency"),
            "weekly_weekday": ("rebalance", "weekly_weekday"),
            "block_low_resolution_entries": ("quality_filters", "block_low_resolution_entries"),
            "low_resolution_tie_threshold": ("quality_filters", "low_resolution_tie_threshold"),
            "min_unique_score_ratio": ("quality_filters", "min_unique_score_ratio"),
            "min_score_std": ("quality_filters", "min_score_std"),
            "min_top_median_gap": ("quality_filters", "min_top_median_gap"),
            "min_cutoff_median_gap": ("quality_filters", "min_cutoff_median_gap"),
            "use_train_dispersion_filter": ("quality_filters", "use_train_dispersion_filter"),
            "use_score_separation_filter": ("quality_filters", "use_score_separation_filter"),
            "train_threshold_quantile": ("quality_filters", "train_threshold_quantile"),
            "trailing_stop_pct": ("exit_rules", "trailing_stop_pct"),
            "cooldown_days": ("exit_rules", "cooldown_days"),
        }
        buckets = {
            "candidate": cand,
            "portfolio": port,
            "exit_rules": exits,
            "costs": costs,
            "rebalance": reb,
            "quality_filters": qf,
        }
        for key, value in overlay.items():
            if key in mapping:
                section, field = mapping[key]
                buckets[section][field] = value
            elif key == "mode":
                cand["mode"] = value
            elif key == "initial_capital":
                data["initial_capital"] = value
            elif key == "execution_policy":
                if isinstance(value, ExecutionPolicy):
                    data["execution_policy"] = {
                        "share_mode": value.share_mode,
                        "minimum_quantity": value.minimum_quantity,
                        "allow_fractional": value.allow_fractional,
                        "sizing_mode": value.sizing_mode,
                        "min_lot_overrides_weight": value.min_lot_overrides_weight,
                    }
                elif isinstance(value, dict):
                    data["execution_policy"] = deepcopy(value)
        # Preserve strategies/cost_presets / execution_policy from original when unset
        if "strategies" not in data:
            data["strategies"] = deepcopy(self.strategies)
        if self.cost_presets is not None:
            data["cost_presets"] = deepcopy(self.cost_presets)
        if "execution_policy" not in data and "execution_policy" in self.raw:
            data["execution_policy"] = deepcopy(self.raw["execution_policy"])
        return parse_simulation_config(data)

    def apply_runtime_thresholds(
        self,
        *,
        min_score_std: float | None = None,
        min_top_median_gap: float | None = None,
        min_unique_score_ratio: float | None = None,
        min_cutoff_median_gap: float | None = None,
    ) -> SimulationConfig:
        """Attach fold-train-derived thresholds without mutating strategy identity."""
        return self.with_overrides(
            min_score_std=min_score_std if min_score_std is not None else self.min_score_std,
            min_top_median_gap=(
                min_top_median_gap
                if min_top_median_gap is not None
                else self.min_top_median_gap
            ),
            min_unique_score_ratio=(
                min_unique_score_ratio
                if min_unique_score_ratio is not None
                else self.min_unique_score_ratio
            ),
            min_cutoff_median_gap=(
                min_cutoff_median_gap
                if min_cutoff_median_gap is not None
                else self.min_cutoff_median_gap
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self.raw)


def load_simulation_config(path: Path | None = None) -> SimulationConfig:
    cfg_path = path or (PROJECT_ROOT / "config" / "simulation.json")
    data = _load_json_with_extends(cfg_path)
    return parse_simulation_config(data)


def _load_json_with_extends(path: Path, *, _seen: set[str] | None = None) -> dict[str, Any]:
    """Load JSON config and recursively resolve ``extends`` chains."""
    resolved = str(path.resolve())
    seen = _seen or set()
    if resolved in seen:
        raise ValueError(f"Circular config extends detected at {path}")
    seen.add(resolved)
    data = json.loads(path.read_text(encoding="utf-8"))
    extends = data.get("extends")
    if not extends:
        return data
    parent_path = PROJECT_ROOT / str(extends) if not Path(extends).is_absolute() else Path(extends)
    parent = _load_json_with_extends(parent_path, _seen=seen)
    merged = _deep_merge(parent, data)
    merged.pop("extends", None)
    return merged


def parse_simulation_config(data: dict[str, Any]) -> SimulationConfig:
    cand = data.get("candidate", {})
    port = data.get("portfolio", {})
    exits = data.get("exit_rules", {})
    costs = data.get("costs", {})
    ranking = data.get("ranking", {})
    fx = data.get("fx", {})
    reb = data.get("rebalance", {})
    qf = data.get("quality_filters", {})
    mode = str(cand.get("mode", "top_percentile"))
    if mode not in ("top_percentile", "top_n"):
        raise ValueError(f"Invalid candidate.mode: {mode}")
    freq = str(reb.get("frequency", "daily"))
    if freq not in ("daily", "weekly", "biweekly"):
        raise ValueError(f"Invalid rebalance.frequency: {freq}")
    presets = {
        "gross": {"commission_rate": 0.0, "slippage_rate": 0.0},
        "low": {"commission_rate": 0.0005, "slippage_rate": 0.00025},
        "net": {"commission_rate": 0.001, "slippage_rate": 0.0005},
        "high": {"commission_rate": 0.0015, "slippage_rate": 0.001},
    }
    if "cost_presets" in data:
        presets.update(dict(data["cost_presets"]))
    policy_raw = data.get("execution_policy")
    if policy_raw is None and isinstance(data.get("execution"), dict):
        policy_raw = data["execution"].get("policy")
    return SimulationConfig(
        raw=deepcopy(data),
        initial_capital=float(data.get("initial_capital", 10_000_000)),
        base_currency=str(data.get("base_currency", "JPY")),
        top_percentile=float(cand.get("top_percentile", 0.10)),
        top_n=int(cand.get("top_n", 3)),
        candidate_mode=mode,  # type: ignore[arg-type]
        max_positions=int(port.get("max_positions", 10)),
        max_position_weight=float(port.get("max_position_weight", 0.10)),
        max_country_weight=float(port.get("max_country_weight", 0.50)),
        holding_period_days=int(exits.get("holding_period_days", 5)),
        stop_loss_pct=_opt_float(exits.get("stop_loss_pct")),
        take_profit_pct=_opt_float(exits.get("take_profit_pct")),
        ranking_exit_enabled=bool(exits.get("ranking_exit_enabled", False)),
        ranking_exit_percentile=float(exits.get("ranking_exit_percentile", 0.30)),
        commission_rate=float(costs.get("commission_rate", 0.001)),
        slippage_rate=float(costs.get("slippage_rate", 0.0005)),
        label_scheme=str(ranking.get("label_scheme", "B")),
        gain_name=str(ranking.get("gain_name", "moderate_exp")),
        feature_set=str(ranking.get("feature_set", "A")),
        param_preset=str(ranking.get("param_preset", "default")),
        horizon_days=int(ranking.get("horizon_days", 5)),
        fx_tickers=dict(fx.get("tickers", {})),
        benchmarks=dict(data.get("benchmarks", {})),
        strategies=dict(data.get("strategies", {})),
        rebalance_frequency=freq,  # type: ignore[arg-type]
        weekly_weekday=int(reb.get("weekly_weekday", 0)),
        block_low_resolution_entries=bool(qf.get("block_low_resolution_entries", False)),
        low_resolution_tie_threshold=float(qf.get("low_resolution_tie_threshold", 0.5)),
        min_unique_score_ratio=_opt_float(qf.get("min_unique_score_ratio")),
        min_score_std=_opt_float(qf.get("min_score_std")),
        min_top_median_gap=_opt_float(qf.get("min_top_median_gap")),
        min_cutoff_median_gap=_opt_float(qf.get("min_cutoff_median_gap")),
        use_train_dispersion_filter=bool(qf.get("use_train_dispersion_filter", False)),
        use_score_separation_filter=bool(qf.get("use_score_separation_filter", False)),
        train_threshold_quantile=float(qf.get("train_threshold_quantile", 0.25)),
        trailing_stop_pct=_opt_float(exits.get("trailing_stop_pct")),
        cooldown_days=int(exits.get("cooldown_days", 0) or 0),
        cost_presets=presets,
        execution_policy=ExecutionPolicy.from_dict(
            policy_raw if isinstance(policy_raw, dict) else None
        ),
    )


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in overlay.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out
