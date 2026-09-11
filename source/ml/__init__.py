"""ML layer facade with lazy exports for optional training dependencies."""

from __future__ import annotations

from importlib import import_module

_EXPORT_MODULES = {
    "EXPERT_VAK_CORRECTIONS": "source.ml.formulas",
    "GAS_CONTEXT_SIGNAL_IDS": "source.ml.blending",
    "DEFAULT_POLICY": "source.ml.policy",
    "ApplicabilityResult": "source.ml.uncertainty",
    "BlendOption": "source.ml.blending",
    "BlendResult": "source.ml.blending",
    "CalibratedPointUpperRegressor": "source.ml.uncertainty",
    "FeatureFrame": "source.ml.features",
    "GasContextSignal": "source.ml.blending",
    "HybridComponentForecast": "source.ml.blending",
    "ModelBundle": "source.ml.artifacts",
    "PolicyDecision": "source.ml.policy",
    "PolicyParameters": "source.ml.policy",
    "PolicyReplayCase": "source.ml.policy",
    "RobustnessCaseResult": "source.ml.uncertainty",
    "SHEET_LA": "source.ml.formulas",
    "SHEET_VAK": "source.ml.formulas",
    "Stage5FitResult": "source.ml.uncertainty",
    "SupervisedDataset": "source.ml.features",
    "TrainingResult": "source.ml.train",
    "UpperBoundCalibration": "source.ml.uncertainty",
    "UpperBoundMetrics": "source.ml.uncertainty",
    "apply_expert_vak_corrections": "source.ml.formulas",
    "apply_hydrotreater_forecast": "source.ml.blending",
    "assess_change_policy": "source.ml.policy",
    "build_features": "source.ml.features",
    "build_supervised_dataset": "source.ml.features",
    "calculate_mass_blend": "source.ml.blending",
    "check_applicability": "source.ml.uncertainty",
    "collect_gas_context": "source.ml.blending",
    "enumerate_two_component_recipes": "source.ml.blending",
    "evaluate_model": "source.ml.evaluate",
    "evaluate_robustness_cases": "source.ml.uncertainty",
    "evaluate_upper_bounds": "source.ml.uncertainty",
    "fit_stage5_uncertainty": "source.ml.uncertainty",
    "fit_upper_calibrator": "source.ml.uncertainty",
    "load_lab_parameters": "source.ml.formulas",
    "load_model": "source.ml.artifacts",
    "load_vak_formulas": "source.ml.formulas",
    "pinball_loss": "source.ml.uncertainty",
    "rank_feasible_blends": "source.ml.blending",
    "save_stage5_model": "source.ml.uncertainty",
    "sulfur_constraint_status": "source.ml.blending",
    "tune_policy": "source.ml.policy",
    "train_model": "source.ml.train",
}


def __getattr__(name: str) -> object:
    """Load facade exports only when they are requested."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module 'source.ml' has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = [
    "EXPERT_VAK_CORRECTIONS",
    "GAS_CONTEXT_SIGNAL_IDS",
    "DEFAULT_POLICY",
    "ApplicabilityResult",
    "BlendOption",
    "BlendResult",
    "CalibratedPointUpperRegressor",
    "FeatureFrame",
    "GasContextSignal",
    "HybridComponentForecast",
    "ModelBundle",
    "PolicyDecision",
    "PolicyParameters",
    "PolicyReplayCase",
    "RobustnessCaseResult",
    "SHEET_LA",
    "SHEET_VAK",
    "Stage5FitResult",
    "SupervisedDataset",
    "TrainingResult",
    "UpperBoundCalibration",
    "UpperBoundMetrics",
    "apply_expert_vak_corrections",
    "apply_hydrotreater_forecast",
    "assess_change_policy",
    "build_features",
    "build_supervised_dataset",
    "calculate_mass_blend",
    "check_applicability",
    "collect_gas_context",
    "enumerate_two_component_recipes",
    "evaluate_model",
    "evaluate_robustness_cases",
    "evaluate_upper_bounds",
    "fit_stage5_uncertainty",
    "fit_upper_calibrator",
    "load_lab_parameters",
    "load_model",
    "load_vak_formulas",
    "pinball_loss",
    "rank_feasible_blends",
    "save_stage5_model",
    "sulfur_constraint_status",
    "tune_policy",
    "train_model",
]
