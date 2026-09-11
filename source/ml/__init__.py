"""ML layer facade with lazy exports for optional training dependencies."""

from __future__ import annotations

from importlib import import_module

_EXPORT_MODULES = {
    "EXPERT_VAK_CORRECTIONS": "source.ml.formulas",
    "GAS_CONTEXT_SIGNAL_IDS": "source.ml.blending",
    "BlendOption": "source.ml.blending",
    "BlendResult": "source.ml.blending",
    "FeatureFrame": "source.ml.features",
    "GasContextSignal": "source.ml.blending",
    "HybridComponentForecast": "source.ml.blending",
    "ModelBundle": "source.ml.artifacts",
    "SHEET_LA": "source.ml.formulas",
    "SHEET_VAK": "source.ml.formulas",
    "SupervisedDataset": "source.ml.features",
    "TrainingResult": "source.ml.train",
    "apply_expert_vak_corrections": "source.ml.formulas",
    "apply_hydrotreater_forecast": "source.ml.blending",
    "build_features": "source.ml.features",
    "build_supervised_dataset": "source.ml.features",
    "calculate_mass_blend": "source.ml.blending",
    "collect_gas_context": "source.ml.blending",
    "enumerate_two_component_recipes": "source.ml.blending",
    "evaluate_model": "source.ml.evaluate",
    "load_lab_parameters": "source.ml.formulas",
    "load_model": "source.ml.artifacts",
    "load_vak_formulas": "source.ml.formulas",
    "rank_feasible_blends": "source.ml.blending",
    "sulfur_constraint_status": "source.ml.blending",
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
    "BlendOption",
    "BlendResult",
    "FeatureFrame",
    "GasContextSignal",
    "HybridComponentForecast",
    "ModelBundle",
    "SHEET_LA",
    "SHEET_VAK",
    "SupervisedDataset",
    "TrainingResult",
    "apply_expert_vak_corrections",
    "apply_hydrotreater_forecast",
    "build_features",
    "build_supervised_dataset",
    "calculate_mass_blend",
    "collect_gas_context",
    "enumerate_two_component_recipes",
    "evaluate_model",
    "load_lab_parameters",
    "load_model",
    "load_vak_formulas",
    "rank_feasible_blends",
    "sulfur_constraint_status",
    "train_model",
]
