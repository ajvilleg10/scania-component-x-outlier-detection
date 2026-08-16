from __future__ import annotations

from pathlib import Path
from typing import Any

from scania_outliers.model_factory import ModelFactory
from scania_outliers.pipelines.context import PipelineContext


class ScaniaOutlierPipeline:
    """Object-oriented orchestration layer for the full TFM workflow."""

    def __init__(self, config: dict[str, Any], config_path: Path, run_id: str | None = None):
        self.ctx = PipelineContext(config=config, config_path=config_path, run_id=run_id)
        self._stages = None

    @property
    def stages(self):
        # Lazy import keeps dry-run and CLI help usable even before PySpark is installed.
        if self._stages is None:
            from scania_outliers.pipelines.stages import (
                CheckDataStage,
                ComparisonStage,
                EDAStage,
                EvaluationStage,
                PreprocessingWindowingStage,
                TrainingStage,
            )
            self._stages = {
                "check-data": CheckDataStage(self.ctx),
                "eda": EDAStage(self.ctx),
                "preprocess": PreprocessingWindowingStage(self.ctx),
                "train": TrainingStage(self.ctx),
                "evaluate": EvaluationStage(self.ctx),
                "compare": ComparisonStage(self.ctx),
            }
        return self._stages

    def dry_run(self, stage: str = "all", model: str = "all") -> None:
        config = self.ctx.config
        self.ctx.logger.info("Dry run for stage=%s model=%s", stage, model)
        self.ctx.logger.info("Project: %s", config.get("project", {}).get("name"))
        self.ctx.logger.info("Execution mode: %s", config.get("execution", {}).get("mode"))
        self.ctx.logger.info("Raw dir: %s", config.get("paths", {}).get("raw_dir"))
        self.ctx.logger.info("Processed dir: %s", config.get("paths", {}).get("processed_dir"))
        self.ctx.logger.info("Models dir: %s", config.get("paths", {}).get("models_dir"))
        self.ctx.logger.info("Outputs dir: %s", config.get("paths", {}).get("outputs_dir"))
        self.ctx.logger.info("Models to run: %s", ModelFactory.resolve_requested_models(model, config))
        self.ctx.logger.info("No heavy computation was executed.")

    def run(self, stage: str = "all", model: str = "all") -> dict[str, Any]:
        if stage == "all":
            outputs = {}
            for stage_name in ["check-data", "eda", "preprocess", "train", "evaluate", "compare"]:
                outputs[stage_name] = self.run(stage=stage_name, model=model)
            return outputs

        if stage not in self.stages:
            raise ValueError(f"Unknown stage: {stage}")

        if stage in {"train", "evaluate"}:
            return self.stages[stage].run(model=model)
        return self.stages[stage].run()

    def close(self) -> None:
        self.ctx.close()
