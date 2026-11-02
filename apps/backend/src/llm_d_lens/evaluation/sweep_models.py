"""Validated, reproducible deployment search definitions."""

import math
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SweepDimension(BaseModel):
    model_config = ConfigDict(extra="forbid")
    minimum: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    maximum: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    step: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    values: list[float | str | bool] | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def valid_domain(self):
        if self.values is not None:
            if any(isinstance(value, (int, float)) and not math.isfinite(value) for value in self.values):
                raise ValueError("Search choices must be finite")
            if any(value is not None for value in (self.minimum, self.maximum, self.step)):
                raise ValueError("Use a range or explicit values, not both")
        elif self.minimum is None or self.maximum is None or self.maximum < self.minimum:
            raise ValueError("A range requires minimum <= maximum")
        return self

    def points(self, *, integer: bool, grid: bool) -> list:
        if self.values is not None:
            return list(dict.fromkeys(self.values))
        low, high = Decimal(str(self.minimum)), Decimal(str(self.maximum))
        if integer and (low != int(low) or high != int(high)):
            raise ValueError("Integer parameters require integer range bounds")
        step = Decimal(str(self.step or (1 if integer else 0.01)))
        if integer and step != int(step):
            raise ValueError("Integer parameters require an integer step")
        count = int((high - low) // step) + 1
        if grid and count > 128:
            raise ValueError("A grid dimension exceeds 128 points; increase its step")
        indices = (
            range(count)
            if grid or count <= 7
            else sorted({0, count - 1, *(round(i * (count - 1) / 6) for i in range(7))})
        )
        return [int(low + i * step) if integer else float(low + i * step) for i in indices]


class SweepDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy: Literal["auto", "grid", "specified"] = "auto"
    dimensions: dict[str, SweepDimension] = Field(min_length=1, max_length=12)
    resource_mode: Literal["maximum", "fixed"] = "maximum"
    accelerator_budget: int = Field(ge=1, le=4096)
    max_trials: int = Field(default=12, ge=1, le=128)
    max_duration_seconds: int = Field(default=14400, ge=60, le=604800)
    objective: Literal["goodput", "throughput", "latency", "resource_efficiency"] = "goodput"
    confirmation_trials: int = Field(default=2, ge=0, le=10)
    scenario_weights: dict[str, float] = Field(default_factory=dict)
    seed: int = Field(default=0, ge=0)
    patience: int | None = Field(default=None, ge=2, le=128)
    specified_candidates: list[dict[str, float | str | bool]] = Field(default_factory=list, max_length=128)

    @model_validator(mode="after")
    def specified_mode(self):
        if any(not math.isfinite(value) or value <= 0 for value in self.scenario_weights.values()):
            raise ValueError("Scenario weights must be finite and positive")
        if (self.strategy == "specified") != bool(self.specified_candidates):
            raise ValueError("Specified search requires explicit candidate configurations")
        return self
