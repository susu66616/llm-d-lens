"""Shared intent and decision-plan contracts for Manual and Agentic deployment."""

from typing import Any, Literal

import math

from pydantic import BaseModel, Field, model_validator


class DeploymentRequirements(BaseModel):
    model: str = Field(min_length=1)
    cluster_id: str | None = None
    objective: Literal["latency", "throughput", "cost", "reliability"] = "throughput"
    hard_constraints: dict[str, Any] = Field(default_factory=dict)
    workload: dict[str, Any] = Field(default_factory=dict)
    performance_targets: dict[str, Any] = Field(default_factory=dict)
    required_features: list[str] = Field(default_factory=list)
    forbidden_features: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_requirements(self):
        if set(self.required_features) & set(self.forbidden_features):
            raise ValueError("A capability cannot be both required and forbidden.")
        for values in (self.hard_constraints, self.workload, self.performance_targets):
            for key, value in values.items():
                if value is None or isinstance(value, (str, bool)):
                    continue
                if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"{key} must be a finite non-negative number.")
        budget = self.hard_constraints.get("maxAccelerators")
        if budget is not None and (isinstance(budget, bool) or not isinstance(budget, int) or budget < 1):
            raise ValueError("maxAccelerators must be a positive whole number.")
        if set(self.hard_constraints) - {"maxAccelerators", "providerRef"}:
            raise ValueError("Unsupported hard constraint; it cannot be silently ignored.")
        provider = self.hard_constraints.get("providerRef")
        if provider is not None and (not isinstance(provider, str) or not provider.strip()):
            raise ValueError("providerRef must be a non-empty provider identifier or null.")
        for key in ("prefixReuse", "prefillHeavy"):
            if self.workload.get(key) is not None and not isinstance(self.workload[key], bool):
                raise ValueError(f"{key} must be true, false or null.")
        for key in ("inputTokens", "p95InputTokens", "outputTokens", "concurrency", "requestRate"):
            value = self.workload.get(key)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0
            ):
                raise ValueError(f"{key} must be a positive number or null.")
        for key, value in self.performance_targets.items():
            if key not in {"ttftMs", "tpotMs", "e2eMs"}:
                raise ValueError(f"Unsupported performance target: {key}.")
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"{key} must be a positive whole number or null.")
        return self


class OptimizationDecision(BaseModel):
    capability_id: str
    label: str
    status: Literal["enabled", "dependency", "excluded", "unknown", "conflict", "user_disabled"]
    reason: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class ConfigurationPlan(BaseModel):
    plan_id: str
    mode: Literal["manual", "agentic"]
    requirements: DeploymentRequirements
    selected_capabilities: list[str] = Field(default_factory=list)
    decisions: list[OptimizationDecision] = Field(default_factory=list)
    resource_estimate: dict[str, Any] = Field(default_factory=dict)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    evidence_level: Literal["recommendation", "predicted", "validated"] = "recommendation"
    validation: dict[str, Any] = Field(default_factory=dict)
    configuration: dict[str, Any] = Field(default_factory=dict)


class PlanRequest(BaseModel):
    mode: Literal["manual", "agentic"]
    requirements: DeploymentRequirements
    candidate_limit: int = Field(default=3, ge=1, le=10)
