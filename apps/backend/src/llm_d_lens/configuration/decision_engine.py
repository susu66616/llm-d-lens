"""Deterministic capability selection shared by Manual and Agentic planning."""

from __future__ import annotations

from hashlib import sha256
from math import floor
from typing import Any

from .plan_models import ConfigurationPlan, DeploymentRequirements, OptimizationDecision, PlanRequest


CAPABILITIES = (
    ("prefix_cache", "Prefix reuse cache", "prefix_reuse", "Requires a known repeated-prefix workload."),
    ("chunked_prefill", "Chunked prefill", "long_context", "Useful for long inputs; runtime support must be present."),
    ("pd_disaggregation", "Prefill / Decode separation", "prefill_heavy", "Requires compatible transport and separate role resources."),
    ("external_kv_cache", "External KV cache", "long_context", "Consumes additional CPU/network capacity."),
    ("autoscaling", "Automatic scaling", "request_rate", "Requires a request-rate signal to size scaling bounds."),
)


def _workload_signals(requirements: DeploymentRequirements) -> dict[str, bool]:
    """Normalize user intent into the signals used by the rule planner.

    The planner deliberately consumes intent, rather than rendering fields. The
    UI may call these values inputTokens/requestRate while the decision rules
    reason about long context, repeated prefixes and prefill-heavy traffic.
    """
    workload = requirements.workload
    input_tokens = workload.get("inputTokens", workload.get("input_tokens"))
    request_rate = workload.get("requestRate", workload.get("request_rate"))
    use_case = str(workload.get("useCase", workload.get("use_case", ""))).lower()
    return {
        "prefix_reuse": bool(workload.get("prefixReuse", workload.get("prefix_reuse")))
        or "prefix_cache" in requirements.required_features
        or "prefix_reuse" in requirements.required_features,
        "long_context": bool(workload.get("longContext", workload.get("long_context")))
        or (isinstance(input_tokens, (int, float)) and input_tokens >= 4096)
        or use_case in {"long-inputs", "summarization"},
        "prefill_heavy": bool(workload.get("prefillHeavy", workload.get("prefill_heavy")))
        or use_case in {"long-inputs", "summarization"},
        "request_rate": bool(request_rate),
    }


def _plan_id(request: PlanRequest) -> str:
    digest = sha256(request.model_dump_json().encode()).hexdigest()[:12]
    return f"{request.mode}-{digest}"


def _decisions(requirements: DeploymentRequirements, mode: str) -> tuple[list[str], list[OptimizationDecision]]:
    workload = requirements.workload
    signals = _workload_signals(requirements)
    required = set(requirements.required_features)
    forbidden = set(requirements.forbidden_features)
    selected: list[str] = []
    decisions: list[OptimizationDecision] = []
    for capability_id, label, signal, explanation in CAPABILITIES:
        explicitly_required = capability_id in required
        explicitly_forbidden = capability_id in forbidden
        known = signals.get(signal, bool(workload.get(signal)))
        if explicitly_forbidden:
            status = "user_disabled"
            reason = "Explicitly disabled by the user."
        elif explicitly_required and not known:
            status = "unknown"
            reason = f"Requested, but the prerequisite signal {signal!r} is unknown."
        elif explicitly_required or (mode == "manual" and known):
            status = "enabled"
            reason = "Prerequisites are present and the capability fits the requested mode."
            selected.append(capability_id)
        elif known:
            status = "excluded" if mode == "agentic" else "enabled"
            reason = "Agentic keeps it as an alternative until its expected benefit is compared." if mode == "agentic" else "Automatically enabled because the prerequisite signal is present."
            if status == "enabled":
                selected.append(capability_id)
        else:
            status = "unknown"
            reason = explanation
        decisions.append(OptimizationDecision(capability_id=capability_id, label=label, status=status, reason=reason))
    return selected, decisions


def build_plan(request: PlanRequest) -> ConfigurationPlan:
    _, decisions = _decisions(request.requirements, request.mode)
    # This compatibility endpoint has no live facts or rendered resources.
    # Intent alone must never attest that a mechanism has been enabled.
    decisions = [
        item.model_copy(update={
            "status": "unknown",
            "reason": "Requested or applicable by workload; live provider, dependency and resource validation is required.",
        }) if item.status == "enabled" else item
        for item in decisions
    ]
    budget = request.requirements.hard_constraints.get("maxAccelerators")
    workload = request.requirements.workload
    configuration = {
        "objective": request.requirements.objective,
        "capabilities": [
            {
                "id": item.capability_id,
                "label": item.label,
                "status": item.status,
                "reason": item.reason,
                "parameters": item.parameters,
            }
            for item in decisions
        ],
        "workload": workload,
        "performance_targets": request.requirements.performance_targets,
    }
    return ConfigurationPlan(
        plan_id=_plan_id(request), mode=request.mode, requirements=request.requirements,
        selected_capabilities=[], decisions=decisions,
        resource_estimate={"max_accelerators": budget, "status": "unknown"},
        candidates=[], evidence_level="recommendation",
        validation={"status": "pending", "checks": ["capability prerequisites", "conflicts", "resource budget"]},
        configuration=configuration,
    )


PROVIDER_MECHANISMS = {
    "optimized-baseline": set(),
    "pd-disaggregation": {"pd_disaggregation"},
    "precise-prefix-cache-routing": {"prefix_routing"},
    "tiered-prefix-cache": {"external_kv_cache"},
}


def candidate_mechanisms(candidate) -> set[str]:
    mechanisms = set(PROVIDER_MECHANISMS.get(candidate.provider_ref, ()))
    if candidate.native_cpu_cache_gib:
        mechanisms.add("external_kv_cache")
    return mechanisms


def manual_combinations(candidates, facts, requirements):
    """Expand only the CPU-cache adapters owned by the source-bound renderer."""
    output = list(candidates)
    if requirements.workload.get("prefixReuse") is not True:
        return output
    for index, candidate in enumerate(candidates):
        replicas = candidate.replicas + (candidate.prefill_replicas or 0)
        capacity = floor(min(8.0, facts.cpu_buffer_gib * 0.25 / replicas) * 1000) / 1000
        if candidate.provider_ref == "tiered-prefix-cache":
            output[index] = candidate.model_copy(update={
                "native_cpu_cache_gib": capacity if capacity >= 1 else None,
                "deployable": candidate.deployable and capacity >= 1,
                "rejection_reasons": [
                    *candidate.rejection_reasons,
                    *([] if capacity >= 1 else ["No discovered host-memory budget for CPU cache."]),
                ],
                "historical_benchmarks": [], "performance_estimate": None,
                "performance_estimate_source": None, "score": None, "score_source": None,
                "slo_status": "unknown", "slo_confidence": 0,
            })
        if (not candidate.deployable or candidate.native_cpu_cache_gib
                or candidate.provider_ref not in {"optimized-baseline", "precise-prefix-cache-routing", "pd-disaggregation"}):
            continue
        # Reserve at least 75% of observed free host memory for serving and other
        # workloads. Source rendering still checks per-node placement and limits.
        if capacity < 1:
            continue
        output.append(candidate.model_copy(update={
            "id": f"{candidate.id}-cpu-cache",
            "native_cpu_cache_gib": capacity,
            "historical_benchmarks": [], "performance_estimate": None,
            "performance_estimate_source": None, "score": None, "score_source": None,
            "slo_status": "unknown", "slo_confidence": 0,
        }))
    return output


def rank_manual_candidates(candidates, requirements):
    """Prefer established coverage, then an explicit, non-measured objective policy."""
    def key(candidate):
        mechanisms = candidate_mechanisms(candidate)
        if requirements.objective == "cost":
            preference = (candidate.required_gpus,)
        elif requirements.objective == "latency":
            preference = (int("prefix_routing" not in mechanisms), candidate.tensor_parallel_size)
        elif requirements.objective == "reliability":
            preference = (int(candidate.replicas < 2), candidate.required_gpus)
        else:
            preference = (int("pd_disaggregation" not in mechanisms), int("prefix_routing" not in mechanisms))
        return (
            {"satisfied": 0, "unknown": 1, "not_satisfied": 2}[candidate.slo_status],
            -len(mechanisms), *preference, candidate.required_gpus, candidate.tensor_parallel_size, candidate.id,
        )
    return sorted((item for item in candidates if item.deployable), key=key)


def candidate_requirement_errors(requirements, candidate) -> list[str]:
    """Filter actual provider configurations, not a second speculative candidate list."""
    if requirements is None:
        return []
    mechanisms = candidate_mechanisms(candidate)
    errors = []
    provider = requirements.hard_constraints.get("providerRef")
    if provider is not None and candidate.provider_ref != provider:
        errors.append(f"Provider {provider} was explicitly selected; this candidate uses {candidate.provider_ref}.")
    budget = requirements.hard_constraints.get("maxAccelerators")
    if budget is not None and candidate.required_gpus > budget:
        errors.append(f"Requires {candidate.required_gpus} accelerators; the hard limit is {budget}.")
    for capability in requirements.forbidden_features:
        if capability in mechanisms:
            errors.append(f"{capability} is forbidden by the user.")
    for capability in requirements.required_features:
        if capability not in mechanisms:
            errors.append(f"{capability} is required but is not established by this provider configuration.")
    if "prefix_routing" in mechanisms and not requirements.workload.get("prefixReuse"):
        errors.append("Prefix-aware routing requires a known repeated-prefix workload.")
    if "external_kv_cache" in mechanisms and not requirements.workload.get("prefixReuse"):
        errors.append("External KV cache requires a known reuse workload.")
    if "pd_disaggregation" in mechanisms and not requirements.workload.get("prefillHeavy"):
        errors.append("P/D applicability requires a prefill-heavy workload or an explicit workload preset.")
    return errors


def resolved_plan(run, configuration: dict) -> ConfigurationPlan:
    selected = next(item for item in run.candidates if item.id == run.selected_candidate.id)
    mechanisms = candidate_mechanisms(selected)
    requirements = run.request.requirements or DeploymentRequirements(model=run.request.model)
    labels = [
        ("pd_disaggregation", "Prefill / Decode separation"),
        ("prefix_routing", "Prefix-aware routing"),
        ("external_kv_cache", "External KV cache"),
        ("prefix_cache", "Engine prefix cache"),
        ("chunked_prefill", "Chunked prefill"),
        ("autoscaling", "Automatic scaling"),
        ("speculative_decoding", "Speculative decoding"),
    ]
    decisions = []
    for capability, label in labels:
        rejected = [
            reason for item in run.rejected_candidates if capability in candidate_mechanisms(item)
            for reason in item.rejection_reasons
        ]
        if capability in requirements.forbidden_features:
            status, reason = "user_disabled", "Excluded by the user's hard capability constraint."
        elif capability in mechanisms:
            if capability == "external_kv_cache" and selected.native_cpu_cache_gib:
                status, reason = "unknown", (
                    "Planned within the discovered host-memory budget. The pinned source connector, runtime image "
                    "and per-node resource compatibility must pass rendering before this mechanism is enabled."
                )
            else:
                status, reason = "enabled", "Included in the selected provider configuration; runtime readiness is still pending."
        elif any(capability in candidate_mechanisms(item) for item in run.candidates):
            status, reason = "conflict", (
                f"Available in another planning-compatible combination. The explicit {requirements.objective}-first "
                "rule selected the displayed recipe; no source-verified overlay joins these recipes."
            )
        elif capability in {"pd_disaggregation", "prefix_routing", "external_kv_cache"}:
            signal = "prefillHeavy" if capability == "pd_disaggregation" else "prefixReuse"
            if requirements.workload.get(signal) is None:
                status, reason = "unknown", f"Missing workload evidence: {signal}. No applicability was assumed."
            elif requirements.workload.get(signal) is False:
                status, reason = "excluded", f"Workload prerequisite {signal} is not satisfied."
            elif rejected:
                status, reason = "excluded", "; ".join(dict.fromkeys(rejected))
            else:
                status, reason = "unknown", "No compatible provider or discovered resource budget established this mechanism."
        else:
            status, reason = "unknown", (
                "Pinned source runtime flags must be inspected during resource generation."
                if capability in {"prefix_cache", "chunked_prefill"}
                else "No source-verified planner adapter establishes software support, dependencies and resource sizing."
            )
        parameters = {}
        if capability == "external_kv_cache" and selected.native_cpu_cache_gib:
            parameters = {"cpu_cache_gib_per_replica": selected.native_cpu_cache_gib}
        decisions.append(OptimizationDecision(
            capability_id=capability, label=label, status=status, reason=reason, parameters=parameters,
        ))
    return ConfigurationPlan(
        plan_id=f"{run.id}:{run.revision}",
        mode=run.request.planning_mode,
        requirements=requirements,
        selected_capabilities=sorted(item.capability_id for item in decisions if item.status == "enabled"),
        decisions=decisions,
        resource_estimate={
            "accelerators": selected.required_gpus, "status": "estimated",
            "cpu_cache_gib": (selected.native_cpu_cache_gib or 0) * (selected.replicas + (selected.prefill_replicas or 0)),
        },
        candidates=[item.model_dump(mode="json") for item in run.candidates],
        validation={
            "status": "planning_checked",
            "checks": ["model memory", "accelerator capacity", "candidate requirements"],
            "pending": ["rendered manifests", "cluster admission", "runtime readiness", "workload performance"],
        },
        configuration=configuration,
    )
