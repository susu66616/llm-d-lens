"""Bounded, deterministic candidate generation and feedback-driven selection."""

import json
import math
import random
import re
from hashlib import sha256
from itertools import product

from .sweep_models import SweepDefinition

# Planner input names are the wire contract, not hardware-specific defaults.
PARAMETERS = {
    "tensorParallelSize": (True, "topology"),
    "replicas": (True, "topology"),
    "prefillTensorParallelSize": (True, "topology"),
    "prefillReplicas": (True, "topology"),
    "maxNumSeqs": (True, "runtime"),
    "maxNumBatchedTokens": (True, "runtime"),
    "maxModelLen": (True, "runtime"),
    "gpuMemoryUtilization": (False, "runtime"),
    "blockSize": (True, "runtime"),
}
ALGORITHM_VERSION = "bounded-diversity-v1"


def parameter_info(name: str) -> tuple[bool, str]:
    if name in PARAMETERS:
        return PARAMETERS[name]
    if re.fullmatch(r"(?:argument|environment):(?:both|prefill|decode):[A-Za-z_][A-Za-z0-9_-]*", name):
        return False, "custom"
    if name in {
        "guide:cacheCpuGiB",
        "guide:rdmaNicCount",
        "guide:slices",
        "guide:cacheLruCapacity",
        "guide:routerValues",
    }:
        return name in {
            "guide:rdmaNicCount",
            "guide:slices",
            "guide:cacheLruCapacity",
        }, "guide"
    raise ValueError(f"Unsupported search parameter: {name}")


def parameter_distance(value, reference):
    if isinstance(value, bool) or isinstance(reference, bool):
        return float(value != reference)
    try:
        return abs(float(value) - float(reference)) / max(1, abs(float(reference)))
    except (ValueError, TypeError):
        return float(value != reference)


def candidate_id(parameters: dict) -> str:
    return sha256(json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]


def generate_candidates(definition: SweepDefinition, base: dict) -> dict:
    for name in definition.dimensions:
        parameter_info(name)
    pd = base.get("guide") == "pd-disaggregation"
    if not pd and any((key.startswith("prefill") or key == "guide:slices") for key in definition.dimensions):
        raise ValueError("Prefill search parameters require a PD configuration")
    if base.get("modelServer") != "vllm" and any(parameter_info(key)[1] == "runtime" for key in definition.dimensions):
        raise ValueError("This runtime has not declared support for these search parameters")
    names = sorted(definition.dimensions)
    domains = [
        definition.dimensions[name].points(
            integer=parameter_info(name)[0], grid=definition.strategy == "grid" or parameter_info(name)[1] == "topology"
        )
        for name in names
    ]
    for name, values in zip(names, domains, strict=True):
        for value in values:
            if parameter_info(name)[1] == "custom" or name in {"guide:routerValues"}:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} requires positive finite numeric values")
            if parameter_info(name)[0] and int(value) != value:
                raise ValueError(f"{name} requires integer values")
            if name == "gpuMemoryUtilization" and value >= 1:
                raise ValueError("GPU memory utilization must be less than 1")
    size = math.prod(map(len, domains))
    if size > 10_000 and definition.strategy != "specified":
        raise ValueError("Search space exceeds 10000 sampled combinations; narrow the ranges")
    points = (
        definition.specified_candidates
        if definition.strategy == "specified"
        else (dict(zip(names, values, strict=True)) for values in product(*domains))
    )
    candidates, exclusions, seen = [], [], set()
    for point in points:
        if set(point) != set(names):
            raise ValueError("Specified candidates must set exactly the selected dimensions")
        if definition.strategy == "specified":
            for name, value in point.items():
                dimension = definition.dimensions[name]
                if dimension.values is not None:
                    valid = value in dimension.values
                else:
                    valid = isinstance(value, (int, float)) and dimension.minimum <= value <= dimension.maximum
                if not valid:
                    raise ValueError(f"Specified {name} is outside its search domain")
        for name, value in point.items():
            if parameter_info(name)[1] == "custom" or name in {"guide:routerValues"}:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} requires a positive finite number")
            if parameter_info(name)[0] and int(value) != value:
                raise ValueError(f"{name} requires an integer")
            if name == "gpuMemoryUtilization" and value >= 1:
                raise ValueError("GPU memory utilization must be less than 1")
        values = {**base, **point}
        cards = int(values.get("replicas", 1)) * int(values.get("tensorParallelSize", 1))
        if pd:
            cards += int(values.get("prefillReplicas", 1)) * int(values.get("prefillTensorParallelSize", 1))
        if pd:
            cards *= int(values.get("guide:slices", (base.get("guideSettings") or {}).get("slices", 1)))
        reason = None
        if cards > definition.accelerator_budget:
            reason = "exceeds accelerator budget"
        elif definition.resource_mode == "fixed" and cards != definition.accelerator_budget:
            reason = "does not match fixed accelerator budget"
        identifier = candidate_id(point)
        if identifier in seen:
            continue
        seen.add(identifier)
        item = {
            "id": identifier,
            "parameters": point,
            "accelerator_count": cards,
            "status": "pending",
            "base_distance": sum(parameter_distance(value, base.get(key, value)) for key, value in point.items()),
            "runtime_distance": sum(
                parameter_distance(value, base.get(key, value))
                for key, value in point.items()
                if parameter_info(key)[1] == "runtime"
            ),
            "topology": {key: value for key, value in point.items() if parameter_info(key)[1] == "topology"},
        }
        if reason:
            exclusions.append({**item, "status": "excluded", "reason": reason})
        else:
            candidates.append(item)
    if not candidates:
        raise ValueError("No candidates satisfy the resource constraints")
    if definition.strategy == "grid" and len(candidates) > definition.max_trials:
        raise ValueError(f"Complete grid needs {len(candidates)} trials; increase the budget or narrow the ranges")
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "seed": definition.seed,
        "candidates": candidates,
        "excluded": exclusions,
        "sampled_combinations": len(seen),
        "compatibility": "Candidates require provider planning validation before deployment",
    }


def select_candidate(definition: SweepDefinition, candidates: list[dict]) -> dict | None:
    pending = [item for item in candidates if item["status"] == "pending"]
    completed = [item for item in candidates if item["status"] in {"succeeded", "failed"}]
    if len(completed) >= definition.max_trials:
        return None
    reserve = min(definition.confirmation_trials, definition.max_trials // 3) if definition.strategy == "auto" else 0
    leaders = [item for item in completed if item.get("score") is not None and not item.get("confirmation_of")]
    if reserve and leaders and (len(completed) >= definition.max_trials - reserve or not pending):
        confirmations = [item for item in candidates if item.get("confirmation_of")]
        unfinished = next((item for item in confirmations if item["status"] == "pending"), None)
        if unfinished:
            return unfinished
        if len(confirmations) < reserve:
            leader = max(leaders, key=lambda item: item["score"])
            repeated = {
                "id": f"{leader['id']}-repeat-{len(confirmations) + 1}",
                "parameters": dict(leader["parameters"]),
                "accelerator_count": leader["accelerator_count"],
                "confirmation_of": leader["id"],
                "status": "pending",
            }
            candidates.append(repeated)
            return repeated
        return None
    if not pending:
        return None
    if definition.strategy != "auto":
        return pending[0]
    rng = random.Random(definition.seed + len(completed))  # noqa: S311 -- reproducible experiment sampling
    ordered = sorted(pending, key=lambda item: item["id"])
    if not completed:
        return min(ordered, key=lambda item: item.get("base_distance", 0))
    topology_trials = min(
        max(2, definition.max_trials // 3), len({candidate_id(item.get("topology", {})) for item in candidates})
    )
    if len(completed) < topology_trials:
        tested = {candidate_id(item.get("topology", {})) for item in completed}
        fresh = [item for item in ordered if candidate_id(item.get("topology", {})) not in tested]
        if fresh:
            rng.shuffle(fresh)
            return min(fresh, key=lambda item: item.get("runtime_distance", 0))
    scored = [item for item in completed if item.get("score") is not None]
    # Retain exploration every third trial instead of eliminating unmeasured regions.
    if not scored or len(completed) % 3 == 0:
        return rng.choice(ordered)
    leaders = sorted(scored, key=lambda item: item["score"], reverse=True)[:3]

    def distance(item):
        return min(
            sum(parameter_distance(item["parameters"][key], leader["parameters"][key]) for key in item["parameters"])
            for leader in leaders
        )

    return min(ordered, key=distance)
