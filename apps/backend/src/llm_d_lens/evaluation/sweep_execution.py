"""Candidate materialization through the existing Guide planner and validators."""

import math
from copy import deepcopy

import yaml

from llm_d_lens.configuration.guide_settings import validate_configuration_extensions
from llm_d_lens.configuration.manifest_facts import validate_manifest_facts
from llm_d_lens.configuration.models import SaveRequest
from llm_d_lens.configuration.service import save_configuration
from llm_d_lens.deploy.guide_planning import deployment_contract, plan_documents
from llm_d_lens.deploy.providers.deployment_bundle import text_checksum
from llm_d_lens.infrastructure.artifacts.manifests import configuration_checksum

RUNTIME_FIELDS = {
    "maxNumSeqs": "max-num-seqs",
    "maxNumBatchedTokens": "max-num-batched-tokens",
    "maxModelLen": "max-model-len",
    "gpuMemoryUtilization": "gpu-memory-utilization",
    "blockSize": "block-size",
}


def planning_base(configuration) -> dict:
    content = configuration.content
    official = content.get("officialGuide") or {}
    source = official.get("source") or {}
    if not official.get("renderedManifest") or not source.get("commit"):
        raise ValueError("Sweep requires a published Guide configuration with retained source and manifest")
    settings = dict(content.get("guideSettings") or {})
    documents = [doc for doc in yaml.safe_load_all(official["renderedManifest"]) if isinstance(doc, dict)]
    slice_counts = {doc.get("spec", {}).get("slices", 1) for doc in documents if doc.get("kind") == "DisaggregatedSet"}
    if len(slice_counts) > 1:
        raise ValueError("Search requires one consistent P/D slice count")
    if slice_counts:
        settings.setdefault("slices", next(iter(slice_counts)))
    runtime = content.get("runtime") or {}
    decode = content.get("decode") or content.get("serving") or {}
    prefill = content.get("prefill") or {}
    result = {
        "guide": configuration.provider_ref,
        "model": content["model"]["name"],
        "accelerator": source.get("accelerator"),
        "modelServer": source.get("modelServer"),
        "replicas": decode.get("replicaCount", 1),
        "tensorParallelSize": decode.get("tensorParallelSize", 1),
        "prefillReplicas": prefill.get("replicaCount", 1),
        "prefillTensorParallelSize": prefill.get("tensorParallelSize", 1),
        "runtimeImage": runtime.get("image", ""),
        "guideVariant": content.get("guideVariant", ""),
        "guideSettings": settings,
        "modelSource": runtime.get("modelSource"),
        "modelPath": runtime.get("mountPath", ""),
        "modelPvcClaimName": runtime.get("modelPvcClaimName", ""),
        "customParameters": content.get("customParameters") or [],
    }
    for field, argument in RUNTIME_FIELDS.items():
        entries = [
            item for item in result["customParameters"] if item.get("name") == argument and item.get("target") == "both"
        ]
        if entries:
            result[field] = (
                float(entries[-1]["value"]) if field == "gpuMemoryUtilization" else int(entries[-1]["value"])
            )
        elif field in decode:
            result[field] = decode[field]
    return result


async def materialize_candidate(configuration, candidate, cluster, session_id):
    content = deepcopy(configuration.content)
    official = content["officialGuide"]
    requested = {**planning_base(configuration), **candidate["parameters"]}
    custom_changes = []
    for key, value in candidate["parameters"].items():
        if key.startswith(("argument:", "environment:")):
            kind, target, name = key.split(":", 2)
            if kind == "argument" and name in RUNTIME_FIELDS.values():
                raise ValueError(f"Use the dedicated search parameter for {name}")
            custom_changes.append({"kind": kind, "target": target, "name": name, "value": value})
        elif key.startswith("guide:"):
            requested["guideSettings"] = {**requested["guideSettings"], key.split(":", 1)[1]: value}
    requested.update(decodeReplicas=requested["replicas"], decodeTensorParallelSize=requested["tensorParallelSize"])
    # Typed runtime fields replace their shared custom override; otherwise the
    # original override would silently win over the selected search parameter.
    changed_arguments = {RUNTIME_FIELDS[key] for key in candidate["parameters"] if key in RUNTIME_FIELDS}
    from llm_d_lens.deploy.guide_runtime import runtime_overrides

    original_custom = [
        item
        for item in deepcopy(requested["customParameters"])
        if not any(
            item.get("kind") == change["kind"]
            and item.get("name") == change["name"]
            and (item.get("target") == change["target"] or "both" in {item.get("target"), change["target"]})
            for change in custom_changes
        )
    ] + custom_changes
    dedicated_arguments = {argument for field, argument in RUNTIME_FIELDS.items() if field in requested}
    requested["customParameters"] = [item for item in original_custom if item.get("name") not in dedicated_arguments]
    runtime_overrides(requested)
    documents = [item for item in yaml.safe_load_all(official["renderedManifest"]) if item]
    planned = plan_documents(documents, requested["model"], {}, cluster, requested)
    errors = planned["validation"]["errors"]
    if errors:
        raise ValueError("; ".join(errors))
    manifest = yaml.safe_dump_all(planned["documents"], sort_keys=False)
    for role in ("serving", "decode"):
        if role in content:
            content[role].update(
                replicaCount=int(requested["replicas"]), tensorParallelSize=int(requested["tensorParallelSize"])
            )
    if content.get("prefill"):
        content["prefill"].update(
            replicaCount=int(requested["prefillReplicas"]),
            tensorParallelSize=int(requested["prefillTensorParallelSize"]),
        )
    content["customParameters"] = [item for item in original_custom if item.get("name") not in changed_arguments] + [
        {"target": "both", "kind": "argument", "name": RUNTIME_FIELDS[key], "value": str(value)}
        for key, value in candidate["parameters"].items()
        if key in RUNTIME_FIELDS
    ]
    content["customParameters"] = runtime_overrides({"customParameters": content["customParameters"]})
    for role in ("serving", "decode", "prefill"):
        if content.get(role):
            for key in ("maxModelLen", "maxNumSeqs"):
                if key in candidate["parameters"] and key in content[role]:
                    content[role][key] = candidate["parameters"][key]
    if "maxModelLen" in candidate["parameters"]:
        content["model"]["maxModelLen"] = candidate["parameters"]["maxModelLen"]
    content["guideSettings"] = requested["guideSettings"]
    official.update(
        renderedManifest=manifest,
        manifestChecksum=text_checksum(manifest),
        deployment=deployment_contract(planned["documents"], configuration.provider_ref),
    )
    official["cluster"]["session_id"] = session_id
    if any(
        key
        in {
            "guide:routerValues",
            "guide:cacheLruCapacity",
        }
        for key in candidate["parameters"]
    ) or ("blockSize" in candidate["parameters"] and configuration.provider_ref == "precise-prefix-cache-routing"):
        from llm_d_lens.deploy.guide_bundle import build_guide_deployment_bundle

        official["deploymentBundle"] = await build_guide_deployment_bundle(
            configuration.provider_ref,
            {"source": official["source"]},
            requested["model"],
            requested["accelerator"],
            requested.get("blockSize"),
            (content.get("guideSettings") or {}).get("routerValues", ""),
            content.get("guideSettings") or {},
        )
    validate_manifest_facts(content, manifest)
    validate_configuration_extensions(content, manifest)
    provenance = {
        **configuration.provenance,
        "cluster_ref": official["cluster"],
        "sweep_candidate_id": candidate["id"],
        "sweep_parameters": candidate["parameters"],
    }
    provenance.pop("edited_from_artifact_id", None)
    derived = configuration.model_copy(
        update={"content": content, "checksum": configuration_checksum(content), "provenance": provenance}
    )
    return save_configuration(SaveRequest(deployable_configuration=derived)).artifact


def score_candidate(cases, objective, accelerators, sla_evaluation, scenario_weights=None):
    scores = []
    for case in cases:
        if case.get("status") != "succeeded":
            return None
        metrics = case.get("metrics") or {}
        targets = case.get("sla_targets") or {}
        sla = sla_evaluation(metrics, targets)
        if sla["checks"] and sla["met"] is not True:
            return None
        if objective == "goodput":
            value = metrics.get("slo_goodput_rps")
        elif objective == "latency":
            value = ((metrics.get("latency_distributions") or {}).get("ttft") or {}).get("p99_ms")
            value = -value if isinstance(value, (float, int)) else None
        else:
            value = metrics.get("throughput_tps")
            if objective == "resource_efficiency" and isinstance(value, (float, int)):
                value /= accelerators
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return None
        weight = (scenario_weights or {}).get(case.get("scenario_id"), 1)
        scores.append((value, weight))
    return sum(value * weight for value, weight in scores) / sum(weight for _, weight in scores) if scores else None


def validate_candidate_workload(parameters: dict, plan: dict) -> None:
    limit = parameters.get("maxModelLen")
    if limit is None:
        return
    workloads = [item["benchmark"] for item in plan.get("scenarios", [])] or [plan.get("benchmark", {})]
    for workload in workloads:
        if workload.get("matrix"):
            required = max(point["isl"] + point["osl"] for point in workload["matrix"])
        elif workload.get("shared_prefix"):
            prefix = workload["shared_prefix"]
            turns = prefix["num_prompts_per_group"] if prefix.get("enable_multi_turn_chat") else 1
            required = prefix["system_prompt_len"] + turns * (prefix["question_len"] + prefix["output_len"])
        else:
            raise ValueError("Context-length search requires explicit matrix or shared-prefix request lengths")
        if limit < required:
            raise ValueError(f"Context length {limit} cannot accommodate the fixed workload ({required} tokens)")
