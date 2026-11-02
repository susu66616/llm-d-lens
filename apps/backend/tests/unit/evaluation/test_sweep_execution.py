"""Search scores must reflect measured evidence and the actual resource budget."""

from types import SimpleNamespace

import pytest

from llm_d_lens.evaluation.sweep_execution import planning_base, score_candidate


def sla(metrics, targets):
    return {"checks": {"latency": {}} if targets else {}, "met": metrics.get("sla_met")}


def test_missing_goodput_does_not_fall_back_to_raw_throughput():
    cases = [{"status": "succeeded", "metrics": {"throughput_tps": 1000}}]
    assert score_candidate(cases, "goodput", 4, sla) is None
    assert score_candidate(cases, "resource_efficiency", 4, sla) == 250


def test_all_scenarios_must_pass_requested_slo():
    cases = [
        {"status": "succeeded", "metrics": {"slo_goodput_rps": 10, "sla_met": False}, "sla_targets": {"ttft_ms": 100}}
    ]
    assert score_candidate(cases, "goodput", 1, sla) is None
    cases[0]["metrics"]["sla_met"] = True
    assert score_candidate(cases, "goodput", 1, sla) == 10


def test_planning_uses_published_replica_count_and_rejects_unreproducible_inputs():
    content = {
        "model": {"name": "model"},
        "decode": {"replicaCount": 3, "tensorParallelSize": 2},
        "officialGuide": {"source": {"commit": "a" * 40}, "renderedManifest": "kind: Deployment"},
    }
    config = SimpleNamespace(content=content, provider_ref="optimized-baseline")
    assert planning_base(config)["replicas"] == 3
    content["officialGuide"].pop("renderedManifest")
    with pytest.raises(ValueError, match="retained source"):
        planning_base(config)


def test_nonfinite_and_failed_results_cannot_win():
    assert (
        score_candidate([{"status": "succeeded", "metrics": {"slo_goodput_rps": float("nan")}}], "goodput", 1, sla)
        is None
    )
    assert score_candidate([{"status": "failed", "metrics": {"slo_goodput_rps": 99}}], "goodput", 1, sla) is None


@pytest.mark.asyncio
async def test_materialization_replaces_runtime_override_and_preserves_source(monkeypatch):
    from llm_d_lens.deploy.contracts import DeployableConfiguration
    from llm_d_lens.deploy.guide_runtime import runtime_overrides
    from llm_d_lens.evaluation import sweep_execution as execution
    from llm_d_lens.infrastructure.artifacts.manifests import configuration_checksum

    content = {
        "model": {"name": "model"},
        "decode": {"replicaCount": 1, "tensorParallelSize": 1},
        "customParameters": [{"target": "both", "kind": "argument", "name": "max-num-seqs", "value": "64"}],
        "officialGuide": {
            "source": {"commit": "a" * 40, "modelServer": "vllm"},
            "cluster": {"session_id": "old"},
            "renderedManifest": "kind: Deployment",
        },
    }
    configuration = DeployableConfiguration(
        type="baseline",
        format="manifest",
        provider_ref="optimized-baseline",
        content=content,
        checksum=configuration_checksum(content),
    )

    def plan(documents, model, machine, cluster, requested):
        overrides = runtime_overrides(requested)
        assert [item["value"] for item in overrides if item["name"] == "max-num-seqs"] == ["128"]
        return {
            "documents": [{"kind": "Deployment", "spec": {"replicas": requested["replicas"]}}],
            "validation": {"errors": []},
        }

    monkeypatch.setattr(execution, "plan_documents", plan)
    monkeypatch.setattr(execution, "deployment_contract", lambda *_: {})
    monkeypatch.setattr(execution, "validate_manifest_facts", lambda *_: None)
    monkeypatch.setattr(execution, "validate_configuration_extensions", lambda *_: None)
    monkeypatch.setattr(
        execution, "save_configuration", lambda request: SimpleNamespace(artifact=request.deployable_configuration)
    )
    result = await execution.materialize_candidate(
        configuration, {"id": "candidate", "parameters": {"replicas": 2, "maxNumSeqs": 128}}, {}, "new"
    )
    assert result.content["decode"]["replicaCount"] == 2
    assert result.content["officialGuide"]["cluster"]["session_id"] == "new"
    assert result.checksum == configuration_checksum(result.content)
    assert configuration.content == content
    assert result.content["officialGuide"]["source"]["commit"] == "a" * 40


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["normal", "pause", "cleanup_failure"])
async def test_search_isolates_trial_failure_and_preserves_distinct_case_ids(monkeypatch, mode):
    import importlib
    from copy import deepcopy
    from uuid import uuid4

    from llm_d_lens.deploy.contracts import DeployableConfiguration
    from llm_d_lens.evaluation import sweep_execution
    from llm_d_lens.evaluation.sweep import generate_candidates
    from llm_d_lens.evaluation.sweep_models import SweepDefinition
    from llm_d_lens.infrastructure.artifacts.manifests import configuration_checksum

    router = importlib.import_module("llm_d_lens.evaluation.router")
    content = {"model": {"name": "model"}, "decode": {"replicaCount": 1, "tensorParallelSize": 1}}
    config = DeployableConfiguration(
        type="baseline",
        format="manifest",
        provider_ref="optimized-baseline",
        content=content,
        checksum=configuration_checksum(content),
    )
    spec = SweepDefinition(
        dimensions={"replicas": {"minimum": 1, "maximum": 2}},
        accelerator_budget=2,
        max_trials=2,
        strategy="grid",
        objective="throughput",
    )
    state = generate_candidates(
        spec, {"guide": "optimized-baseline", "modelServer": "vllm", "replicas": 1, "tensorParallelSize": 1}
    )
    state.update(
        definition=spec.model_dump(),
        base_configuration=config.model_dump(),
        cluster={},
        base_plan={"id": "base", "configuration_artifact_id": str(uuid4()), "include_baseline": False},
    )
    state["pause_requested"] = mode == "pause"
    workflow = {
        "id": str(uuid4()),
        "kind": "workflow",
        "status": "queued",
        "cluster_session_id": str(uuid4()),
        "sweep": state,
        "configuration_artifacts": {},
    }
    records = {workflow["id"]: deepcopy(workflow)}
    monkeypatch.setattr(router, "_get", lambda kind, identifier: deepcopy(records[identifier]))
    monkeypatch.setattr(router, "_save", lambda value: records.update({value["id"]: deepcopy(value)}))
    monkeypatch.setattr(router, "_comparison_report", lambda _: {})

    async def materialize(*args):
        return SimpleNamespace(artifact_id=str(uuid4()), deployable_configuration=config)

    monkeypatch.setattr(sweep_execution, "materialize_candidate", materialize)
    monkeypatch.setattr(
        router, "_evaluation_cases", lambda *_: [{"id": "guide-1", "status": "queued", "kind": "guide"}]
    )
    calls = []

    async def execute(identifier):
        current = records[identifier]
        active = current["sweep"]["active_candidate_id"]
        calls.append(active)
        for case in current["cases"]:
            if case["sweep_candidate_id"] == active:
                case.update(status="failed" if len(calls) == 1 else "succeeded", metrics={"throughput_tps": 20})
                if mode == "cleanup_failure":
                    case["deployment_run_id"] = "deployment"

    monkeypatch.setattr(router, "_execute_evaluation_cases", execute)
    if mode == "cleanup_failure":
        from unittest.mock import AsyncMock

        worker = SimpleNamespace(cancel_evaluation_deployment=AsyncMock(side_effect=ValueError("cleanup failed")))
        monkeypatch.setattr(router, "_store", SimpleNamespace(get_run=lambda _: SimpleNamespace(cases=[])))
        monkeypatch.setattr(
            router,
            "deployment_run_manager",
            SimpleNamespace(is_run_active=lambda _: True, worker_for_run=lambda _: worker),
        )
    await router._execute_evaluation(workflow["id"])
    result = records[workflow["id"]]
    if mode == "pause":
        assert result["status"] == "paused"
        assert not calls
        return
    if mode == "cleanup_failure":
        assert result["status"] == "paused"
        assert result["sweep"]["cleanup_pending"] == calls[0]
        assert len(calls) == 1
        return
    assert len(calls) == 2
    assert len({case["id"] for case in result["cases"]}) == 2
    assert [item["status"] for item in result["sweep"]["candidates"]] == ["failed", "succeeded"]
    assert result["sweep"]["best_candidate_id"] == calls[1]
    assert result["status"] == "succeeded"
