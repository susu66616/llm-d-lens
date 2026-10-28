"""Tests for Agentic Deploy's delegation to existing Prism services."""

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from llm_d_lens.agentic.facts import PlanningEvidence, ResolvedPlanningFacts
from llm_d_lens.agentic.configuration import planning_resource_request
from llm_d_lens.agentic.generator import CandidateProposal
from llm_d_lens.agentic.historical_candidate_seeds import HistoricalCandidateSeed
from llm_d_lens.agentic.models import (
    AgenticCandidateRefinementRequest,
    AgenticDeploymentCreateRequest,
    AgenticDeploymentStatus,
)
from llm_d_lens.agentic.planner import (
    AICCandidatePrediction,
    DeterministicPlanner,
    OpenAICompatiblePlanner,
    OpenAIPlannerSettings,
    PerformanceScoreInput,
    PlanningFacts,
    WorkloadProfile,
)
from llm_d_lens.agentic.planning_snapshot import PlanningSnapshotStore
from llm_d_lens.agentic.service import AgenticDeploymentService

_ORIGINAL_VALIDATE_MODEL_CACHE = AgenticDeploymentService._validate_model_cache


async def _validate_model_cache(*_args, **_kwargs) -> None:
    pass


@pytest.fixture(autouse=True)
def validated_model_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(AgenticDeploymentService, "_validate_model_cache", staticmethod(_validate_model_cache))
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.PlanningSnapshotStore",
        lambda: PlanningSnapshotStore(tmp_path),
    )


@pytest.mark.parametrize("target", ["concurrency", "request_rate"])
def test_load_target_uses_exact_aic_replicas_without_plain_estimates(monkeypatch, target):
    from llm_d_lens.agentic.planner import AICCandidatePrediction, WorkloadProfile

    facts = PlanningFacts(
        8,
        32,
        8,
        workload_profile=WorkloadProfile(**{target: 10}),
        aic_predictions=(AICCandidatePrediction(mode="agg", tensor_parallel_size=1, replicas=3),),
        evidence_ids=("aic:search",),
    )

    async def resolve(*args, **kwargs):
        return SimpleNamespace(facts=facts)

    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", resolve)
    service = AgenticDeploymentService()
    candidates, _, _ = asyncio.run(
        service._generate_candidates(
            None,
            "cluster-a",
            "Qwen/Qwen3-8B",
            facts,
            None,
            provider_requested=False,
        )
    )
    assert len(candidates) == 1
    assert candidates[0].provider_ref == "optimized-baseline"
    assert candidates[0].replicas == 3
    assert candidates[0].required_gpus == 3


def test_joint_load_targets_fail_before_cluster_or_cache_access(monkeypatch):
    from llm_d_lens.agentic.planner import WorkloadProfile

    request = SimpleNamespace(
        planning_facts=PlanningFacts(
            8,
            32,
            8,
            workload_profile=WorkloadProfile(concurrency=10, request_rate=2),
        )
    )
    with pytest.raises(ValueError, match="choose one and clear the other"):
        asyncio.run(AgenticDeploymentService().create(request))


def _request(**overrides) -> AgenticDeploymentCreateRequest:
    overrides.setdefault("storage_volume_id", "storage-model-cache")
    return AgenticDeploymentCreateRequest(
        model="Qwen/Qwen3-8B",
        deployment_name="agentic-qwen3",
        cluster_session_id="a" * 36,
        replicas=1,
        tensor_parallel_size=1,
        planning_facts={"model_weight_gib": 8, "vram_per_gpu_gib": 32, "free_gpu_count": 4},
        **overrides,
    )


@pytest.mark.parametrize("context_length", [40960, 262144, 1048576])
def test_model_card_context_reaches_effective_planning_facts(context_length):
    request = _request(max_model_len=context_length)
    facts = AgenticDeploymentService._planning_only_facts(
        request.planning_facts,
        max_model_len=request.max_model_len,
        gpu_memory_utilization=request.gpu_memory_utilization,
    )

    assert request.planning_facts.context_length == 4096
    assert facts.context_length == context_length


@pytest.mark.parametrize("context_length", [255, 1048577])
def test_deployment_context_rejects_values_outside_supported_range(context_length):
    with pytest.raises(ValidationError, match="max_model_len"):
        _request(max_model_len=context_length)


def _resolved_facts(*, free_gpu_count=4, aic_predictions=()) -> ResolvedPlanningFacts:
    facts = PlanningFacts(
        model_weight_gib=8,
        vram_per_gpu_gib=32,
        free_gpu_count=free_gpu_count,
        aic_predictions=aic_predictions,
    )
    return ResolvedPlanningFacts(
        facts=facts,
        cluster_snapshot={
            "hardware": {
                "availableGpuCount": free_gpu_count,
                "accelerators": [{"id": "nvidia", "gpuCount": free_gpu_count}],
            }
        },
        evidence=[PlanningEvidence(id="cluster-overview:cluster-a", source="cluster-overview", status="available")],
    )


async def _resolve_facts(*_args, **_kwargs) -> ResolvedPlanningFacts:
    return _resolved_facts()


def test_create_generates_an_approval_plan_without_deploying(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_environment",
        lambda: None,
    )

    run = asyncio.run(service.create(_request()))

    assert run.status == AgenticDeploymentStatus.AWAITING_APPROVAL
    assert run.deployment_run_id is None
    assert run.configuration_artifact_id is None
    assert run.selected_candidate.replicas == 1
    assert run.selected_candidate.tensor_parallel_size == 1
    assert run.planning_evidence[0].source == "cluster-overview"
    assert run.decision_metadata is not None
    assert run.decision_metadata.planner == "deterministic"
    assert "Relative rank" in run.decision_metadata.score_method
    assert [item["phase"] for item in run.planning_trace] == [
        "input",
        "generation",
        "candidate_validation",
        "deterministic_ranking",
        "scoring",
        "selection",
    ]
    assert run.planning_trace[-1]["candidate_id"] == run.selected_candidate.id
    assert run.planning_trace[3]["ranking"][0]["candidate_id"] == run.selected_candidate.id


def _shared_request(mode="manual", **requirements):
    return _request(
        planning_mode=mode,
        requirements={
            "model": "Qwen/Qwen3-8B", "cluster_id": "cluster-a",
            "hard_constraints": {"maxAccelerators": 2},
            **requirements,
        },
    )


def _shared_service(monkeypatch):
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)
    return AgenticDeploymentService()


def test_manual_uses_live_candidates_without_calling_ai_and_preserves_unknowns(monkeypatch):
    service = _shared_service(monkeypatch)
    def unexpected_ai(*_args):
        pytest.fail("Manual must not initialize an external AI planner")
    monkeypatch.setattr(OpenAICompatiblePlanner, "from_provider_id", unexpected_ai)
    run = asyncio.run(service.create(_shared_request()))
    assert run.selected_candidate.provider_ref == "optimized-baseline"
    assert all(item.required_gpus <= 2 for item in run.candidates)
    assert run.configuration_plan.mode == "manual"
    assert run.configuration_plan.validation["status"] == "planning_checked"
    assert "rendered manifests" in run.configuration_plan.validation["pending"]
    assert run.configuration_plan.configuration["provider_ref"] == run.selected_candidate.provider_ref
    assert run.configuration_plan.selected_capabilities == []
    assert run.rejected_candidates
    restored = AgenticDeploymentService(snapshot_store=service._snapshot_store).get(run.id)
    assert restored.configuration_plan == run.configuration_plan
    assert restored.status == AgenticDeploymentStatus.AWAITING_APPROVAL


def test_manual_selects_applicable_mechanism_and_enforces_required_budget(monkeypatch):
    service = _shared_service(monkeypatch)
    run = asyncio.run(service.create(_shared_request(workload={"prefillHeavy": True})))
    assert run.selected_candidate.provider_ref == "pd-disaggregation"
    assert run.configuration_plan.selected_capabilities == ["pd_disaggregation"]
    with pytest.raises(ValueError, match="no deployable"):
        asyncio.run(service.create(_shared_request(
            hard_constraints={"maxAccelerators": 1}, workload={"prefillHeavy": True},
            required_features=["pd_disaggregation"],
        )))
    forbidden = asyncio.run(service.create(_shared_request(
        workload={"prefillHeavy": True}, forbidden_features=["pd_disaggregation"],
    )))
    assert all(item.provider_ref != "pd-disaggregation" for item in forbidden.candidates)


@pytest.mark.parametrize("objective,provider", [
    ("throughput", "pd-disaggregation"),
    ("latency", "precise-prefix-cache-routing"),
    ("cost", "precise-prefix-cache-routing"),
])
def test_manual_compares_combinations_with_explicit_objective_and_bounded_cache(monkeypatch, objective, provider):
    service = _shared_service(monkeypatch)

    async def resolve(*_args, **_kwargs):
        resolved = _resolved_facts()
        resolved.facts = replace(resolved.facts, cpu_buffer_gib=32)
        return resolved

    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", resolve)
    run = asyncio.run(service.create(_shared_request(
        objective=objective, workload={"prefillHeavy": True, "prefixReuse": True},
    )))
    assert run.selected_candidate.provider_ref == provider
    assert run.selected_candidate.native_cpu_cache_gib is not None
    assert len({item.provider_ref for item in run.candidates}) >= 2
    for candidate in run.candidates:
        cache = candidate.native_cpu_cache_gib or 0
        assert cache <= 8
        assert cache * (candidate.replicas + (candidate.prefill_replicas or 0)) <= 32 * 0.25
        if cache:
            assert candidate.performance_estimate is None
            assert candidate.slo_status == "unknown"
    assert run.configuration_plan.resource_estimate["cpu_cache_gib"] <= 8
    assert next(item for item in run.configuration_plan.decisions if item.capability_id == "external_kv_cache").status == "unknown"
    assert f"{objective}-first" in run.decision.rationale
    assert run.request.max_num_seqs <= run.selected_candidate.max_concurrent_requests
    body = planning_resource_request(run)
    assert body["guideComposition"]["nativeCpuCacheGiB"] == run.selected_candidate.native_cpu_cache_gib


def test_manual_disabled_and_missing_workload_mechanisms_have_distinct_reasons(monkeypatch):
    service = _shared_service(monkeypatch)
    run = asyncio.run(service.create(_shared_request(
        workload={"prefillHeavy": False}, forbidden_features=["external_kv_cache"],
    )))
    decisions = {item.capability_id: item for item in run.configuration_plan.decisions}
    assert decisions["pd_disaggregation"].status == "excluded"
    assert "not satisfied" in decisions["pd_disaggregation"].reason
    assert decisions["prefix_routing"].status == "unknown"
    assert decisions["external_kv_cache"].status == "user_disabled"
    assert decisions["autoscaling"].status == "unknown"
    assert decisions["speculative_decoding"].status == "unknown"


def test_manual_required_cache_can_compose_with_pd_without_maximizing_parameters(monkeypatch):
    service = _shared_service(monkeypatch)

    async def resolve(*_args, **_kwargs):
        resolved = _resolved_facts()
        resolved.facts = replace(resolved.facts, cpu_buffer_gib=128)
        return resolved

    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", resolve)
    run = asyncio.run(service.create(_shared_request(
        workload={"prefillHeavy": True, "prefixReuse": True},
        required_features=["pd_disaggregation", "external_kv_cache"],
    )))
    assert run.selected_candidate.provider_ref == "pd-disaggregation"
    assert run.selected_candidate.replicas == 1
    assert run.selected_candidate.tensor_parallel_size == 1
    assert run.selected_candidate.native_cpu_cache_gib == 8
    assert run.configuration_plan.configuration["content"]["guideComposition"] == {"nativeCpuCacheGiB": 8}


def test_manual_cache_rounds_down_and_never_uses_client_host_memory(monkeypatch):
    from llm_d_lens.configuration.decision_engine import manual_combinations
    from llm_d_lens.configuration.plan_models import DeploymentRequirements
    facts = PlanningFacts(model_weight_gib=8, vram_per_gpu_gib=32, free_gpu_count=4, cpu_buffer_gib=20)
    candidates = DeterministicPlanner().candidates(facts)
    candidates.append(candidates[0].model_copy(update={"id": "three-replicas", "replicas": 3, "required_gpus": 3}))
    requirements = DeploymentRequirements(model="test/model", workload={"prefixReuse": True})
    for candidate in manual_combinations(candidates, facts, requirements):
        assert (candidate.native_cpu_cache_gib or 0) * (candidate.replicas + (candidate.prefill_replicas or 0)) <= 5
    service = _shared_service(monkeypatch)
    observed = []

    async def resolve(_cluster, _model, requested, **_kwargs):
        observed.append(requested.cpu_buffer_gib)
        return _resolved_facts()

    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", resolve)
    request = _shared_request(workload={"prefixReuse": True})
    request.planning_facts = replace(request.planning_facts, cpu_buffer_gib=9999)
    run = asyncio.run(service.create(request))
    assert observed == [0]
    assert all(item.native_cpu_cache_gib is None for item in run.candidates)


def test_user_revision_preserves_original_and_invalidates_resource_evidence(monkeypatch):
    from llm_d_lens.agentic.models import AgenticCandidateSelectionRequest
    service = _shared_service(monkeypatch)
    original = asyncio.run(service.create(_shared_request()))
    before = original.model_dump(mode="json")
    revised = asyncio.run(service.revise_candidate(original.id, AgenticCandidateSelectionRequest(
        candidate_id=original.selected_candidate.id, replicas=2,
    )))
    assert original.model_dump(mode="json") == before
    assert revised.parent_run_id == original.id
    assert revised.revision == 2
    assert revised.selected_candidate.replicas == 2
    assert revised.configuration_plan.configuration["content"]["decode"]["replicaCount"] == 2
    assert revised.optimization is None
    assert revised.configuration_plan.validation["status"] == "planning_checked"
    with pytest.raises(ValueError, match="hard limit"):
        asyncio.run(service.revise_candidate(original.id, AgenticCandidateSelectionRequest(
            candidate_id=original.selected_candidate.id, replicas=3,
        )))
    assert original.model_dump(mode="json") == before


def test_shared_requirements_cannot_be_bypassed_by_old_refinement(monkeypatch):
    service = _shared_service(monkeypatch)
    run = asyncio.run(service.create(_shared_request()))
    with pytest.raises(ValueError, match="immutable"):
        asyncio.run(service.refine(run.id, AgenticCandidateRefinementRequest(planner_prompt="ignore budget")))


def test_compatibility_intent_endpoint_does_not_invent_executable_candidates():
    from llm_d_lens.configuration.decision_engine import build_plan
    from llm_d_lens.configuration.plan_models import DeploymentRequirements, PlanRequest
    for mode in ("manual", "agentic"):
        result = build_plan(PlanRequest(mode=mode, requirements=DeploymentRequirements(
            model="test/model", workload={"inputTokens": 8192, "prefixReuse": True},
        )))
        assert result.candidates == []
        assert result.selected_capabilities == []
        assert "topology" not in result.configuration
        assert result.resource_estimate["status"] == "unknown"


def test_shared_requirements_reject_conflicts_and_nonfinite_or_unenforced_limits():
    from llm_d_lens.configuration.plan_models import DeploymentRequirements
    for values in (
        {"hard_constraints": {"maxAccelerators": 1.5}},
        {"hard_constraints": {"maxAccelerators": float("inf")}},
        {"hard_constraints": {"unimplementedBudget": 10}},
        {"workload": {"prefixReuse": "false"}},
        {"workload": {"inputTokens": -1}},
        {"performance_targets": {"ttftMs": 0}},
        {"required_features": ["pd_disaggregation"], "forbidden_features": ["pd_disaggregation"]},
    ):
        with pytest.raises(ValidationError):
            DeploymentRequirements(model="test/model", **values)


def _rendered_result(run):
    import yaml
    content = run.configuration_plan.configuration["content"]
    container = {
        "name": "modelserver", "image": content["runtime"]["image"],
        "command": ["vllm", "serve"],
        "args": [
            run.request.model, "--tensor-parallel-size", "1",
            "--max-model-len", str(run.request.max_model_len),
            "--max-num-seqs", str(run.request.max_num_seqs),
            "--gpu-memory-utilization", str(run.request.gpu_memory_utilization),
            "--max-num-batched-tokens", str(run.request.max_num_batched_tokens),
        ],
        "volumeMounts": [{"name": "model-cache", "mountPath": "/model-cache"}],
    }
    manifest = yaml.safe_dump({
        "apiVersion": "apps/v1", "kind": "Deployment",
        "metadata": {"name": "model", "namespace": "__PRISM_NAMESPACE__"},
        "spec": {"replicas": 1, "template": {"spec": {
            "containers": [container],
            "volumes": [{"name": "model-cache", "persistentVolumeClaim": {"claimName": "registered-cache"}}],
        }}},
    })
    return {
        "source": {"guide": run.selected_candidate.provider_ref, "commit": "a" * 40, "modelServer": "vllm", "accelerator": "nvidia"},
        "deployment": {"readinessDeployments": ["model"], "endpoint": {"protocol": "http", "serviceName": "model", "port": 8000}},
        "plannedDeployment": {"content": manifest},
        "validation": {"errors": [], "warnings": ["Readiness requires deployment."]},
    }


def test_reviewed_manifest_is_the_configuration_sent_to_deploy(monkeypatch):
    from llm_d_lens.infrastructure.artifacts.manifests import configuration_checksum
    service = _shared_service(monkeypatch)
    run = asyncio.run(service.create(_shared_request()))
    body = planning_resource_request(run)
    assert body["clusterSessionId"] == run.request.cluster_session_id
    assert body["recommendationGpuCount"] == 2
    assert body["optimizationPolicy"]["prefixReuse"] == "unknown"
    result = _rendered_result(run)
    manifest = result["plannedDeployment"]["content"]
    service.accept_rendered_plan(run, result)
    assert run.configuration_plan.validation["status"] == "resources_checked"
    reviewed = run.configuration_plan.configuration
    assert reviewed["checksum"] == configuration_checksum(reviewed["content"])
    captured = []
    async def start(request, **_kwargs):
        captured.append(request.configurations[0])
        return SimpleNamespace(id="deployment-1")
    monkeypatch.setattr("llm_d_lens.agentic.service.deployment_run_manager.start_run", start)
    approved = asyncio.run(service.approve(run.id))
    assert captured[0].content["officialGuide"]["renderedManifest"] == manifest
    assert captured[0].checksum == reviewed["checksum"]
    assert approved.status == AgenticDeploymentStatus.DEPLOYING
    assert service._snapshot_store.load_run(run.id).deployment_run_id == "deployment-1"


def test_manual_source_comparison_reports_rejected_recipes_and_checks_the_winner(monkeypatch):
    service = _shared_service(monkeypatch)
    original = asyncio.run(service.create(_shared_request(workload={"prefixReuse": True})))
    assert original.selected_candidate.provider_ref == "precise-prefix-cache-routing"
    before = original.model_dump(mode="json")
    checked = []

    async def render(trial):
        checked.append(trial.selected_candidate.provider_ref)
        if trial.selected_candidate.provider_ref == "precise-prefix-cache-routing":
            raise ValueError("Pinned runtime does not support this routing recipe.")
        return _rendered_result(trial)

    winner = asyncio.run(service.compare_manual_resources(original.id, original.selected_candidate.id, render))
    assert set(checked) == {"precise-prefix-cache-routing", "optimized-baseline"}
    assert original.model_dump(mode="json") == before
    assert winner.selected_candidate.provider_ref == "optimized-baseline"
    assert winner.configuration_plan.validation["status"] == "resources_checked"
    assert "1 passed, 1 excluded" in winner.decision.rationale
    assert len(winner.rejected_candidates) > len(original.rejected_candidates)
    decision = next(item for item in winner.configuration_plan.decisions if item.capability_id == "prefix_routing")
    assert decision.status == "excluded"
    assert "Pinned runtime" in decision.reason
    assert service.get(original.id) is winner
    assert service._snapshot_store.load_run(original.id).configuration_plan == winner.configuration_plan


def test_manual_source_comparison_failure_preserves_original_and_does_not_deploy(monkeypatch):
    service = _shared_service(monkeypatch)
    original = asyncio.run(service.create(_shared_request(workload={"prefixReuse": True})))
    before = original.model_dump(mode="json")

    async def render(_trial):
        raise ValueError("Unsupported pinned source.")

    with pytest.raises(ValueError, match="No retained combination passed"):
        asyncio.run(service.compare_manual_resources(original.id, original.selected_candidate.id, render))
    assert service.get(original.id).model_dump(mode="json") == before
    assert service.get(original.id).deployment_run_id is None


def test_manual_source_comparison_rejects_concurrent_selection(monkeypatch):
    service = _shared_service(monkeypatch)
    original = asyncio.run(service.create(_shared_request(workload={"prefixReuse": True})))
    candidate_id = original.selected_candidate.id
    alternative = next(item for item in original.candidates if item.id != candidate_id)

    async def render(trial):
        service.select_candidate(original.id, alternative.id)
        return _rendered_result(trial)

    with pytest.raises(ValueError, match="plan changed during source comparison"):
        asyncio.run(service.compare_manual_resources(original.id, candidate_id, render))
    assert service.get(original.id).selected_candidate.id == alternative.id
    assert service.get(original.id).configuration_plan.validation["status"] == "planning_checked"


def test_rendered_guide_inherited_runtime_flags_keep_their_actual_enabled_state(monkeypatch):
    service = _shared_service(monkeypatch)
    run = asyncio.run(service.create(_shared_request()))
    result = _rendered_result(run)
    result["optimizationDecisions"] = [
        {"id": "prefix", "status": "inherited", "effective_enabled": True, "reason": "Enabled in pinned source."},
        {"id": "chunked", "status": "inherited", "effective_enabled": False, "reason": "Disabled in pinned source."},
    ]
    service.accept_rendered_plan(run, result)
    decisions = {item.capability_id: item.status for item in run.configuration_plan.decisions}
    assert decisions["prefix_cache"] == "enabled"
    assert decisions["chunked_prefill"] == "excluded"


def test_manifest_validation_failure_does_not_replace_the_reviewed_configuration(monkeypatch):
    service = _shared_service(monkeypatch)
    run = asyncio.run(service.create(_shared_request()))
    before = run.configuration_plan.model_dump()
    with pytest.raises(ValueError, match="no modelserver"):
        service.accept_rendered_plan(run, {
            "source": {"guide": run.selected_candidate.provider_ref},
            "deployment": {}, "plannedDeployment": {"content": "kind: Service\nmetadata: {name: service}"},
            "validation": {"errors": []},
        })
    assert run.configuration_plan.model_dump() == before


def test_shared_http_approval_requires_the_current_review_checksum(monkeypatch):
    import importlib
    from fastapi import HTTPException
    from llm_d_lens.agentic.models import AgenticApprovalRequest
    router = importlib.import_module("llm_d_lens.agentic.router")
    service = _shared_service(monkeypatch)
    monkeypatch.setattr(router, "service", service)
    run = asyncio.run(service.create(_shared_request()))
    with pytest.raises(HTTPException) as pending:
        asyncio.run(router.approve_agentic_deployment(run.id))
    assert pending.value.status_code == 409
    run.configuration_plan.validation["status"] = "resources_checked"
    with pytest.raises(HTTPException) as stale:
        asyncio.run(router.approve_agentic_deployment(
            run.id, approval=AgenticApprovalRequest(configuration_checksum="stale"),
        ))
    assert stale.value.status_code == 409
    assert run.deployment_run_id is None


def test_approval_excludes_concurrent_mutations_and_releases_the_lock(monkeypatch):
    service = _shared_service(monkeypatch)
    run = asyncio.run(service.create(_shared_request()))

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        async def approve(*_args, **_kwargs):
            entered.set()
            await release.wait()
            return run
        monkeypatch.setattr(service, "_approve", approve)
        task = asyncio.create_task(service.approve(run.id))
        await entered.wait()
        with pytest.raises(ValueError, match="already in progress"):
            service.select_candidate(run.id, run.selected_candidate.id)
        with pytest.raises(ValueError, match="already in progress"):
            await service.approve(run.id)
        release.set()
        await task
        assert service.select_candidate(run.id, run.selected_candidate.id).id == run.id
    asyncio.run(scenario())


def test_create_preserves_full_prompt_in_append_only_planning_snapshot(monkeypatch, tmp_path):
    service = AgenticDeploymentService(snapshot_store=PlanningSnapshotStore(tmp_path))
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)

    run = asyncio.run(service.create(_request(planner_prompt="persist this complete prompt")))

    assert run.planning_snapshot_id is not None
    assert run.planning_snapshot_path is not None
    payload = json.loads((tmp_path / run.planning_snapshot_path).read_text(encoding="utf-8"))
    assert payload["event"] == "created"
    assert payload["run_id"] == run.id
    assert payload["request"]["planner_prompt_present"] is True
    assert payload["schema_version"] == 2
    assert payload["request"]["planner_prompt"] == "persist this complete prompt"
    assert payload["generator"]["requests"] == []
    assert payload["planner"]["requests"] == []
    assert payload["selected_candidate"]["id"] == run.selected_candidate.id


def test_approve_publishes_configuration_then_starts_existing_deploy(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setenv("PRISM_DEPLOY_ACCELERATOR", "intel-xpu")
    session = SimpleNamespace(id="a" * 36, server_id="cluster-a")
    monkeypatch.setattr("llm_d_lens.agentic.service.require_active_session", lambda _session_id: session)
    planning_lengths = []
    planning_utilizations = []

    async def resolve_facts(_cluster_id, _model, facts, **_kwargs):
        planning_lengths.append(facts.context_length)
        planning_utilizations.append(facts.gpu_memory_utilization)
        resolved = _resolved_facts()
        resolved.facts = replace(
            resolved.facts,
            context_length=facts.context_length,
            gpu_memory_utilization=facts.gpu_memory_utilization,
        )
        return resolved

    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", resolve_facts)
    saved = {}

    async def start_run(request, **_kwargs):
        saved["configuration"] = request.configurations[0]
        saved["deployment"] = request
        return SimpleNamespace(id="deploy-run-1")

    monkeypatch.setattr("llm_d_lens.agentic.service.deployment_run_manager.start_run", start_run)

    async def nodes(*_args, **_kwargs):
        return [{"metadata": {"labels": {"kubernetes.io/hostname": "smc-22"}}}]

    monkeypatch.setattr("llm_d_lens.agentic.service.list_resources", nodes)
    run = asyncio.run(
        service.create(
            _request(
                max_model_len=32768,
                gpu_memory_utilization=0.6,
                placement_replicas="same-node",
                placement_node_label={"key": "kubernetes.io/hostname", "value": "smc-22"},
            )
        )
    )

    approved = asyncio.run(service.approve(run.id))

    assert approved.status == AgenticDeploymentStatus.DEPLOYING
    assert approved.configuration_artifact_id is None
    assert approved.deployment_run_id == "deploy-run-1"
    assert planning_lengths == [32768, 32768]
    assert planning_utilizations == [0.6, 0.6]
    assert run.selected_candidate.gpu_memory_utilization == 0.6
    assert saved["configuration"].content["decode"]["maxModelLen"] == 32768
    assert saved["configuration"].content["runtime"]["image"] == "docker.io/vllm/vllm-openai:v0.30.0"
    assert saved["configuration"].content["hardware_profile"] == "nvidia"
    assert saved["configuration"].content["decode"]["gpuMemoryUtilization"] == 0.6
    assert saved["configuration"].provenance["cluster_ref"]["session_id"] == session.id
    assert saved["deployment"].runtime_binding.placement == {
        "replicas": "same-node",
        "nodeLabel": {"key": "kubernetes.io/hostname", "value": "smc-22"},
    }
    assert saved["deployment"].provenance["agentic_deployment_run_id"] == run.id


def test_select_candidate_only_allows_a_validated_alternative(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)
    run = asyncio.run(service.create(_request()))
    alternative = run.candidates[1]

    updated = service.select_candidate(run.id, alternative.id)

    assert updated.selected_candidate.id == alternative.id
    assert updated.decision is not None
    assert updated.decision.rationale.startswith("Selected by the user")
    with pytest.raises(ValueError, match="not a validated alternative"):
        service.select_candidate(run.id, "not-a-candidate")


def test_approve_carries_model_specific_vllm_arguments_to_deployment(monkeypatch):
    service = AgenticDeploymentService()
    session = SimpleNamespace(id="a" * 36, server_id="cluster-a")
    monkeypatch.setattr("llm_d_lens.agentic.service.require_active_session", lambda _session_id: session)
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)
    saved = {}

    async def start_run(request, **_kwargs):
        saved["configuration"] = request.configurations[0]
        return SimpleNamespace(id="deploy-run-1")

    monkeypatch.setattr("llm_d_lens.agentic.service.deployment_run_manager.start_run", start_run)
    run = asyncio.run(
        service.create(
            _request(
                vllm_arguments=[
                    {"name": "enable-auto-tool-choice", "value": "true"},
                    {"name": "tool-call-parser", "value": "hermes"},
                ]
            )
        )
    )

    asyncio.run(service.approve(run.id))

    arguments = saved["configuration"].content["customParameters"]
    assert {item["name"] for item in arguments} >= {"enable-auto-tool-choice", "tool-call-parser"}


def test_planning_uses_deployment_context_length_and_preserves_workload_signals(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    captured = {}

    async def resolve(_cluster_id, _model, facts, **_kwargs):
        captured["facts"] = facts
        return _resolved_facts(
            aic_predictions=(AICCandidatePrediction(mode="agg", tensor_parallel_size=1, replicas=1),)
        )

    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", resolve)
    request = _request(
        max_model_len=32768,
        gpu_memory_utilization=0.6,
        max_num_seqs=8,
        max_num_batched_tokens=1024,
        vllm_arguments=[{"name": "enforce-eager", "value": "true"}],
    ).model_copy(
        update={
            "planning_facts": PlanningFacts(
                model_weight_gib=8,
                vram_per_gpu_gib=32,
                free_gpu_count=4,
                context_length=4096,
                use_case="long-inputs",
                workload_profile=WorkloadProfile(mean_input_tokens=16384, concurrency=32),
                vllm_arguments=(("enforce-eager", "true"),),
            )
        }
    )
    asyncio.run(service.create(request))

    facts = captured["facts"]
    assert facts.context_length == 32768
    assert facts.gpu_memory_utilization == 0.6
    assert facts.use_case == "long-inputs"
    assert facts.workload_profile is not None
    assert facts.workload_profile.mean_input_tokens == 16384
    assert facts.vllm_arguments == ()


def test_automatic_execution_is_not_enabled(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )

    with pytest.raises(ValueError, match="automatic Agentic execution is not enabled"):
        asyncio.run(service.create(_request(execution_policy={"mode": "automatic"})))


def test_apply_mode_merges_only_validated_historical_candidate_seeds(monkeypatch):
    service = AgenticDeploymentService(historical_seed_mode="apply")
    facts = PlanningFacts(model_weight_gib=8, vram_per_gpu_gib=32, free_gpu_count=2)
    candidates = DeterministicPlanner().candidates(facts)
    monkeypatch.setattr(
        service._historical_seed_retriever,
        "retrieve",
        lambda *_args: (
            HistoricalCandidateSeed(
                proposal=CandidateProposal(
                    provider_ref="precise-prefix-cache-routing",
                    replicas=1,
                    tensor_parallel_size=1,
                    rationale="Qualified history.",
                ),
                evidence_refs=("benchmark:seed-1",),
            ),
            HistoricalCandidateSeed(
                proposal=CandidateProposal(
                    provider_ref="optimized-baseline",
                    replicas=3,
                    tensor_parallel_size=1,
                    rationale="Exceeds available capacity.",
                ),
                evidence_refs=("benchmark:invalid",),
            ),
        ),
    )

    merged, seed_count = asyncio.run(service._merge_historical_seeds("Qwen/Qwen3-8B", facts, candidates))

    seeded = next(candidate for candidate in merged if candidate.id == "precise-prefix-cache-routing-tp1-r1")
    assert seed_count == 1
    assert "benchmark:seed-1" in seeded.evidence_ids
    assert all(candidate.id != "optimized-baseline-tp1-r3" for candidate in merged)


def test_create_records_selected_benchmark_refs_in_decision_snapshot(monkeypatch, tmp_path):
    service = AgenticDeploymentService(snapshot_store=PlanningSnapshotStore(tmp_path))
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)

    def score_inputs(_self, _model, _facts, candidates):
        selected = DeterministicPlanner().decide(_facts, candidates).candidate_id
        return (
            PerformanceScoreInput(
                candidate_id=selected,
                source="benchmark",
                status="measured",
                evidence_refs=("benchmark:run-1", "benchmark:run-2"),
            ),
        )

    monkeypatch.setattr("llm_d_lens.agentic.service.BenchmarkPerformanceEvidence.score_inputs", score_inputs)
    run = asyncio.run(service.create(_request()))

    assert run.decision_evidence_ids.count("benchmark:run-1") == 1
    assert "benchmark:run-2" in run.decision_evidence_ids
    snapshot = json.loads((tmp_path / run.planning_snapshot_path).read_text(encoding="utf-8"))
    assert snapshot["decision_evidence_ids"] == run.decision_evidence_ids


def test_decision_evidence_includes_selected_ai_raw_samples_once():
    facts = PlanningFacts(model_weight_gib=8, vram_per_gpu_gib=32, free_gpu_count=2)
    candidate = (
        DeterministicPlanner()
        .candidates(facts)[0]
        .model_copy(
            update={
                "evidence_ids": ["benchmark:shared", "benchmark:seed"],
                "historical_benchmarks": [
                    {"benchmark_id": "shared"},
                    {"benchmark_id": "raw"},
                ],
            }
        )
    )

    assert AgenticDeploymentService._decision_evidence_ids(facts, candidate) == [
        "benchmark:shared",
        "benchmark:seed",
        "benchmark:raw",
    ]


def test_create_falls_back_to_deterministic_planning_when_provider_fails(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)

    class FailingProvider:
        async def recommend(self, *_args, **_kwargs):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(
        "llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_environment",
        lambda: FailingProvider(),
    )

    run = asyncio.run(service.create(_request()))

    assert run.planner == "deterministic"
    assert run.planner_fallback_reason is not None
    assert run.decision is not None
    assert run.decision.action == "select_candidate"


def test_create_falls_back_when_provider_selects_an_unvalidated_candidate(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)

    class InvalidProvider:
        async def recommend(self, *_args, **_kwargs):
            from llm_d_lens.agentic.planner import PlannerDecision

            return PlannerDecision(
                action="select_candidate",
                candidate_id="external-invented-candidate",
                confidence=0.8,
                rationale="provider choice",
            ), []

    monkeypatch.setattr(
        "llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_environment",
        lambda: InvalidProvider(),
    )

    run = asyncio.run(service.create(_request()))

    assert run.planner == "deterministic"
    assert run.planner_fallback_reason == "invalid_response"
    assert run.selected_candidate.id != "external-invented-candidate"
    assert run.selected_candidate.id in {candidate.id for candidate in run.candidates}


def test_create_prefers_openai_compatible_planning_when_provider_is_available(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)

    class AvailableProvider:
        async def recommend(self, _facts, candidates, **_kwargs):
            selected = next(candidate for candidate in candidates if candidate.deployable)
            from llm_d_lens.agentic.planner import PlannerDecision

            decision = PlannerDecision(
                action="select_candidate",
                candidate_id=selected.id,
                confidence=0.8,
                rationale="provider choice",
            )
            scored = [
                candidate.model_copy(update={"score": 0.9, "score_source": "openai-compatible"})
                for candidate in candidates
                if candidate.deployable
            ]
            return decision, scored

    monkeypatch.setattr(
        "llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_environment",
        lambda: AvailableProvider(),
    )

    run = asyncio.run(service.create(_request()))

    assert run.planner == "openai-compatible"
    assert run.decision is not None
    assert run.decision.rationale == "provider choice"
    assert run.decision_metadata is not None
    assert run.decision_metadata.planner == "openai-compatible"
    assert run.decision_metadata.planner_model is None
    assert 1 <= len(run.candidates) <= 3
    assert all(candidate.deployable and candidate.score_source == "openai-compatible" for candidate in run.candidates)


def test_create_shows_three_of_provider_selected_candidates(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)

    class Provider:
        async def recommend(self, _facts, candidates, **_kwargs):
            selected = [
                candidate.model_copy(update={"score": 1 - index / 10, "score_source": "openai-compatible"})
                for index, candidate in enumerate(candidates[:10])
            ]
            from llm_d_lens.agentic.planner import PlannerDecision

            return PlannerDecision(
                action="select_candidate", candidate_id=selected[0].id, confidence=0.8, rationale="provider choice"
            ), selected

    monkeypatch.setattr("llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_environment", lambda: Provider())
    run = asyncio.run(service.create(_request()))
    assert len(run.candidates) == 3


def test_create_uses_valid_ai_generated_candidates_without_falling_back(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )

    async def unexpected_resolve(*_args, **_kwargs):
        raise AssertionError("successful AI generation must not pre-resolve facts internally")

    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", unexpected_resolve)

    class Provider:
        settings = OpenAIPlannerSettings(base_url="http://provider", model="generator-model")

        async def recommend(self, facts, candidates, **_kwargs):
            from llm_d_lens.agentic.planner import DeterministicPlanner

            return DeterministicPlanner().decide(facts, candidates), DeterministicPlanner().score(facts, candidates)

    async def generate(*_args, **_kwargs):
        from llm_d_lens.agentic.generator import CandidateGenerationResult

        return CandidateGenerationResult(
            proposals={
                "candidates": [
                    {
                        "provider_ref": "optimized-baseline",
                        "replicas": 1,
                        "tensor_parallel_size": 1,
                        "rationale": "Fits.",
                    },
                    {
                        "provider_ref": "pd-disaggregation",
                        "replicas": 2,
                        "tensor_parallel_size": 2,
                        "prefill_replicas": 2,
                        "prefill_tensor_parallel_size": 2,
                        "rationale": "Too large.",
                    },
                ]
            },
            resolved=_resolved_facts(),
        )

    monkeypatch.setattr("llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_provider_id", lambda _id: Provider())
    monkeypatch.setattr("llm_d_lens.agentic.service.AICandidateGenerator.generate", generate)

    run = asyncio.run(service.create(_request(ai_provider_id="provider-1")))

    assert run.generator == "ai-mcp"
    assert run.generator_model == "generator-model"
    assert run.generator_fallback_reason is None
    assert [proposal["provider_ref"] for proposal in run.generator_candidate_proposals] == [
        "optimized-baseline",
        "pd-disaggregation",
    ]
    assert run.generator_candidate_proposals[1]["rationale"] == "Too large."
    assert [candidate.id for candidate in run.candidates] == ["optimized-baseline-tp1-r1"]
    assert len(run.rejected_candidates) == 1
    assert run.rejected_candidates[0].rejection_reasons == ["insufficient free accelerator cards"]


def test_create_scores_successful_ai_generated_candidates_without_resource_policy(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    captured: dict = {}
    score_calls: list[list[str]] = []
    original_score = DeterministicPlanner.score

    class Response:
        status_code = 200

        @property
        def text(self):
            return json.dumps(self.json())

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"candidate_id":"optimized-baseline-tp2-r1",'
                                '"candidate_ids":["optimized-baseline-tp2-r1","optimized-baseline-tp1-r1"],'
                                '"scores":{"optimized-baseline-tp2-r1":0.9,"optimized-baseline-tp1-r1":0.8},'
                                '"confidence":0.9,"rationale":"AIC-backed choice"}'
                            )
                        }
                    }
                ]
            }

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def post(self, _url, **kwargs):
            captured.update(kwargs)
            return Response()

    async def generate(*_args, **_kwargs):
        from llm_d_lens.agentic.generator import CandidateGenerationResult

        return CandidateGenerationResult(
            proposals={
                "candidates": [
                    {
                        "provider_ref": "optimized-baseline",
                        "replicas": 1,
                        "tensor_parallel_size": 1,
                        "rationale": "Fits.",
                    },
                    {
                        "provider_ref": "optimized-baseline",
                        "replicas": 1,
                        "tensor_parallel_size": 2,
                        "rationale": "Scales.",
                    },
                ]
            },
            resolved=_resolved_facts(
                free_gpu_count=2,
                aic_predictions=(
                    AICCandidatePrediction(
                        mode="agg", tensor_parallel_size=1, replicas=1, throughput_tokens_per_sec=100
                    ),
                    AICCandidatePrediction(
                        mode="agg", tensor_parallel_size=2, replicas=1, throughput_tokens_per_sec=200
                    ),
                ),
            ),
        )

    def score_spy(self, facts, candidates):
        score_calls.append([candidate.id for candidate in candidates])
        return original_score(self, facts, candidates)

    monkeypatch.setattr("llm_d_lens.agentic.service.AICandidateGenerator.generate", generate)
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_provider_id",
        lambda _id: OpenAICompatiblePlanner("http://provider", "planner"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.planner.DeterministicPlanner.score", score_spy)
    monkeypatch.setattr("llm_d_lens.ai_providers.client.httpx.AsyncClient", lambda **_kwargs: Client())

    async def validate_provider(_url):
        pass

    monkeypatch.setattr("llm_d_lens.ai_providers.service.validate_provider_endpoint", validate_provider)

    run = asyncio.run(service.create(_request(ai_provider_id="provider-1")))

    planner_context = (
        captured["json"]["messages"][1]["content"].split("Planning context (JSON):\n", 1)[1].partition("\n\n")[0]
    )
    supplied = json.loads(planner_context)["valid_candidates"]
    supplied_by_id = {candidate["id"]: candidate for candidate in supplied}
    assert run.generator == "ai-mcp"
    assert run.decision_metadata is not None
    assert run.decision_metadata.planner == "openai-compatible"
    assert score_calls
    assert score_calls[0] == ["optimized-baseline-tp1-r1", "optimized-baseline-tp2-r1"]
    assert "resource_policy" not in supplied_by_id["optimized-baseline-tp1-r1"]
    assert "resource_policy" not in supplied_by_id["optimized-baseline-tp2-r1"]


def test_create_uses_aggregate_inputs_only_for_deterministic_planning(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)
    captured = {}

    class Provider:
        async def recommend(self, facts, candidates, **_kwargs):
            captured["performance_score_inputs"] = facts.performance_score_inputs
            captured["baseline_deployable"] = next(
                candidate.deployable for candidate in candidates if candidate.id == "optimized-baseline-tp1-r1"
            )
            planner = DeterministicPlanner()
            return planner.decide(facts, candidates), planner.score(facts, candidates)

    def score_inputs(_self, _model, _facts, candidates):
        baseline = next(candidate for candidate in candidates if candidate.id == "optimized-baseline-tp1-r1")
        return (
            PerformanceScoreInput(
                candidate_id=baseline.id,
                source="benchmark",
                status="measured",
                confidence=1,
                effective_sample_size=3,
                has_blocking_negative_evidence=True,
            ),
        )

    deterministic_calls = []
    original_decide = DeterministicPlanner.decide

    def decide_spy(self, facts, candidates):
        deterministic_calls.append(facts.performance_score_inputs)
        return original_decide(self, facts, candidates)

    monkeypatch.setattr("llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_provider_id", lambda _id: Provider())
    monkeypatch.setattr("llm_d_lens.agentic.service.BenchmarkPerformanceEvidence.score_inputs", score_inputs)
    monkeypatch.setattr("llm_d_lens.agentic.planner.DeterministicPlanner.decide", decide_spy)

    run = asyncio.run(service.create(_request(ai_provider_id="provider-1")))

    assert run.decision_metadata is not None
    assert run.decision_metadata.planner == "openai-compatible"
    assert deterministic_calls[0][0].has_blocking_negative_evidence
    assert captured["performance_score_inputs"] == ()
    assert captured["baseline_deployable"] is True


def test_relaxed_candidate_gets_exact_estimate_before_scoring(monkeypatch):
    calls = []

    async def estimated(request):
        calls.append(request)
        return {"mode": "agg", "tp": 2, "replicas": 1, "ttft_ms": 180, "tpot_ms": 22}

    monkeypatch.setattr("llm_d_lens.agentic.service.estimate", estimated)
    facts = replace(_resolved_facts().facts, ttft_slo_ms=30)
    candidate = (
        DeterministicPlanner()
        .candidates(facts)[0]
        .model_copy(
            update={
                "id": "optimized-baseline-tp2-r1",
                "provider_ref": "optimized-baseline",
                "tensor_parallel_size": 2,
                "replicas": 1,
                "required_gpus": 2,
            }
        )
    )
    candidates, enriched = asyncio.run(
        AgenticDeploymentService._estimate_missing_topologies(
            "Qwen/Qwen3-8B",
            [candidate],
            facts,
            [
                {
                    "tool": "search_candidates",
                    "arguments": {
                        "workload": {"isl": 4096, "osl": 256, "ttftMs": 30, "tpotMs": 10},
                        "searchConfig": {"aicSystemName": "h100_sxm", "aicBackendName": "vllm"},
                    },
                }
            ],
        )
    )

    assert calls[0].gpu_count == 2
    assert calls[0].aic_system_name == "h100_sxm"
    assert calls[0].ttft_target_ms is None
    assert calls[0].tpot_target_ms is None
    assert candidates[0].performance_estimate["ttft_ms"] == 180
    assert candidates[0].performance_estimate_source == "aic_estimate"
    assert DeterministicPlanner()._slo_assessment(enriched, candidates[0])[0] == "not_satisfied"
    assert OpenAICompatiblePlanner._candidate_context(enriched, candidates[0])["performance_estimate"] == {
        "ttft_ms": 180,
        "tpot_ms": 22,
        "throughput_tokens_per_sec": None,
    }


def test_missing_aic_predictions_estimate_each_supported_candidate_for_ai_scoring(monkeypatch):
    calls = []

    async def estimated(request):
        calls.append(request)
        return {"ttft_ms": 180 + len(calls), "tpot_ms": 22, "throughput_tokens_per_sec": 1200}

    monkeypatch.setattr("llm_d_lens.agentic.service.estimate", estimated)
    facts = _resolved_facts(free_gpu_count=4).facts
    candidates = DeterministicPlanner().candidates(facts)
    supported = [
        next(candidate for candidate in candidates if candidate.provider_ref == provider_ref)
        for provider_ref in ("optimized-baseline", "pd-disaggregation")
    ]
    enriched_candidates, enriched_facts = asyncio.run(
        AgenticDeploymentService._estimate_missing_topologies(
            "Qwen/Qwen3-8B",
            supported,
            facts,
            [
                {
                    "tool": "search_candidates",
                    "arguments": {
                        "workload": {"isl": 1024, "osl": 256},
                        "searchConfig": {"aicSystemName": "h100_sxm"},
                    },
                }
            ],
        )
    )

    assert len(calls) == len(supported)
    for index, (candidate, request) in enumerate(zip(enriched_candidates, calls, strict=True), start=1):
        assert request.gpu_count == candidate.required_gpus
        assert request.tp == candidate.tensor_parallel_size
        assert request.replicas == candidate.replicas
        if candidate.provider_ref == "pd-disaggregation":
            assert request.prefill_tp == candidate.prefill_tensor_parallel_size
            assert request.prefill_replicas == candidate.prefill_replicas
        context = OpenAICompatiblePlanner._candidate_context(enriched_facts, candidate)
        assert context["performance_estimate_source"] == "aic_estimate"
        assert context["performance_estimate"] == {
            "ttft_ms": 180 + index,
            "tpot_ms": 22,
            "throughput_tokens_per_sec": 1200,
        }


def test_relaxed_candidate_keeps_unknown_performance_when_estimate_fails(monkeypatch):
    from llm_d_lens.aic.service import AICError

    async def unavailable(_request):
        raise AICError("no matching topology")

    monkeypatch.setattr("llm_d_lens.agentic.service.estimate", unavailable)
    facts = _resolved_facts().facts
    candidate = DeterministicPlanner().candidates(facts)[0]
    candidates, enriched = asyncio.run(
        AgenticDeploymentService._estimate_missing_topologies(
            "Qwen/Qwen3-8B",
            [candidate],
            facts,
            [
                {
                    "tool": "search_candidates",
                    "arguments": {"workload": {}, "searchConfig": {"aicSystemName": "h100_sxm"}},
                }
            ],
        )
    )

    assert candidates[0].performance_estimate is None
    assert enriched.aic_predictions == ()


def test_create_falls_back_only_when_all_ai_generated_candidates_are_invalid(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)

    class Provider:
        settings = OpenAIPlannerSettings(base_url="http://provider", model="generator-model")

        async def recommend(self, facts, candidates, **_kwargs):
            from llm_d_lens.agentic.planner import DeterministicPlanner

            return DeterministicPlanner().decide(facts, candidates), DeterministicPlanner().score(facts, candidates)

    async def generate(*_args, **_kwargs):
        from llm_d_lens.agentic.generator import CandidateGenerationResult

        return CandidateGenerationResult(
            proposals={
                "candidates": [
                    {
                        "provider_ref": "optimized-baseline",
                        "replicas": 8,
                        "tensor_parallel_size": 8,
                        "rationale": "Exceeds capacity.",
                    }
                ]
            },
            resolved=_resolved_facts(),
        )

    monkeypatch.setattr("llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_provider_id", lambda _id: Provider())
    monkeypatch.setattr("llm_d_lens.agentic.service.AICandidateGenerator.generate", generate)
    progress = []

    async def on_progress(event):
        progress.append(event)

    run = asyncio.run(service.create(_request(ai_provider_id="provider-1"), on_progress=on_progress))

    assert run.generator == "deterministic"
    assert run.generator_fallback_reason == "all_ai_candidates_invalid"
    assert run.candidates
    assert run.rejected_candidates[0].id == "optimized-baseline-tp8-r8"
    assert any(
        event["phase"] == "generation"
        and event["status"] == "fallback"
        and "deterministic generation" in event["message"]
        for event in progress
    )


@pytest.mark.parametrize(
    ("error_message", "reason", "detail"),
    [
        ('MCP tool "search_candidates" failed: unavailable', "mcp_unavailable", None),
        (
            "AI candidate generation returned too many tool calls in round 1: 5 (limit 4)",
            "invalid_response",
            "AI candidate generation returned too many tool calls in round 1: 5 (limit 4)",
        ),
    ],
)
def test_create_falls_back_when_ai_generator_mcp_is_unavailable(monkeypatch, tmp_path, error_message, reason, detail):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)

    class Provider:
        settings = OpenAIPlannerSettings(base_url="http://provider", model="generator-model")

        async def recommend(self, facts, candidates, **_kwargs):
            from llm_d_lens.agentic.planner import DeterministicPlanner

            return DeterministicPlanner().decide(facts, candidates), DeterministicPlanner().score(facts, candidates)

    async def generate(*_args, **_kwargs):
        from llm_d_lens.agentic.generator import CandidateGenerationError

        raise CandidateGenerationError(error_message)

    estimate_requests = []

    async def estimated(request):
        estimate_requests.append(request)
        return {"ttft_ms": 150, "tpot_ms": 20, "throughput_tokens_per_sec": 100}

    monkeypatch.setattr("llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_provider_id", lambda _id: Provider())
    monkeypatch.setattr("llm_d_lens.agentic.service.AICandidateGenerator.generate", generate)
    monkeypatch.setattr("llm_d_lens.agentic.service.estimate", estimated)

    run = asyncio.run(service.create(_request(ai_provider_id="provider-1")))

    assert run.generator == "deterministic"
    assert run.generator_fallback_reason == reason
    assert run.generator_fallback_detail == detail
    assert estimate_requests == []
    assert all(candidate.performance_estimate_source != "aic_estimate" for candidate in run.candidates)
    snapshot = json.loads((tmp_path / run.planning_snapshot_path).read_text(encoding="utf-8"))
    assert snapshot["generator"]["fallback_detail"] == detail
    assert all(
        candidate["performance_estimate_source"] != "aic_estimate" for candidate in snapshot["candidate_catalog"]
    )


def test_missing_requested_provider_estimates_fallback_candidates(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)
    monkeypatch.setattr("llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_provider_id", lambda _id: None)
    estimate_requests = []

    async def estimated(request):
        estimate_requests.append(request)
        return {"ttft_ms": 150, "tpot_ms": 20}

    monkeypatch.setattr("llm_d_lens.agentic.service.estimate", estimated)

    run = asyncio.run(service.create(_request(ai_provider_id="provider-1")))

    assert run.generator_fallback_reason == "provider_unavailable"
    assert estimate_requests == []
    assert all(candidate.performance_estimate_source != "aic_estimate" for candidate in run.candidates)


def test_generator_provider_http_error_is_not_classified_as_invalid_candidate_response():
    from llm_d_lens.agentic.generator import CandidateGenerationError

    error = CandidateGenerationError(
        "AI candidate generation failed: OpenAI-compatible provider request failed: "
        "Client error '400 Bad Request' for url 'https://api.deepseek.com/v1/chat/completions'"
    )

    assert AgenticDeploymentService._generator_failure_reason(error) == "provider_unavailable"


def test_generator_invalid_initial_tool_batch_is_classified_as_invalid_response():
    from llm_d_lens.agentic.generator import CandidateGenerationError

    error = CandidateGenerationError(
        "AI candidate generation must first call only get_cluster_overview and AIC search_candidates"
    )

    assert AgenticDeploymentService._generator_failure_reason(error) == "invalid_response"


def test_refine_keeps_provider_selected_candidates(monkeypatch, tmp_path):
    service = AgenticDeploymentService(snapshot_store=PlanningSnapshotStore(tmp_path))
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)

    class Provider:
        async def recommend(self, _facts, candidates, **_kwargs):
            selected = [
                candidate.model_copy(update={"score": 1 - index / 10, "score_source": "openai-compatible"})
                for index, candidate in enumerate(candidates[:10])
            ]
            from llm_d_lens.agentic.planner import PlannerDecision

            return PlannerDecision(
                action="select_candidate", candidate_id=selected[0].id, confidence=0.8, rationale="provider choice"
            ), selected

    monkeypatch.setattr("llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_environment", lambda: Provider())
    run = asyncio.run(service.create(_request()))
    created_path = tmp_path / run.planning_snapshot_path
    created_payload = created_path.read_text(encoding="utf-8")
    from llm_d_lens.agentic.models import AgenticCandidateRefinementRequest

    refined = asyncio.run(service.refine(run.id, AgenticCandidateRefinementRequest(planner_prompt="more capacity")))
    assert 1 <= len(refined.candidates) <= 10
    refined_path = tmp_path / refined.planning_snapshot_path
    assert refined_path != created_path
    assert json.loads(created_payload)["event"] == "created"
    assert json.loads(refined_path.read_text(encoding="utf-8"))["event"] == "refined"
    assert created_path.read_text(encoding="utf-8") == created_payload


@pytest.mark.parametrize("planner_fails", [False, True])
def test_refine_regenerates_candidates_with_ai_and_mcp(monkeypatch, tmp_path, planner_fails):
    from llm_d_lens.agentic.planner import OpenAIPlannerError
    from llm_d_lens.ai_providers.client import OpenAICompatibleClient

    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)
    prompts = []
    planning_lengths = []

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def post(self, endpoint, *, json, headers):
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "complete raw response"}}]},
                request=httpx.Request("POST", endpoint),
            )

    async def validate_endpoint(_url):
        pass

    monkeypatch.setattr("llm_d_lens.ai_providers.service.validate_provider_endpoint", validate_endpoint)
    monkeypatch.setattr("llm_d_lens.ai_providers.client.httpx.AsyncClient", lambda **_kwargs: Client())

    class Provider:
        settings = OpenAIPlannerSettings(base_url="http://provider", model="generator-model")

        async def recommend(self, facts, candidates, **_kwargs):
            from llm_d_lens.agentic.planner import DeterministicPlanner

            await OpenAICompatibleClient("https://api.openai.com/", "planner-model").complete(
                system_prompt="full planner system", user_content=_kwargs["operator_prompt"]
            )
            if planner_fails:
                raise OpenAIPlannerError("invalid recorded response")
            return DeterministicPlanner().decide(facts, candidates), DeterministicPlanner().score(facts, candidates)

    async def generate(_self, _cluster_id, _model, facts, operator_prompt, on_progress=None):
        from llm_d_lens.agentic.generator import CandidateGenerationResult

        await OpenAICompatibleClient("https://api.openai.com/", "generator-model").complete(
            system_prompt="full generator system", user_content=operator_prompt
        )
        prompts.append(operator_prompt)
        planning_lengths.append(facts.context_length)
        return CandidateGenerationResult(
            proposals={
                "candidates": [
                    {
                        "provider_ref": "optimized-baseline",
                        "replicas": 1,
                        "tensor_parallel_size": 1,
                        "rationale": "Fits the refreshed preference.",
                    }
                ]
            },
            resolved=_resolved_facts(),
        )

    monkeypatch.setattr("llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_provider_id", lambda _id: Provider())
    monkeypatch.setattr("llm_d_lens.agentic.service.AICandidateGenerator.generate", generate)

    run = asyncio.run(
        service.create(_request(ai_provider_id="provider-1", planner_prompt="balanced", max_model_len=32768))
    )
    original_path = tmp_path / run.planning_snapshot_path
    original_payload = original_path.read_text()
    assert run.generator_requests[0].request["messages"][1]["content"] == "balanced"
    assert run.planner_requests[0].request["messages"][1]["content"] == "balanced"
    refined = asyncio.run(
        service.refine(
            run.id,
            AgenticCandidateRefinementRequest(planner_prompt="prefer fewer replicas"),
        )
    )

    assert prompts == ["balanced", "prefer fewer replicas"]
    assert planning_lengths == [32768, 32768]
    assert refined.generator == "ai-mcp"
    assert refined.generator_model == "generator-model"
    assert refined.generator_fallback_reason is None
    assert refined.planning_trace[-1]["candidate_id"] == refined.selected_candidate.id
    assert refined.planning_trace[3]["ranking"]
    payload = json.loads((tmp_path / refined.planning_snapshot_path).read_text())
    assert payload["request"]["planner_prompt"] == "prefer fewer replicas"
    for stage in ("generator", "planner"):
        record = payload[stage]["requests"][0]
        assert record["request"]["messages"][1]["content"] == "prefer fewer replicas"
        assert "complete raw response" in record["response_body"]
        assert record["response_status"] == 200
        assert record["finished_at"] is not None
    assert refined.planner_fallback_reason == ("invalid_response" if planner_fails else None)
    assert original_path.read_text() == original_payload
    assert len(list((tmp_path / run.id / "requests").glob("*.json"))) == 4


def test_refine_falls_back_to_deterministic_planning_when_provider_fails(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)

    class FailingProvider:
        async def recommend(self, *_args, **_kwargs):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(
        "llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_environment", lambda: FailingProvider()
    )
    run = asyncio.run(service.create(_request()))
    refined = asyncio.run(
        service.refine(
            run.id,
            AgenticCandidateRefinementRequest(planner_prompt="distributed"),
        )
    )

    assert refined.planner == "deterministic"
    assert refined.planner_fallback_reason is not None
    assert refined.decision_metadata.planner == "deterministic"
    assert all(
        candidate.score_source == "deterministic" and candidate.score is not None for candidate in refined.candidates
    )


def test_refine_without_ai_provider_uses_deterministic_planning(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )

    async def resolve_facts(_cluster_id, _model, facts, **_kwargs):
        resolved = _resolved_facts()
        resolved.facts = replace(
            resolved.facts,
            operator_preference=facts.operator_preference,
            gpu_memory_utilization=facts.gpu_memory_utilization,
        )
        return resolved

    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", resolve_facts)
    monkeypatch.setattr("llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_environment", lambda: None)

    run = asyncio.run(service.create(_request(planner_prompt="balanced", gpu_memory_utilization=0.6)))
    refined = asyncio.run(service.refine(run.id, AgenticCandidateRefinementRequest(planner_prompt="distributed")))

    assert refined.status == AgenticDeploymentStatus.AWAITING_APPROVAL
    assert refined.generator == "deterministic"
    assert refined.generator_fallback_reason is None
    assert refined.planner == "deterministic"
    assert refined.planner_fallback_reason is None
    assert refined.request.planner_prompt == "distributed"
    assert refined.selected_candidate.id == refined.candidates[0].id
    assert refined.selected_candidate.provider_ref == "pd-disaggregation"
    assert refined.selected_candidate.gpu_memory_utilization == 0.6
    assert all(candidate.gpu_memory_utilization == 0.6 for candidate in refined.candidates)
    assert all(
        candidate.score_source == "deterministic" and candidate.score is not None for candidate in refined.candidates
    )


def test_refine_allows_external_ranking_of_exact_aic_candidates(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    resolved = _resolved_facts()
    resolved.facts = PlanningFacts(
        model_weight_gib=8,
        vram_per_gpu_gib=32,
        free_gpu_count=4,
        aic_predictions=(
            AICCandidatePrediction(
                mode="agg",
                tensor_parallel_size=1,
                replicas=1,
                ttft_ms=300,
                tpot_ms=25,
                throughput_tokens_per_sec=700,
            ),
        ),
    )

    async def resolve(*_args, **_kwargs):
        return resolved

    class Provider:
        async def recommend(self, _facts, candidates, **_kwargs):
            selected = [
                candidate.model_copy(update={"score": 0.9, "score_source": "openai-compatible"})
                for candidate in candidates[:2]
            ]
            from llm_d_lens.agentic.planner import PlannerDecision

            return PlannerDecision(
                action="select_candidate", candidate_id=selected[0].id, confidence=0.8, rationale="preference choice"
            ), selected

    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", resolve)
    monkeypatch.setattr("llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_environment", lambda: Provider())

    run = asyncio.run(service.create(_request()))
    refined = asyncio.run(
        service.refine(
            run.id,
            AgenticCandidateRefinementRequest(planner_prompt="distributed"),
        )
    )

    assert refined.planner == "openai-compatible"
    assert refined.decision.rationale == "preference choice"


def test_create_allows_external_planning_when_an_exact_aic_recommendation_exists(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    resolved = _resolved_facts()
    resolved.facts = PlanningFacts(
        model_weight_gib=8,
        vram_per_gpu_gib=32,
        free_gpu_count=4,
        aic_predictions=(
            AICCandidatePrediction(
                mode="agg",
                tensor_parallel_size=1,
                replicas=1,
                ttft_ms=300,
                tpot_ms=25,
                throughput_tokens_per_sec=700,
            ),
        ),
    )

    async def resolve(*_args, **_kwargs):
        return resolved

    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", resolve)

    class AvailableProvider:
        async def recommend(self, _facts, candidates, **_kwargs):
            selected = [
                candidate.model_copy(update={"score": 0.9, "score_source": "openai-compatible"})
                for candidate in candidates[:2]
            ]
            from llm_d_lens.agentic.planner import PlannerDecision

            return PlannerDecision(
                action="select_candidate", candidate_id=selected[0].id, confidence=0.8, rationale="preference choice"
            ), selected

    monkeypatch.setattr(
        "llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_environment",
        lambda: AvailableProvider(),
    )

    run = asyncio.run(service.create(_request()))

    assert run.planner == "openai-compatible"
    assert run.selected_candidate.id == "optimized-baseline-tp1-r1"
    assert all(candidate.score_source == "openai-compatible" for candidate in run.candidates)
    assert len(run.candidates) == 2


def test_create_requires_a_model_cache_volume(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        service,
        "_validate_model_cache",
        _ORIGINAL_VALIDATE_MODEL_CACHE,
    )
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )

    with pytest.raises(ValidationError, match="storage_volume_id"):
        asyncio.run(service.create(_request(storage_volume_id="")))


@pytest.mark.parametrize(
    ("token_source", "expected"),
    [
        (
            {"mode": "existing-secret", "namespace": "models", "name": "hf-token"},
            {"mode": "existing-secret", "sourceNamespace": "models", "sourceName": "hf-token"},
        ),
        # Model Cache ``host`` is best-effort; Deploy ``host`` is not, and
        # mapping it would disable the cluster default Secret fallback.
        ({"mode": "host"}, None),
        ({"mode": "none"}, None),
    ],
    ids=["existing-secret", "host", "none"],
)
def test_validate_model_cache_maps_only_existing_secret_token_source(monkeypatch, token_source, expected):
    from llm_d_lens.model_cache.contracts import ModelCacheEntry
    from llm_d_lens.storage.contracts import StorageVolumeKind, StorageVolumePurpose

    entry = ModelCacheEntry.model_validate(
        {
            "clusterId": "cluster-a",
            "storageVolumeId": "vol-1",
            "status": "ready",
            "cachePath": "hub/models--Qwen--Qwen3-8B",
            "source": {"kind": "huggingface", "huggingface": {"repoId": "Qwen/Qwen3-8B", "revision": "main"}},
            "tokenSource": token_source,
        }
    )
    volume = SimpleNamespace(purposes=[StorageVolumePurpose.MODEL_CACHE], kind=StorageVolumeKind.NFS)

    async def get_ready_volume(*_args, **_kwargs):
        return volume

    class _CacheService:
        async def list(self, **_kwargs):
            return [entry]

    monkeypatch.setattr("llm_d_lens.storage.service.get_ready_volume", get_ready_volume)
    monkeypatch.setattr("llm_d_lens.model_cache.service.default_service", lambda: _CacheService())

    result = asyncio.run(
        _ORIGINAL_VALIDATE_MODEL_CACHE(
            _request(storage_type="model-cache", storage_volume_id="vol-1"),
            "cluster-a",
        )
    )

    assert (result.model_dump(mode="json", by_alias=True) if result else None) == expected


def test_approve_rejects_a_candidate_when_live_cluster_capacity_has_changed(monkeypatch):
    service = AgenticDeploymentService()
    session = SimpleNamespace(id="a" * 36, server_id="cluster-a")
    monkeypatch.setattr("llm_d_lens.agentic.service.require_active_session", lambda _session_id: session)
    resolved = iter([_resolved_facts(free_gpu_count=4), _resolved_facts(free_gpu_count=0)])

    async def resolve(*_args, **_kwargs):
        return next(resolved)

    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", resolve)
    run = asyncio.run(service.create(_request()))

    with pytest.raises(ValueError, match="sufficient free accelerator cards"):
        asyncio.run(service.approve(run.id))


def test_approve_rejects_a_candidate_when_preflight_capacity_validation_fails(monkeypatch):
    service = AgenticDeploymentService()
    session = SimpleNamespace(id="a" * 36, server_id="cluster-a")
    monkeypatch.setattr("llm_d_lens.agentic.service.require_active_session", lambda _session_id: session)

    llama_cfg = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 14336,
        "vocab_size": 128256,
        "max_position_embeddings": 8192,
        "torch_dtype": "bfloat16",
    }
    # First call: cluster has 80GB VRAM, fits
    normal = _resolved_facts(free_gpu_count=4)
    # Second call (during approve): cluster VRAM drops to 8GB, impossible to fit 20GB weights + activation
    oom_facts = _resolved_facts(free_gpu_count=4)
    oom_facts.facts = replace(
        oom_facts.facts,
        vram_per_gpu_gib=8.0,
        model_name="meta-llama/Llama-3-8B",
        model_config_dict=llama_cfg,
    )

    resolved = iter([normal, oom_facts])

    async def resolve(*_args, **_kwargs):
        return next(resolved)

    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", resolve)
    run = asyncio.run(service.create(_request()))

    with pytest.raises(
        ValueError, match="(fails pre-flight capacity validation|satisfies current resource constraints)"
    ):
        asyncio.run(service.approve(run.id))


def test_explicit_provider_is_a_hard_constraint_without_fallback(monkeypatch):
    service = _shared_service(monkeypatch)
    run = asyncio.run(service.create(_shared_request(
        workload={"prefillHeavy": True},
        hard_constraints={"maxAccelerators": 2, "providerRef": "optimized-baseline"},
    )))
    assert run.selected_candidate.provider_ref == "optimized-baseline"
    assert all(candidate.provider_ref == "optimized-baseline" for candidate in run.candidates)
    assert any("explicitly selected" in reason for candidate in run.rejected_candidates for reason in candidate.rejection_reasons)
    with pytest.raises(ValueError, match="no deployable"):
        asyncio.run(service.create(_shared_request(
            hard_constraints={"maxAccelerators": 1, "providerRef": "pd-disaggregation"},
            workload={"prefillHeavy": True},
        )))
    with pytest.raises(ValueError, match="Unsupported planning provider"):
        asyncio.run(service.create(_shared_request(hard_constraints={"providerRef": "unregistered-provider"})))


def test_shared_planning_preserves_percentiles_end_to_end_target_and_preference(monkeypatch):
    service = _shared_service(monkeypatch)
    request = _shared_request(
        workload={"inputTokens": 1024, "p95InputTokens": 4096, "outputTokens": 128},
        performance_targets={"e2eMs": 2000},
    ).model_copy(update={"planner_prompt": "Prefer simpler operations."})
    run = asyncio.run(service.create(request))
    assert run.request.planning_facts.workload_profile.mean_input_tokens == 1024
    assert run.request.planning_facts.workload_profile.p95_input_tokens == 4096
    assert run.request.planning_facts.end_to_end_latency_slo_ms == 2000
    assert run.request.planner_prompt == "Prefer simpler operations."
    unknown = asyncio.run(service.create(_shared_request(workload={"inputTokens": 1024})))
    assert unknown.request.planning_facts.workload_profile.p95_input_tokens is None
