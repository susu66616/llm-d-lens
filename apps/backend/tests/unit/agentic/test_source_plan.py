import asyncio
import json
from types import SimpleNamespace

import pytest

from llm_d_lens.agentic import source_plan as source
from llm_d_lens.agentic.models import AgenticCandidate, AgenticDeploymentCreateRequest, AgenticDeploymentRun
from llm_d_lens.agentic.planner import PlannedCandidate, PlanningFacts
from llm_d_lens.agentic.planning_snapshot import PlanningSnapshotStore
from llm_d_lens.agentic.service import AgenticDeploymentService
from llm_d_lens.configuration.decision_engine import rank_manual_candidates
from llm_d_lens.configuration.plan_models import ConfigurationPlan, DeploymentRequirements
from llm_d_lens.deploy.source_composition import bind_model, dump


def bundle(features=("prefix",)):
    binding = {
        "entryId": "cache",
        "storageVolumeId": "volume",
        "model": "test/model",
        "revision": "pinned",
        "pvcName": "weights",
        "namespace": "cache",
    }
    docs = [
        {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": "cache"}},
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "engine", "namespace": "cache"},
            "spec": {
                "replicas": 1,
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": "modelserver",
                                "image": "test:pinned",
                                "command": ["vllm", "serve"],
                                "args": [
                                    "test/model",
                                    "--tensor-parallel-size=1",
                                    "--max-model-len=1024",
                                    "--gpu-memory-utilization=0.9",
                                    "--enable-prefix-caching",
                                ],
                            }
                        ]
                    }
                },
            },
        },
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": "inference", "namespace": "cache"},
            "spec": {"ports": [{"port": 8000}]},
        },
    ]
    bind_model(docs, binding)
    return {
        "commit": "a" * 40,
        "compositionChecksum": "baseline",
        "resourceCount": len(docs),
        "optimizations": {"features": [{"id": identifier, "state": "configured"} for identifier in features]},
        "validation": {"warnings": [], "hardware": {"demand": {"fixture/device": 1}}, "clusterValidated": False},
        "files": [
            {"name": "deployment.yaml", "content": dump(docs)},
            {
                "name": "provenance.json",
                "content": json.dumps({"modelBinding": binding, "components": [{"path": "guides/base"}]}),
            },
        ],
    }


@pytest.fixture
def workflow(monkeypatch, tmp_path):
    facts = PlanningFacts(8, 32, 4, cpu_buffer_gib=128)
    requirements = DeploymentRequirements(
        model="test/model", cluster_id="cluster", hard_constraints={"maxAccelerators": 2}
    )
    request = AgenticDeploymentCreateRequest(
        model="test/model",
        deployment_name="fixture",
        cluster_session_id="a" * 36,
        storage_volume_id="volume",
        replicas=1,
        tensor_parallel_size=1,
        max_model_len=1024,
        planning_facts=facts,
        planning_mode="manual",
        requirements=requirements,
    )
    candidates = [
        PlannedCandidate(
            id=identifier,
            provider_ref="optimized-baseline",
            replicas=1,
            tensor_parallel_size=1,
            required_gpus=1,
            deployable=True,
            max_model_len=1024,
            gpu_memory_utilization=0.9,
        )
        for identifier in ("a", "b")
    ]
    run = AgenticDeploymentRun(
        request=request,
        status="awaiting_approval",
        candidates=candidates,
        selected_candidate=AgenticCandidate(
            id="a", replicas=1, tensor_parallel_size=1, max_model_len=1024, gpu_memory_utilization=0.9
        ),
        configuration_plan=ConfigurationPlan(plan_id="initial", mode="manual", requirements=requirements),
    )
    service = AgenticDeploymentService(snapshot_store=PlanningSnapshotStore(tmp_path))
    service._runs[run.id] = run
    session = SimpleNamespace(id="a" * 36, server_id="cluster")
    monkeypatch.setattr(source, "require_active_session", lambda _: session)
    monkeypatch.setattr(source, "all_profiles", lambda: [SimpleNamespace(id="fixture", upstream_variant="fixture")])

    async def resolve(*args, **kwargs):
        return SimpleNamespace(facts=facts)

    async def binding(*args, **kwargs):
        return source.bundle_binding(bundle())

    async def compose(request, body):
        assert body["deploymentIntent"]["maxDevices"] == 2
        return bundle(("prefix", "chunked") if body.get("extra") else ("prefix",))

    monkeypatch.setattr(source, "resolve_planning_facts", resolve)
    monkeypatch.setattr(source, "model_binding", binding)
    monkeypatch.setattr(source, "compose", compose)
    proposals = [
        {
            "candidate_id": identifier,
            "draft": {"profileId": "fixture"},
            "composition": {"clusterId": "cluster", "modelCacheEntryId": "cache", "extra": identifier == "b"},
        }
        for identifier in ("a", "b")
    ]
    return service, run, proposals


def test_manual_compares_final_features_persists_revision_and_restores(workflow):
    service, run, proposals = workflow
    result = asyncio.run(
        source.render_source_plan(
            service, run.id, source.SourcePlanRequest(revision=1, compare=True, candidates=proposals), None
        )
    )
    assert result.selected_candidate.id == "b"
    assert result.revision == 2
    assert {"prefix", "chunked"} <= set(result.configuration_plan.selected_capabilities)
    assert len(result.configuration_plan.decisions) == 40
    assert result.configuration_plan.configuration["checksum"] != "baseline"
    assert result.source_endpoint is None
    restored = service._snapshot_store.load_run(run.id)
    assert restored.configuration_plan.source_bundle == result.configuration_plan.source_bundle
    with pytest.raises(ValueError, match="plan changed"):
        asyncio.run(
            source.render_source_plan(
                service, run.id, source.SourcePlanRequest(revision=1, compare=True, candidates=proposals), None
            )
        )


def test_selecting_reviewed_source_candidate_restores_exact_parameters(workflow):
    service, run, proposals = workflow
    result = asyncio.run(
        source.render_source_plan(
            service, run.id, source.SourcePlanRequest(revision=1, compare=True, candidates=proposals), None
        )
    )
    result.optimization = {"status": "completed", "best_candidate_id": "a"}
    result.source_endpoint = result.configuration_plan.endpoints[0]["value"]
    selected = service.select_candidate(run.id, "a")
    assert selected.revision == 3
    assert selected.selected_candidate.id == "a"
    assert selected.source_endpoint is None
    assert selected.optimization == result.optimization
    assert selected.configuration_plan.source_bundle == result.configuration_plan.source_candidates["a"]["bundle"]
    assert selected.configuration_plan.source_request["extra"] is False
    assert selected.configuration_plan.selected_capabilities == ["prefix"]
    assert service._snapshot_store.load_run(run.id).configuration_plan == selected.configuration_plan
    assert result.selected_candidate.id == "b"
    selected.optimization = {"status": "running"}
    with pytest.raises(ValueError, match="Finish or clean up"):
        service.select_candidate(run.id, "b")


@pytest.mark.parametrize("edit", [{"render_resources": True}, {"replicas": 2}])
def test_legacy_edits_cannot_replace_reviewed_source_parameters(workflow, monkeypatch, edit):
    from fastapi import HTTPException

    from llm_d_lens.agentic import router
    from llm_d_lens.agentic.models import AgenticCandidateSelectionRequest

    service, run, proposals = workflow
    result = asyncio.run(
        source.render_source_plan(
            service, run.id, source.SourcePlanRequest(revision=1, compare=True, candidates=proposals), None
        )
    )
    monkeypatch.setattr(router, "service", service)
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            router.select_agentic_candidate(run.id, AgenticCandidateSelectionRequest(candidate_id="b", **edit), None)
        )
    assert error.value.status_code == 409
    assert "composition endpoint" in error.value.detail
    assert service.get(run.id) is result


def test_required_feature_failure_and_foreign_cluster_leave_original_untouched(workflow):
    service, run, proposals = workflow
    run.request.requirements.required_features = ["chunked_prefill"]
    proposals[0]["composition"]["clusterId"] = "another-cluster"
    proposals[1]["composition"]["extra"] = False
    with pytest.raises(ValueError, match="No composition passed"):
        asyncio.run(
            source.render_source_plan(
                service, run.id, source.SourcePlanRequest(revision=1, compare=True, candidates=proposals), None
            )
        )
    assert service.get(run.id) is run
    assert run.configuration_plan.source_bundle is None


def test_identical_source_failures_are_reported_once_without_changing_plan(workflow, monkeypatch):
    service, run, proposals = workflow
    message = "Inspect the chart and pin a concrete version before generating."

    async def failed_compose(*_):
        raise source.GuideError(message)

    monkeypatch.setattr(source, "compose", failed_compose)
    with pytest.raises(ValueError) as error:
        asyncio.run(
            source.render_source_plan(
                service, run.id, source.SourcePlanRequest(revision=1, compare=True, candidates=proposals), None
            )
        )
    assert str(error.value) == f"No composition passed: {message}"
    assert service.get(run.id) is run
    assert run.configuration_plan.source_bundle is None


def test_compare_requires_all_candidates_and_rejects_midflight_revisions(workflow, monkeypatch):
    service, run, proposals = workflow
    with pytest.raises(ValueError, match="every retained"):
        asyncio.run(
            source.render_source_plan(
                service, run.id, source.SourcePlanRequest(revision=1, compare=True, candidates=proposals[:1]), None
            )
        )
    original = source.compose

    async def raced(request, body):
        run.revision += 1
        return await original(request, body)

    monkeypatch.setattr(source, "compose", raced)
    with pytest.raises(ValueError, match="changed during"):
        asyncio.run(
            source.render_source_plan(
                service, run.id, source.SourcePlanRequest(revision=1, candidates=proposals[:1]), None
            )
        )
    assert run.configuration_plan.source_bundle is None


def test_agentic_preserves_selected_candidate_instead_of_maximizing_coverage(workflow):
    service, run, proposals = workflow
    run.request.planning_mode = "agentic"
    result = asyncio.run(
        source.render_source_plan(
            service, run.id, source.SourcePlanRequest(revision=1, compare=True, candidates=proposals), None
        )
    )
    assert result.selected_candidate.id == "a"
    assert len(result.configuration_plan.source_candidates) == 2


def test_agentic_can_select_a_checked_alternative_with_explicit_reason(workflow):
    service, run, proposals = workflow
    run.request.planning_mode = "agentic"
    proposals[0]["error"] = "This source variant has unsupported prerequisites."
    result = asyncio.run(
        source.render_source_plan(
            service, run.id, source.SourcePlanRequest(revision=1, compare=True, candidates=proposals), None
        )
    )
    assert result.selected_candidate.id == "b"
    assert "first recommendation failed" in result.configuration_plan.validation["warnings"][-1]
    assert result.configuration_plan.validation["coverage"]["rejected"][0]["candidate_id"] == "a"


def test_memory_and_required_exclusion_use_final_manifest(workflow):
    _, run, _ = workflow
    facts = PlanningFacts(8, 32, 4)
    parsed = source.bundle_documents(bundle())
    assert source.validate_memory(run, parsed, facts)
    with pytest.raises(ValueError):
        source.validate_memory(run, parsed, PlanningFacts(800, 1, 4))
    run.request.requirements.forbidden_features = ["prefix_cache"]
    with pytest.raises(ValueError, match="Forbidden"):
        source.validate_requirements(run, bundle())


def test_final_edits_update_candidate_sizes_and_invalidate_old_predictions(workflow):
    _, run, _ = workflow
    candidate = run.candidates[0].model_copy(update={"slo_status": "satisfied", "score": 0.9})
    rendered = bundle()
    parsed = source.bundle_documents(rendered)
    parsed[1]["spec"]["replicas"] = 2
    parsed[1]["spec"]["template"]["spec"]["containers"][0]["args"].append("--max-num-seqs=12")
    rendered["files"][0]["content"] = dump(parsed)
    rendered["validation"]["hardware"]["demand"] = {"nvidia.com/gpu": 2}
    updated = source.candidate_from_bundle(candidate, rendered)
    assert updated.replicas == 2
    assert updated.required_gpus == 2
    assert updated.slo_status == "unknown"
    assert updated.score is None
    assert candidate.slo_status == "satisfied"


def test_final_resources_clear_removed_seed_optimizations(workflow):
    _, run, _ = workflow
    candidate = run.candidates[0].model_copy(update={"native_cpu_cache_gib": 8, "prefix_routing": True})
    result = source.candidate_from_bundle(candidate, bundle())
    assert result.native_cpu_cache_gib is None
    assert result.prefix_routing is False


def test_midflight_legacy_selection_invalidates_source_render(workflow, monkeypatch):
    service, run, proposals = workflow
    original_compose = source.compose

    async def compose(request, body):
        result = await original_compose(request, body)
        run.configuration_plan = run.configuration_plan.model_copy(deep=True)
        return result

    monkeypatch.setattr(source, "compose", compose)
    with pytest.raises(ValueError, match="changed during rendering"):
        asyncio.run(
            source.render_source_plan(
                service, run.id, source.SourcePlanRequest(revision=1, candidates=proposals[:1]), None
            )
        )


def test_approval_requires_authorized_context_and_unchanged_reviewed_bytes(workflow):
    service, run, proposals = workflow
    run = asyncio.run(
        source.render_source_plan(service, run.id, source.SourcePlanRequest(revision=1, candidates=proposals[:1]), None)
    )
    with pytest.raises(ValueError, match="authorized"):
        asyncio.run(source.approve_source_plan(service, run, owner_user_id=None, request=None))
    run.configuration_plan.source_bundle["files"][0]["content"] += "\n# changed after review\n"
    with pytest.raises(ValueError, match="changed after review"):
        asyncio.run(source.approve_source_plan(service, run, owner_user_id=None, request=object()))


def test_manual_never_prefers_more_features_that_fail_slo():
    more = SimpleNamespace(
        id="more",
        provider_ref="pd-disaggregation",
        prefix_routing=True,
        native_cpu_cache_gib=1,
        deployable=True,
        slo_status="not_satisfied",
        required_gpus=4,
        tensor_parallel_size=2,
        replicas=1,
    )
    less = SimpleNamespace(
        id="less",
        provider_ref="optimized-baseline",
        prefix_routing=False,
        native_cpu_cache_gib=None,
        deployable=True,
        slo_status="satisfied",
        required_gpus=1,
        tensor_parallel_size=1,
        replicas=1,
    )
    assert [item.id for item in rank_manual_candidates([more, less], DeploymentRequirements(model="test"))] == ["less"]


def test_managed_artifact_uses_the_selected_service_and_exact_manifest(workflow, monkeypatch):
    service, run, proposals = workflow
    run = asyncio.run(
        source.render_source_plan(service, run.id, source.SourcePlanRequest(revision=1, candidates=proposals[:1]), None)
    )
    monkeypatch.setattr(
        source.registry, "require_cluster", lambda _: SimpleNamespace(llm_d_repo_path="/fixture", llm_d_ref="pinned")
    )
    import llm_d_lens.configuration.service as configuration_service

    monkeypatch.setattr(configuration_service, "require_active_session", source.require_active_session)
    with pytest.raises(ValueError, match="Service"):
        source.source_configuration(run, run.configuration_plan.source_bundle, "unknown:8000")
    result = source.source_configuration(run, run.configuration_plan.source_bundle, "inference:8000")
    assert result.provider_ref == "source-composition"
    assert result.content["officialGuide"]["deployment"]["endpoint"]["serviceName"] == "inference"
    assert "--enable-prefix-caching" in result.content["officialGuide"]["renderedManifest"]
    assert result.content["runtime"]["storageVolumeId"] == "volume"
