"""Independent HTTP API for Agentic Deploy orchestration."""

import logging
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from llm_d_lens.agentic.public_benchmark_record.access import require_query_access
from llm_d_lens.agentic.public_benchmark_record.case_retrieval import assert_case_access
from llm_d_lens.auth.access import current_principal, owner_for_create
from llm_d_lens.auth.contracts import Principal
from llm_d_lens.core.exceptions import ForbiddenError, UnauthenticatedError

from .models import (
    AgenticApprovalRequest,
    AgenticCandidateRefinementRequest,
    AgenticCandidateSelectionRequest,
    AgenticDeploymentCreateRequest,
    AgenticDeploymentRun,
)
from .service import service
from .streaming import planning_stream_response
from .configuration import planning_resource_request
from .source_plan import SourcePlanRequest, render_source_plan

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/agentic-deployments", tags=["agentic-deployments"])

@router.post(
    "/{run_id}/composition", response_model=AgenticDeploymentRun,
    operation_id="render_deployment_composition",
    summary="Render and compare source-backed deployment plans",
    description=(
        "Render pinned source proposals for an idle Manual or Agentic plan. Requires the current revision, "
        "candidate IDs, scoped model cache and composition inputs. compare=true accounts for every retained "
        "candidate; Manual chooses maximum verified coverage, Agentic preserves its recommendation when valid. "
        "Validate resources, memory and required/forbidden mechanisms, persist the new reviewed revision and "
        "invalidate previous approval and performance evidence. No deployment starts. Stale revisions or rejected "
        "compositions return 409 without replacing the previous plan; timeouts return 504."
    ),
)
async def compose_deployment_plan(run_id: str, body: SourcePlanRequest, request: Request):
    from llm_d_lens.auth.context import require_route_permission

    await require_route_permission(Request(dict(request.scope, path="/api/guide-planning/compose", method="POST")))
    _optimization_run(run_id, request)
    try:
        return await render_source_plan(service, run_id, body, request)
    except KeyError as error:
        raise HTTPException(404, str(error)) from error
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    except TimeoutError as error:
        raise HTTPException(504, "Source comparison timed out; the previous plan is unchanged.") from error


def _require_plan_evidence_access(run: AgenticDeploymentRun, principal: Principal | None) -> None:
    if any(item.source == "public-benchmark-rag" and item.status == "available" for item in run.planning_evidence):
        require_query_access(principal)
    for item in run.planning_evidence:
        if item.source == "configuration-cases":
            assert_case_access(tuple(item.details.get("cases", ())), principal)


@router.post(
    "",
    response_model=AgenticDeploymentRun,
    status_code=201,
    summary="Create a new Agentic Deployment plan.",
    description=(
        "Create a new Agentic Deployment plan. Provide the standard deployment inputs plus planning facts so Lens "
        "can propose, score, and stage candidate deployment topologies. Set planning_mode=manual for deterministic "
        "selection without an external AI provider. Optional shared requirements enforce model/cluster identity, "
        "accelerator limits and required/forbidden mechanisms. A configuration_plan reports planning checks "
        "separately from pending manifest, admission, readiness and performance checks. Both modes can combine "
        "native CPU cache with compatible serving/routing topologies. Manual prioritizes capability coverage; "
        "balanced is the default objective and uses fewer resources to break ties. Unknown workload traits remain "
        "unknown and never exclude a technically compatible combination. Prefix-affinity routing and CPU KV cache "
        "are independent choices on combined or P/D topologies, validated against pinned source plugins. "
        "With an explicit AI provider, planner_prompt can supply missing qualitative workload traits; "
        "structured values take precedence and interpretation failures are reported. "
        "Numeric limits are never inferred. "
        "Cache uses at most 8 GiB per serving replica and 25% of observed free host memory; "
        "source compatibility "
        "remains pending until rendering. Decisions distinguish missing evidence, unsupported combinations and "
        "Agentic alternatives from real conflicts. No cluster writes occur."
    ),
    operation_id="create_agentic_plan",
)
async def create_agentic_deployment(
    request: AgenticDeploymentCreateRequest,
    http_request: Request,
) -> AgenticDeploymentRun:
    try:
        principal = current_principal(http_request)
        return await service.create(request, principal_id=owner_for_create(principal), principal=principal)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/stream", summary="Create an Agentic Deployment plan with live progress events.")
async def stream_agentic_deployment(
    request: AgenticDeploymentCreateRequest,
    http_request: Request,
) -> StreamingResponse:
    principal = current_principal(http_request)
    principal_id = owner_for_create(principal)

    return planning_stream_response(
        lambda on_progress: service.create(
            request, on_progress=on_progress, principal_id=principal_id, principal=principal
        ),
        expected_errors=(ValueError,),
        failure_message="Unable to generate an Agentic recommendation.",
        on_unexpected_error=lambda: logger.exception("Agentic recommendation generation failed"),
    )


@router.get(
    "",
    response_model=list[AgenticDeploymentRun],
    summary=(
        "List Agentic Deployment plans (recommendations) created in this Lens instance, with their current status."
    ),
    description=(
        "List Agentic Deployment plans (recommendations) created in this Lens instance, with their current status."
    ),
    operation_id="list_agentic_plans",
)
async def list_agentic_deployments(http_request: Request) -> list[AgenticDeploymentRun]:
    principal = current_principal(http_request)
    visible = []
    for run in service.list():
        try:
            _require_plan_evidence_access(run, principal)
        except (ForbiddenError, UnauthenticatedError):
            continue
        visible.append(run)
    return visible


@router.get(
    "/{run_id}",
    response_model=AgenticDeploymentRun,
    summary=(
        "Get the full detail of one Agentic Deployment plan: candidates, evidence, scoring source (deterministic vs."
    ),
    description=(
        "Get the full detail of one Agentic Deployment plan: candidates, evidence, scoring source (deterministic vs. "
    ),
    operation_id="get_agentic_plan",
)
async def get_agentic_deployment(run_id: str, http_request: Request) -> AgenticDeploymentRun:
    run = service.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="agentic deployment run not found")
    _require_plan_evidence_access(run, current_principal(http_request))
    return run


@router.post(
    "/{run_id}/select",
    response_model=AgenticDeploymentRun,
    summary="Select one candidate inside an existing Agentic Deployment plan.",
    description=(
        "Select one candidate inside an existing Agentic Deployment plan. Use this when the planner produced "
        "multiple candidates and you want to pin one before approval or refinement. Optional role, replicas and "
        "tensor_parallel_size request a live-revalidated topology revision instead. Optional max_model_len, "
        "gpu_memory_utilization, max_num_seqs and max_num_batched_tokens revise all serving roles. "
        "native_cpu_cache_gib edits an already selected CPU KV cache within current host-memory limits. "
        "A revision returns a new run "
        "with parent_run_id, preserves the original, and invalidates previous performance evidence. Invalid edits "
        "return 409 and do not modify the original plan. render_resources=true renders source-pinned YAML and "
        "checks manifest consistency for the selected candidate, without cluster writes. Rendering can fetch "
        "official source files and read cluster inventory. With compare_combinations=true, Manual checks all retained "
        "combinations and selects the greatest verified capability coverage, using the planning order only for ties. "
        "Supported runtime flags are completed in Manual; explicit required/forbidden flags are checked before "
        "accepting the result. Missing composition adapters are reported as unsupported, not as conflicts. "
        "The response includes validation.coverage for Manual combination comparison. "
        "For source compositions, candidate-only selection restores the exact reviewed snapshot, increments revision "
        "and resets the endpoint choice. Unreviewed alternatives and selections during unfinished trials return 409. "
        "Source parameter edits and rendering must use the composition endpoint, not these legacy edit fields. "
        "All failures preserve the original recommendation and return 409. Intermediate selections are not published. "
        "Admission, readiness and performance remain pending."
    ),
    operation_id="select_agentic_plan_candidate",
)
async def select_agentic_candidate(
    run_id: str,
    request: AgenticCandidateSelectionRequest,
    http_request: Request,
) -> AgenticDeploymentRun:
    try:
        run = service.get(run_id)
        if run is not None:
            _require_plan_evidence_access(run, current_principal(http_request))
        edit_requested = bool(request.model_dump(exclude_none=True, include={
            "replicas", "tensor_parallel_size", "max_model_len", "gpu_memory_utilization",
            "max_num_seqs", "max_num_batched_tokens", "native_cpu_cache_gib",
        }))
        if request.compare_combinations and not request.render_resources:
            raise ValueError("Combination comparison requires render_resources=true.")
        if edit_requested or request.render_resources:
            current = service.get(run_id)
            if (current is not None and current.configuration_plan is not None
                    and current.configuration_plan.source_bundle is not None):
                raise ValueError("Edit and render source parameters through the composition endpoint.")
        if request.render_resources:
            from llm_d_lens.deploy.guide_planning import plan_resources
            from llm_d_lens.storage.service import get_ready_volume

            if edit_requested:
                raise ValueError("Apply topology edits before rendering resources.")
            run = service.get(run_id)
            if run is None:
                raise KeyError("deployment plan not found")
            if run.selected_candidate.id != request.candidate_id or run.status != "awaiting_approval":
                raise ValueError("Only the selected, idle candidate can be rendered.")
            if run.configuration_plan is None:
                raise ValueError("Regenerate this legacy plan before reviewing resources.")
            expected_checksum = run.configuration_plan.configuration["checksum"] if run.configuration_plan else None
            volume = await get_ready_volume(
                run.request.storage_volume_id, cluster_id=run.request.requirements.cluster_id,
            )
            if request.compare_combinations:
                async def render_trial(trial):
                    body = planning_resource_request(trial)
                    body["modelPvcClaimName"] = volume.pvc_name
                    result = await plan_resources(http_request, body)
                    payload = json.loads(result.body)
                    if result.status_code in {400, 404, 409, 422}:
                        raise ValueError(payload.get("error", "Source/resource compatibility check failed."))
                    if result.status_code >= 400:
                        raise HTTPException(status_code=result.status_code, detail=payload.get("error", "Resource generation failed."))
                    return payload
                return await service.compare_manual_resources(run_id, request.candidate_id, render_trial)
            body = planning_resource_request(run)
            body["modelPvcClaimName"] = volume.pvc_name
            result = await plan_resources(http_request, body)
            payload = json.loads(result.body)
            if result.status_code >= 400:
                raise HTTPException(status_code=result.status_code, detail=payload.get("error", "Resource generation failed."))
            if (
                run.status != "awaiting_approval"
                or run.selected_candidate.id != request.candidate_id
                or run.configuration_plan.configuration["checksum"] != expected_checksum
            ):
                raise ValueError("The plan changed during rendering. Generate resources for the current selection.")
            return service.accept_rendered_plan(run, payload)
        if edit_requested:
            return await service.revise_candidate(run_id, request)
        return service.select_candidate(run_id, request.candidate_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.post(
    "/{run_id}/refine",
    response_model=AgenticDeploymentRun,
    summary="Ask Lens to refine an Agentic Deployment plan with a new planner prompt.",
    description=(
        "Ask Lens to refine an Agentic Deployment plan with a new planner prompt. Use this to request another "
        "planning pass with tighter constraints or different priorities."
    ),
    operation_id="refine_agentic_plan",
)
async def refine_agentic_deployment(
    run_id: str,
    request: AgenticCandidateRefinementRequest,
    http_request: Request,
) -> AgenticDeploymentRun:
    try:
        principal = current_principal(http_request)
        run = service.get(run_id)
        if run is not None:
            _require_plan_evidence_access(run, principal)
        return await service.refine(run_id, request, principal_id=owner_for_create(principal), principal=principal)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.post("/{run_id}/refine/stream", summary="Refine an Agentic Deployment plan with live progress events.")
async def stream_refine_agentic_deployment(
    run_id: str,
    request: AgenticCandidateRefinementRequest,
    http_request: Request,
) -> StreamingResponse:
    principal = current_principal(http_request)
    principal_id = owner_for_create(principal)
    run = service.get(run_id)
    if run is not None:
        _require_plan_evidence_access(run, principal)

    return planning_stream_response(
        lambda on_progress: service.refine(
            run_id,
            request,
            on_progress=on_progress,
            principal_id=principal_id,
            principal=principal,
        ),
        expected_errors=(KeyError, ValueError),
        failure_message="Unable to recalculate the Agentic recommendation.",
    )


@router.post(
    "/{run_id}/approve",
    response_model=AgenticDeploymentRun,
    status_code=202,
    summary="Approve an Agentic Deployment plan so Lens can begin executing it.",
    description=(
        "Approve an Agentic Deployment plan so Lens can begin executing it. "
        "Use this only after reviewing the selected candidate and evidence. Shared requirements-based plans "
        "require generated resources and a matching configuration_checksum in the JSON body; stale or missing "
        "review evidence returns 409. Source compositions also require endpoint to identify a reviewed generated "
        "Service. Capacity and model-cache binding are refreshed before deploying the stored reviewed manifest."
    ),
    operation_id="approve_agentic_plan",
)
async def approve_agentic_deployment(
    run_id: str, request: Request = None, approval: AgenticApprovalRequest | None = None,
) -> AgenticDeploymentRun:
    try:
        run = service.get(run_id)
        if run is not None:
            _require_plan_evidence_access(run, current_principal(request))
        if run is not None and run.request.requirements is not None and (
            run.configuration_plan is None or run.configuration_plan.validation.get("status") != "resources_checked"
        ):
            raise ValueError("Generate and review the current configuration resources before approval.")
        if run is not None and run.request.requirements is not None and (
            approval is None or approval.configuration_checksum != run.configuration_plan.configuration["checksum"]
        ):
            raise ValueError("The configuration changed or was not reviewed. Confirm the current checksum before deployment.")
        if run is not None and run.configuration_plan is not None and run.configuration_plan.source_bundle is not None:
            _optimization_run(run_id, request)
            if approval is None or approval.endpoint not in {item["value"] for item in run.configuration_plan.endpoints}:
                raise ValueError("Choose the reviewed inference Service before deployment.")
            return await service.approve(
                run_id, owner_user_id=owner_for_create(current_principal(request)),
                http_request=request, source_endpoint=approval.endpoint,
            )
        return await service.approve(run_id, owner_user_id=owner_for_create(current_principal(request)))
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


from .optimization import MeasuredOptimizer, OptimizationRequest

optimizer = MeasuredOptimizer(service)


def _optimization_run(run_id, http_request):
    from llm_d_lens.auth.access import require_cluster_access
    from llm_d_lens.cluster.sessions import require_active_session

    run = service.get(run_id)
    if run is None:
        raise HTTPException(404, "Agentic plan not found")
    session = require_active_session(run.request.cluster_session_id)
    require_cluster_access(current_principal(http_request), session.server_id)
    return run


@router.post("/{run_id}/optimize", response_model=AgenticDeploymentRun, status_code=202,
    operation_id="optimize_agentic_plan", summary="Measure candidates within an explicit resource budget",
    description="Deploy, benchmark and clean up selected validated candidates sequentially using one workload. Source-backed plans require a reviewed composition and an endpoints mapping of candidate IDs to generated Service values; trials use the exact rendered resources, not a regenerated provider recipe. Return the best measured candidate subject to SLOs; this does not prove a global optimum.")
async def optimize_agentic_plan(run_id: str, request: OptimizationRequest, http_request: Request):
    run = _optimization_run(run_id, http_request)
    from llm_d_lens.auth.context import require_route_permission
    await require_route_permission(Request(dict(http_request.scope, path="/api/v1/evaluate/runs", method="POST")))
    try:
        from llm_d_lens.cluster.sessions import require_active_session
        return optimizer.start(run, request, http_request, owner_for_create(current_principal(http_request)), cluster_id=require_active_session(run.request.cluster_session_id).server_id)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@router.post("/{run_id}/optimize/cancel", response_model=AgenticDeploymentRun,
    operation_id="cancel_agentic_optimization", summary="Stop measured optimization and clean up its test resources",
    description="Cancel an active measured candidate search and clean up only the deployments created by that search.")
async def cancel_agentic_optimization(run_id: str, http_request: Request, cluster_session_id: str | None = None):
    run = service.get(run_id)
    if run is None:
        raise HTTPException(404, "Agentic plan not found")
    if cluster_session_id and (run.optimization or {}).get('status') in {'interrupted', 'cleanup_failed'}:
        from llm_d_lens.cluster.sessions import require_active_session
        from llm_d_lens.auth.access import require_cluster_access
        session = require_active_session(cluster_session_id)
        if session.server_id != run.optimization.get('cluster_id'):
            raise HTTPException(409, "Cleanup session must refer to the original test cluster")
        require_cluster_access(current_principal(http_request), session.server_id)
        run.request = run.request.model_copy(update={'cluster_session_id': cluster_session_id})
    else:
        _optimization_run(run_id, http_request)
    return await optimizer.cancel(run_id)


@router.on_event("shutdown")
async def stop_measured_optimization():
    await optimizer.close()


@router.on_event("startup")
async def recover_measured_optimization():
    optimizer.recover()
