import { createBlueprint, blueprintClusterProfiles } from './blueprint';
import { compositionDefaults, compositionRequest, resolveCompositionDraft, compositionResult } from './compositionPolicy';
import { recommendOptimizations, DEFAULT_DEPLOYMENT_INTENT } from './optimizationPolicy';
import { generateSourceBundle, loadSourceCatalog } from '../../ui/inference-designer/sourceCompositionBackend';
import { rebaseFinalConfiguration } from './sourceParameters';
import { getHardwareProfiles } from '../../../clusters/ui/ClusterMonitoringStack/hardwareProfilesBackend';
import { loadClusterOverview } from '../../../optimization/ui/OptimizationWorkspace/clusterBackend';
import { listModelCacheEntries } from '../../../model-cache/ui/ModelCache/modelCacheBackend';
import { hasPendingSync, sourceRepository } from '../../../model-cache/ui/ModelCache/modelCachePresentation';
import { listStorageVolumes } from '../../../storage/ui/StorageManagement/storageManagementBackend';
import { postJson } from '../../../../shared/api/httpClient';

export const catalogCapability = id => ({
    prefix_cache: 'prefix', chunked_prefill: 'chunked', pd_disaggregation: 'transfer',
    prefix_routing: 'affinity', external_kv_cache: 'cache', autoscaling: 'scaling', speculative_decoding: 'speculation',
}[id] || id);

export function candidateCompositionDraft(run, candidate, catalog, profile, entry, volume) {
    const requirements = run.request.requirements;
    const overrides = {};
    for (const id of requirements.required_features) overrides[catalogCapability(id)] = true;
    for (const id of requirements.forbidden_features) overrides[catalogCapability(id)] = false;
    const guide = candidate.provider_ref === 'baseline-vllm' ? 'optimized-baseline' : candidate.provider_ref;
    // CPU cache is a donor component, not a second complete serving Guide.
    const selectedGuides = [guide === 'tiered-prefix-cache' ? 'optimized-baseline' : guide];
    if (candidate.prefix_routing !== undefined && overrides.affinity === undefined && (run.request.planning_mode !== 'manual' || candidate.prefix_routing)) overrides.affinity = candidate.prefix_routing;
    if (candidate.native_cpu_cache_gib) {
        if (overrides.prefix === false || overrides.cache === false) throw new Error('CPU KV cache conflicts with a forbidden cache capability.');
        overrides.prefix = true;
        delete overrides.cache;
    }
    let draft = {
        ...createBlueprint(), sourceMode: true, sourceValidate: false,
        clusterId: requirements.cluster_id, model: run.request.model,
        modelCacheEntryId: entry.id, storageVolumeId: volume.id,
        profileId: profile.id, accelerator: profile.upstream_variant, image: '',
        sourceCommit: catalog.commit, sourceNamespace: volume.dynamicPvc?.namespace || 'llm-d-bench-storage',
        sourceVariables: compositionDefaults(catalog), selectedGuides, capabilityOverrides: overrides, guideChoices: {},
        planningMode: run.request.planning_mode,
        nativeCpuCacheGiB: candidate.native_cpu_cache_gib ?? undefined,
        deploymentIntent: { ...DEFAULT_DEPLOYMENT_INTENT, objective: requirements.objective,
            scenario: requirements.workload.prefillHeavy ? 'agent' : 'chat',
            contextLength: String(run.request.max_model_len), concurrency: String(run.request.max_num_seqs),
            maxDevices: requirements.hard_constraints.maxAccelerators == null ? '' : String(requirements.hard_constraints.maxAccelerators) },
        runtimeConfiguration: {
            profileId: profile.id, tp: candidate.tensor_parallel_size, replicas: candidate.replicas,
            prefillTp: candidate.prefill_tensor_parallel_size, prefillReplicas: candidate.prefill_replicas,
            maxModelLen: run.request.max_model_len, maxNumSeqs: run.request.max_num_seqs,
            maxNumBatchedTokens: run.request.max_num_batched_tokens, gpuMemoryUtilization: run.request.gpu_memory_utilization,
            customParameters: run.request.vllm_arguments.map(item => ({ ...item, target: 'both', kind: 'argument' })),
        },
    };
    if (candidate.guide_variant) draft.sourceVariables[guide] = {
        ...draft.sourceVariables[guide], MODEL_SERVER: 'vllm', VARIANT: candidate.guide_variant,
    };
    draft = { ...draft, ...recommendOptimizations(catalog, draft, profile) };
    return resolveCompositionDraft(catalog, draft);
}

export async function renderDeploymentPlan(run, { compare = false, editedDraft, profileId } = {}) {
    const clusterId = run.request.requirements.cluster_id;
    const [catalog, hardware, overview, entries, volumes] = await Promise.all([
        loadSourceCatalog(clusterId), getHardwareProfiles(), loadClusterOverview(clusterId),
        listModelCacheEntries({ clusterId, storageVolumeId: run.request.storage_volume_id }),
        listStorageVolumes({ clusterId }),
    ]);
    const profiles = blueprintClusterProfiles(overview, hardware.profiles || []);
    const selectedProfile = editedDraft?.profileId || profileId || run.configuration_plan?.source_draft?.profileId;
    const profile = profiles.find(item => item.id === selectedProfile) || (!selectedProfile && profiles.length === 1 ? profiles[0] : null);
    if (!profile?.upstream_variant) throw Object.assign(new Error('Select a supported accelerator profile for this cluster, then retry resource generation.'), { profiles });
    const entry = entries.find(item => item.status === 'ready' && !hasPendingSync(item)
        && sourceRepository(item) === run.request.model && item.storageVolumeId === run.request.storage_volume_id);
    const volume = volumes.find(item => item.id === entry?.storageVolumeId && item.status === 'ready');
    if (!entry || !volume) throw new Error('The selected cached model is no longer ready and synchronized.');
    const candidates = compare ? run.candidates : run.candidates.filter(item => item.id === run.selected_candidate.id);
    const proposals = await Promise.all(candidates.map(async candidate => {
        try {
            const savedDraft = editedDraft || run.configuration_plan?.source_candidates?.[candidate.id]?.draft
                || (candidate.id === run.selected_candidate.id ? run.configuration_plan?.source_draft : null);
            let draft = savedDraft ? resolveCompositionDraft(catalog, savedDraft)
                : candidateCompositionDraft(run, candidate, catalog, profile, entry, volume);
            if (draft.sourceCommit !== catalog.commit) throw new Error('The source revision changed. Rebuild the plan before editing.');
            if (draft.finalConfiguration) {
                const base = await generateSourceBundle({ ...draft, finalConfiguration: undefined, sourceValidate: false });
                draft = { ...draft, finalConfiguration: rebaseFinalConfiguration(draft.finalConfiguration, base) };
            }
            return { candidate_id: candidate.id, draft, composition: compositionRequest(draft, run.request.cluster_session_id) };
        } catch (error) {
            return { candidate_id: candidate.id, error: error.message };
        }
    }));
    return postJson(`/api/agentic-deployments/${encodeURIComponent(run.id)}/composition`, {
        revision: run.revision, compare, candidates: proposals,
    });
}

export function deploymentEditorSnapshot(run) {
    const plan = run.configuration_plan;
    if (!plan?.source_bundle || !plan.source_draft) throw new Error('Generate the current source composition before editing.');
    return { draft: plan.source_draft, result: compositionResult(plan.source_bundle, plan.source_draft) };
}
