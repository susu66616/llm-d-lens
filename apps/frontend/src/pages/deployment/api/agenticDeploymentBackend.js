import { streamSseRequest } from '../../../shared/api/sse.js';
import yaml from 'js-yaml';

export const DEFAULT_REQUIREMENTS = {
    objective: 'balanced', maxAccelerators: '', ttftMs: '', tpotMs: '', e2eMs: '', p95InputTokens: '', providerRef: '', plannerPrompt: '',
    inputTokens: '', outputTokens: '', concurrency: '', requestRate: '',
    prefixReuse: 'no', useCase: 'general-chat', requiredFeatures: [], forbiddenFeatures: [],
    maxCandidates: '3', maxDuration: '60', allowTrial: false,
};

export function defaultRequirements() {
    return { ...DEFAULT_REQUIREMENTS, requiredFeatures: [], forbiddenFeatures: [] };
}

export const OPTIMIZATION_CONTROLS = [
    ['pd_disaggregation', 'Prefill / Decode separation'],
    ['prefix_routing', 'Prefix-aware routing'],
    ['external_kv_cache', 'CPU KV cache'],
    ['prefix_cache', 'Engine prefix cache'],
    ['chunked_prefill', 'Chunked prefill'],
];

export function candidateLabel(candidate) {
    if (!candidate) return 'Deployment plan';
    const parts = [candidate.prefill_replicas ? 'Separate prefill and decode' : 'Combined serving'];
    if (candidate.prefix_routing || candidate.provider_ref === 'precise-prefix-cache-routing') parts.push('prefix-aware routing');
    if (candidate.native_cpu_cache_gib || candidate.provider_ref === 'tiered-prefix-cache') parts.push('CPU KV cache');
    return parts.join(' + ');
}

export function requirementsFromRun(run) {
    const request = run?.request || {};
    const requirements = request.requirements;
    if (!requirements) {
        return {
            ...defaultRequirements(),
            plannerPrompt: request.planner_prompt?.replace(/^(?:balanced|throughput|latency|cost|reliability) priority\.(?: Compare at most \d+ candidates\.)?\s*/, '') || '',
            maxCandidates: String(request.planning_facts?.max_candidates ?? DEFAULT_REQUIREMENTS.maxCandidates),
            maxDuration: String(request.execution_policy?.max_total_duration_minutes ?? DEFAULT_REQUIREMENTS.maxDuration),
            allowTrial: Boolean(request.execution_policy?.allow_trials),
        };
    }
    const workload = requirements.workload || {};
    const text = value => String(value ?? '');
    return {
        objective: requirements.objective,
        maxAccelerators: text(requirements.hard_constraints?.maxAccelerators),
        ttftMs: text(requirements.performance_targets?.ttftMs),
        tpotMs: text(requirements.performance_targets?.tpotMs),
        e2eMs: text(requirements.performance_targets?.e2eMs),
        p95InputTokens: text(workload.p95InputTokens),
        providerRef: text(requirements.hard_constraints?.providerRef),
        plannerPrompt: request.planner_prompt?.replace(/^(?:balanced|throughput|latency|cost|reliability) priority\.(?: Compare at most \d+ candidates\.)?\s*/, '') || '',
        inputTokens: text(workload.inputTokens), outputTokens: text(workload.outputTokens),
        concurrency: text(workload.concurrency), requestRate: text(workload.requestRate),
        prefixReuse: workload.prefixReuse == null ? DEFAULT_REQUIREMENTS.prefixReuse : workload.prefixReuse ? 'yes' : 'no',
        useCase: workload.prefillHeavy == null ? DEFAULT_REQUIREMENTS.useCase : request.planning_facts?.use_case || DEFAULT_REQUIREMENTS.useCase,
        requiredFeatures: [...(requirements.required_features || [])],
        forbiddenFeatures: [...(requirements.forbidden_features || [])],
        maxCandidates: text(request.planning_facts?.max_candidates || DEFAULT_REQUIREMENTS.maxCandidates),
        maxDuration: text(request.execution_policy?.max_total_duration_minutes || DEFAULT_REQUIREMENTS.maxDuration),
        allowTrial: Boolean(request.execution_policy?.allow_trials),
    };
}

export function deploymentPlanningRequest(draft, values, clusterSessionId, mode, aiProviderId = '') {
    const number = (key, integer = false) => {
        const value = values[key];
        if (value === '' || value == null) return null;
        const parsed = Number(value);
        if (!Number.isFinite(parsed) || parsed <= 0 || (integer && !Number.isSafeInteger(parsed))) {
            throw new Error(`${key} must be a positive ${integer ? 'whole ' : ''}number.`);
        }
        return parsed;
    };
    if (!draft?.model || !draft?.clusterId || !draft?.deploymentName || !clusterSessionId) {
        throw new Error('Choose a model, cluster and deployment name before planning.');
    }
    if (!draft.storageVolumeId) throw new Error('Choose a ready model cache before planning.');
    const concurrency = number('concurrency', true), requestRate = number('requestRate');
    if (concurrency && requestRate) throw new Error('Choose concurrency or request rate, not both.');
    const maxCandidates = number('maxCandidates', true) ?? 3;
    const maxDuration = number('maxDuration', true) ?? 60;
    if (maxCandidates > 10 || maxDuration > 180) throw new Error('Search is limited to 10 candidates and 180 minutes.');
    const budget = number('maxAccelerators', true);
    const inputTokens = number('inputTokens', true), outputTokens = number('outputTokens', true);
    const ttft = number('ttftMs', true), tpot = number('tpotMs', true), e2e = number('e2eMs', true);
    const p95InputTokens = number('p95InputTokens', true);
    const plannerPrompt = mode === 'agentic' ? (values.plannerPrompt || '').trim() : '';
    if (plannerPrompt.length > 1800) throw new Error('Deployment preference must be at most 1800 characters.');
    if (plannerPrompt && !aiProviderId) throw new Error('Select an AI provider to interpret your goal, or use the structured traffic and limits.');
    const prefillHeavy = ['long-inputs', 'summarization'].includes(values.useCase);
    const prefixReuse = values.prefixReuse === 'unknown' ? null : values.prefixReuse === 'yes';
    const workload = { inputTokens, p95InputTokens, outputTokens, concurrency, requestRate, prefixReuse, prefillHeavy: values.useCase === 'unknown' ? null : prefillHeavy };
    return {
        model: draft.model, deployment_name: draft.deploymentName, description: draft.description || '',
        cluster_session_id: clusterSessionId, storage_type: 'model-cache', storage_volume_id: draft.storageVolumeId,
        replicas: 1, tensor_parallel_size: 1,
        max_model_len: draft.contextLength, gpu_memory_utilization: draft.gpuMemoryUtilization,
        max_num_seqs: draft.maxNumSeqs, max_num_batched_tokens: draft.maxNumBatchedTokens,
        vllm_arguments: draft.vllmArguments || [],
        planning_mode: mode, ai_provider_id: mode === 'agentic' ? aiProviderId || null : null,
        planner_prompt: plannerPrompt || null,
        execution_policy: { mode: 'approval_required', max_total_duration_minutes: maxDuration, allow_trials: mode === 'agentic' && Boolean(values.allowTrial) },
        planning_facts: {
            model_weight_gib: draft.modelWeightGiB, vram_per_gpu_gib: 1, free_gpu_count: 0,
            hardware_capacity_mode: 'free', context_length: draft.contextLength,
            operator_preference: values.objective, max_candidates: mode === 'manual' ? 10 : maxCandidates,
            ttft_slo_ms: ttft, tpot_slo_ms: tpot, end_to_end_latency_slo_ms: e2e,
            use_case: values.useCase === 'unknown' ? 'general-chat' : values.useCase,
            workload_profile: {
                mean_input_tokens: inputTokens, p95_input_tokens: p95InputTokens,
                mean_output_tokens: outputTokens, concurrency, request_rate: requestRate,
                shared_prefix_ratio: prefixReuse === null ? null : prefixReuse ? 1 : 0,
            },
        },
        requirements: {
            model: draft.model, cluster_id: draft.clusterId, objective: values.objective,
            hard_constraints: { maxAccelerators: budget, ...(values.providerRef ? { providerRef: values.providerRef } : {}) },
            workload, performance_targets: { ttftMs: ttft, tpotMs: tpot, e2eMs: e2e },
            required_features: values.requiredFeatures || [], forbidden_features: values.forbiddenFeatures || [],
            unknowns: Object.keys(workload).filter(key => workload[key] == null),
        },
    };
}

export function candidateEvidence(run) {
    const id = run.selected_candidate?.id;
    const optimization = run.optimization;
    const measured = optimization?.ranking?.find(item => item.candidate_id === id);
    if (measured && optimization.best_candidate_id === id && optimization.status === 'completed') {
        return 'Best verified candidate within the tested workload and budget';
    }
    if (measured) return 'Measured candidate; not the verified winner';
    const selected = run.candidates?.find(item => item.id === id);
    if (selected?.performance_estimate_source === 'aic_estimate') return 'Predicted performance; not measured here';
    return 'Recommendation; performance not verified';
}

export function streamAgenticRecommendation(payload, onEvent, signal) {
    return streamSseRequest('/api/agentic-deployments/stream', payload, onEvent, signal);
}

export function streamAgenticRefinement(runId, payload, onEvent, signal) {
    return streamSseRequest(`/api/agentic-deployments/${encodeURIComponent(runId)}/refine/stream`, payload, onEvent, signal);
}

// Explicit browser-only fixture. No cluster record, session, or deployment API is created.
export const DEPLOYMENT_PREVIEW = Object.freeze({
    id: 'intel-xpu', label: 'Intel XPU', model: 'Qwen/Qwen3-0.6B',
    clusterId: 'mock-intel-xpu', clusterName: 'Intel XPU sandbox',
    nodes: 2, devices: 4, deviceModel: 'Intel Arc Pro B60', memoryGiB: 24,
});

export function mockEnvironmentFromProfiles(profiles) {
    const profile = profiles.find(item => item.id === DEPLOYMENT_PREVIEW.id);
    const resourceName = profile?.deployment?.modes?.plugin?.resource_names?.at(-1);
    if (!profile?.deployment?.runtime_image || !resourceName) throw new Error('The registered hardware profile is unavailable. Retry loading the mock environment.');
    return { ...DEPLOYMENT_PREVIEW, profile, resourceName, accessMode: 'plugin' };
}

export function createMockDeploymentPlan(draft, mode, objective = 'balanced', overrides = {}) {
    const environment = draft.preview;
    if (!environment || draft.model !== DEPLOYMENT_PREVIEW.model) throw new Error('This mock scenario uses Qwen3-0.6B. Reopen the mock environment.');
    const defaults = {
        replicas: mode === 'manual' || objective === 'latency' ? 1 : objective === 'throughput' ? 4 : 2,
        tp: 1, maxModelLen: 4096, maxNumSeqs: objective === 'latency' ? 32 : 256,
        prefixCaching: true, chunkedPrefill: true,
    };
    const settings = { ...defaults, ...overrides };
    for (const [key, max] of [['replicas', 16], ['tp', 2], ['maxModelLen', 40960], ['maxNumSeqs', 4096]]) {
        const min = key === 'maxModelLen' ? 256 : 1;
        if (!Number.isSafeInteger(settings[key]) || settings[key] < min || settings[key] > max) throw new Error(`${key} must be a whole number between ${min} and ${max}.`);
    }
    const required = settings.replicas * settings.tp;
    const labels = { app: 'mock-modelserver' };
    const annotations = { 'lens-preview': 'mock-only-not-a-validated-deployment' };
    const args = [draft.model, '--tensor-parallel-size', String(settings.tp), '--max-model-len', String(settings.maxModelLen), '--max-num-seqs', String(settings.maxNumSeqs),
        settings.prefixCaching ? '--enable-prefix-caching' : '--no-enable-prefix-caching',
        settings.chunkedPrefill ? '--enable-chunked-prefill' : '--no-enable-chunked-prefill'];
    const documents = [
        { apiVersion: 'apps/v1', kind: 'Deployment', metadata: { name: 'mock-modelserver', annotations }, spec: {
            replicas: settings.replicas, selector: { matchLabels: labels }, template: { metadata: { labels }, spec: {
                nodeSelector: environment.profile.deployment.node_selector,
                containers: [{ name: 'modelserver', image: environment.profile.deployment.runtime_image, command: ['vllm', 'serve'], args,
                    ports: [{ containerPort: 8000 }], resources: { requests: { [environment.resourceName]: settings.tp }, limits: { [environment.resourceName]: settings.tp } },
                    env: [{ name: 'HF_HOME', value: '/models' }], volumeMounts: [{ name: 'model-cache', mountPath: '/models' }] }],
                volumes: [{ name: 'model-cache', persistentVolumeClaim: { claimName: 'mock-model-cache' } }],
            } },
        } },
        { apiVersion: 'v1', kind: 'Service', metadata: { name: 'mock-modelserver', annotations }, spec: { selector: labels, ports: [{ port: 8000, targetPort: 8000 }] } },
    ];
    return {
        settings, required, objective,
        enabled: ['parallel', ...(settings.prefixCaching ? ['prefix'] : []), ...(settings.chunkedPrefill ? ['chunked'] : [])],
        rationale: mode === 'manual' ? 'All three optimizations covered by this mock are enabled by default.'
            : objective === 'latency' ? 'Low latency example: one replica with a smaller batch limit.'
                : objective === 'throughput' ? 'High throughput example: four replicas use the mock device budget.'
                    : 'Balanced example: two replicas leave two devices available.',
        yaml: '# MOCK ONLY — model-server excerpt, not a complete llm-d release.\n# Gateway, EPP, model-cache contents and runtime compatibility are not validated.\n' + documents.map(doc => yaml.dump(doc, { lineWidth: -1 })).join('---\n'),
    };
}

export function checkMockDeployment(plan, environment, scenario = 'ready') {
    const available = scenario === 'capacity' ? 0 : environment.devices;
    return { available, required: plan.required, passed: plan.required <= available,
        message: plan.required <= available ? 'Mock capacity check passed' : `Needs ${plan.required} devices; ${available} available. Reduce replicas or restore mock capacity.` };
}
