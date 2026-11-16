import assert from 'node:assert/strict';
import test from 'node:test';
import yaml from 'js-yaml';
import { readFileSync } from 'node:fs';
import { DEPLOYMENT_PREVIEW, mockEnvironmentFromProfiles, createMockDeploymentPlan, checkMockDeployment } from './agenticDeploymentBackend.js';
import { candidateEvidence, defaultRequirements, DEFAULT_REQUIREMENTS, deploymentPlanningRequest, requirementsFromRun, streamAgenticRecommendation, streamAgenticRefinement } from './agenticDeploymentBackend.js';

const draft = { model: 'test/model', clusterId: 'cluster-a', deploymentName: 'test', storageVolumeId: 'cache-a', contextLength: 40960 };
const mockProfile = JSON.parse(readFileSync(new URL('../../../../../backend/src/llm_d_lens/hardware/profiles/intel_xpu.json', import.meta.url)));
const mockDraft = { model: DEPLOYMENT_PREVIEW.model, preview: mockEnvironmentFromProfiles([mockProfile]) };

test('mock uses registered hardware image and resource names without backend writes', t => {
    t.mock.method(globalThis, 'fetch', () => { throw new Error('Mock planning must not call a backend'); });
    const plan = createMockDeploymentPlan(mockDraft, 'manual');
    const [deployment] = yaml.loadAll(plan.yaml);
    const container = deployment.spec.template.spec.containers[0];
    assert.equal(container.image, mockProfile.deployment.runtime_image);
    assert.deepEqual(container.resources.limits, { [mockDraft.preview.resourceName]: 1 });
    assert.deepEqual(plan.enabled, ['parallel', 'prefix', 'chunked']);
    assert.match(plan.yaml, /MOCK ONLY/);
    assert.equal(plan.performance_estimate, undefined);
    assert.throws(() => mockEnvironmentFromProfiles([]), /profile is unavailable/);
});

test('mock objectives, edits and capacity checks agree with displayed YAML', () => {
    for (const [objective, replicas] of [['balanced', 2], ['latency', 1], ['throughput', 4]]) {
        assert.equal(createMockDeploymentPlan(mockDraft, 'agentic', objective).settings.replicas, replicas);
    }
    const plan = createMockDeploymentPlan(mockDraft, 'manual', 'balanced', { replicas: 3, tp: 2, prefixCaching: false, maxModelLen: 8192 });
    const [deployment] = yaml.loadAll(plan.yaml);
    assert.equal(deployment.spec.replicas, 3);
    assert.equal(plan.required, 6);
    assert.ok(!plan.enabled.includes('prefix'));
    assert.ok(deployment.spec.template.spec.containers[0].args.includes('--no-enable-prefix-caching'));
    assert.ok(deployment.spec.template.spec.containers[0].args.includes('8192'));
    assert.equal(checkMockDeployment(plan, mockDraft.preview).passed, false);
    const valid = createMockDeploymentPlan(mockDraft, 'manual');
    assert.equal(checkMockDeployment(valid, mockDraft.preview, 'capacity').passed, false);
    assert.equal(checkMockDeployment(valid, mockDraft.preview, 'ready').passed, true);
    assert.throws(() => createMockDeploymentPlan(mockDraft, 'manual', 'balanced', { tp: 3 }), /tp must be/);
    assert.throws(() => createMockDeploymentPlan(mockDraft, 'manual', 'balanced', { replicas: 1.5 }), /whole number/);
    assert.throws(() => createMockDeploymentPlan({ ...mockDraft, model: 'another/model' }, 'manual'), /Qwen3-0.6B/);
});
test('restoring a saved plan preserves requirements, unknowns and trial limits', () => {
    for (const values of [
        DEFAULT_REQUIREMENTS,
        { ...DEFAULT_REQUIREMENTS, objective: 'latency', maxAccelerators: '8', ttftMs: '200', tpotMs: '40',
            inputTokens: '1024', outputTokens: '256', requestRate: '0.5', prefixReuse: 'no',
            useCase: 'summarization', requiredFeatures: ['pd_disaggregation'],
            forbiddenFeatures: ['external_kv_cache'], maxCandidates: '5', maxDuration: '20', allowTrial: true },
    ]) {
        const request = deploymentPlanningRequest(draft, values, 'session-a', 'agentic', 'provider-a');
        assert.deepEqual(requirementsFromRun({ request }), values);
    }
});

test('both modes use editable general-chat workload traits by default', () => {
    for (const mode of ['manual', 'agentic']) {
        const request = deploymentPlanningRequest(draft, DEFAULT_REQUIREMENTS, 'session-a', mode, 'provider');
        assert.equal(request.planning_mode, mode);
        assert.equal(request.max_model_len, 40960);
        assert.equal(request.planning_facts.workload_profile.mean_input_tokens, null);
        assert.equal(request.planning_facts.workload_profile.mean_output_tokens, null);
        assert.equal(request.planning_facts.workload_profile.shared_prefix_ratio, 0);
        assert.equal(request.planning_facts.ttft_slo_ms, null);
        assert.ok(!request.requirements.unknowns.includes('prefixReuse'));
        assert.equal(request.requirements.workload.prefillHeavy, false);
        assert.equal(request.requirements.objective, 'balanced');
        assert.equal(request.requirements.workload.concurrency, null);
        assert.equal(request.ai_provider_id, mode === 'manual' ? null : 'provider');
        assert.equal(request.execution_policy.mode, 'approval_required');
        assert.equal(request.planning_facts.max_candidates, mode === 'manual' ? 10 : 3);
    }
});

test('model context is not mistaken for observed request length', () => {
    for (const context of [256, 512, 1024, 1280, 40960]) {
        const values = defaultRequirements(context);
        assert.equal(values.inputTokens, '');
        assert.equal(values.outputTokens, '');
        const request = deploymentPlanningRequest({ ...draft, contextLength: context }, values, 'session-a', 'manual');
        assert.equal(request.max_model_len, context);
    }
    assert.deepEqual(defaultRequirements(undefined), DEFAULT_REQUIREMENTS);
});

test('notes do not become intent and natural language requires an AI provider', () => {
    const target = { ...draft, description: 'Keep for next month' };
    assert.equal(deploymentPlanningRequest(target, DEFAULT_REQUIREMENTS, 'session-a', 'manual').planner_prompt, null);
    const values = { ...DEFAULT_REQUIREMENTS, plannerPrompt: 'Long documents with short answers' };
    assert.throws(() => deploymentPlanningRequest(target, values, 'session-a', 'agentic'), /Select an AI provider/);
    const request = deploymentPlanningRequest(target, values, 'session-a', 'agentic', 'ai');
    assert.equal(request.planner_prompt, values.plannerPrompt);
    assert.equal(requirementsFromRun({ request }).plannerPrompt, values.plannerPrompt);
    assert.equal(requirementsFromRun({ request: { ...request, planner_prompt: 'balanced priority.' } }).plannerPrompt, '');
});

test('cleared values remain unspecified while legacy unknown traits restore to defaults', () => {
    const values = { ...DEFAULT_REQUIREMENTS, inputTokens: '', outputTokens: '', useCase: 'unknown', prefixReuse: 'unknown' };
    const request = deploymentPlanningRequest(draft, values, 'session-a', 'manual');
    assert.equal(request.requirements.workload.inputTokens, null);
    assert.equal(request.requirements.workload.prefixReuse, null);
    assert.deepEqual(requirementsFromRun({ request }), { ...values, prefixReuse: 'no', useCase: 'general-chat', maxCandidates: '10' });
});

test('legacy plans without structured requirements restore editable defaults', () => {
    assert.deepEqual(requirementsFromRun({ request: {
        planner_prompt: 'latency priority. Prefer stable serving.',
        planning_facts: { max_candidates: 5 },
        execution_policy: { max_total_duration_minutes: 30, allow_trials: true },
    } }), {
        ...DEFAULT_REQUIREMENTS,
        plannerPrompt: 'Prefer stable serving.',
        maxCandidates: '5',
        maxDuration: '30',
        allowTrial: true,
        requiredFeatures: [],
        forbiddenFeatures: [],
    });
});
test('request mapping rejects invalid budgets instead of applying silent defaults', () => {
    for (const values of [
        { maxAccelerators: '0' }, { maxAccelerators: '1.5' }, { maxAccelerators: 'Infinity' },
        { inputTokens: '-4' }, { concurrency: '4', requestRate: '2' }, { maxCandidates: '11' },
    ]) assert.throws(() => deploymentPlanningRequest(draft, { ...DEFAULT_REQUIREMENTS, ...values }, 'session-a', 'manual'));
    const request = deploymentPlanningRequest(draft, { ...DEFAULT_REQUIREMENTS, maxAccelerators: '2', forbiddenFeatures: ['pd_disaggregation'] }, 'session-a', 'agentic');
    assert.equal(request.requirements.hard_constraints.maxAccelerators, 2);
    assert.deepEqual(request.requirements.forbidden_features, ['pd_disaggregation']);
});

test('only the measured winner is labelled best, not every ranked candidate', () => {
    const run = { selected_candidate: { id: 'a' }, candidates: [], optimization: {
        status: 'completed', best_candidate_id: 'b', ranking: [{ candidate_id: 'a' }, { candidate_id: 'b' }],
    } };
    assert.match(candidateEvidence(run), /not the verified winner/);
    run.selected_candidate.id = 'b';
    assert.match(candidateEvidence(run), /Best verified candidate/);
    run.selected_candidate.id = 'c';
    assert.match(candidateEvidence(run), /performance not verified/);
});

test('agentic recommendation stream preserves events across response chunks', async (t) => {
    const signal = new AbortController().signal;
    const encoder = new TextEncoder();
    const previousDocument = globalThis.document;
    globalThis.document = { cookie: 'prism_csrf=csrf-value-123' };
    t.after(() => { globalThis.document = previousDocument; });
    t.mock.method(globalThis, 'fetch', async (url, options) => {
        assert.equal(url, '/api/agentic-deployments/stream');
        assert.equal(options.signal, signal);
        assert.equal(options.headers['X-Prism-CSRF'], 'csrf-value-123');
        assert.deepEqual(JSON.parse(options.body), { model: 'Qwen/test' });
        return new Response(new ReadableStream({
            start(controller) {
                controller.enqueue(encoder.encode('event: progress\ndata: {"phase":"ai","message":"Wait'));
                controller.enqueue(encoder.encode('ing"}\n\nevent: complete\ndata: {"id":"run-1"}\n\n'));
                controller.close();
            },
        }));
    });
    const events = [];

    await streamAgenticRecommendation(
        { model: 'Qwen/test' }, (...event) => events.push(event), signal,
    );

    assert.deepEqual(events, [
        ['progress', { phase: 'ai', message: 'Waiting' }],
        ['complete', { id: 'run-1' }],
    ]);
});

test('agentic recalculation streams through the run-specific refine endpoint', async (t) => {
    const encoder = new TextEncoder();
    t.mock.method(globalThis, 'fetch', async (url, options) => {
        assert.equal(url, '/api/agentic-deployments/run%2F1/refine/stream');
        assert.deepEqual(JSON.parse(options.body), { planner_prompt: 'prefer latency' });
        return new Response(new ReadableStream({
            start(controller) {
                controller.enqueue(encoder.encode('event: progress\ndata: {"phase":"scoring","status":"running"}\n\n'));
                controller.enqueue(encoder.encode('event: complete\ndata: {"id":"run/1","status":"awaiting_approval"}\n\n'));
                controller.close();
            },
        }));
    });
    const events = [];

    await streamAgenticRefinement(
        'run/1', { planner_prompt: 'prefer latency' }, (...event) => events.push(event),
    );

    assert.deepEqual(events, [
        ['progress', { phase: 'scoring', status: 'running' }],
        ['complete', { id: 'run/1', status: 'awaiting_approval' }],
    ]);
});


test('explicit method, workload percentiles and all SLOs survive planning and restoration', () => {
    const values = { ...DEFAULT_REQUIREMENTS, providerRef: 'optimized-baseline', p95InputTokens: '8192',
        e2eMs: '2000', plannerPrompt: 'Prefer simpler operations.' };
    const request = deploymentPlanningRequest(draft, values, 'session-a', 'agentic', 'ai');
    assert.equal(request.requirements.hard_constraints.providerRef, 'optimized-baseline');
    assert.equal(request.planning_facts.workload_profile.mean_input_tokens, null);
    assert.equal(request.planning_facts.workload_profile.p95_input_tokens, 8192);
    assert.equal(request.planning_facts.end_to_end_latency_slo_ms, 2000);
    assert.deepEqual(requirementsFromRun({ request }), values);
    assert.equal(deploymentPlanningRequest(draft, DEFAULT_REQUIREMENTS, 'session-a', 'manual').planning_facts.workload_profile.p95_input_tokens, null);
    for (const invalid of [{ p95InputTokens: '-1' }, { p95InputTokens: '2.5' }, { e2eMs: '0' }, { plannerPrompt: 'x'.repeat(1801) }]) {
        assert.throws(() => deploymentPlanningRequest(draft, { ...values, ...invalid }, 'session-a', 'agentic'));
    }
});
