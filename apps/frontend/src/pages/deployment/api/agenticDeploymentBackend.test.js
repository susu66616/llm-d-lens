import assert from 'node:assert/strict';
import test from 'node:test';
import { candidateEvidence, defaultRequirements, DEFAULT_REQUIREMENTS, deploymentPlanningRequest, requirementsFromRun, streamAgenticRecommendation, streamAgenticRefinement } from './agenticDeploymentBackend.js';

const draft = { model: 'test/model', clusterId: 'cluster-a', deploymentName: 'test', storageVolumeId: 'cache-a', contextLength: 40960 };
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

test('both modes use explicit conservative assumptions without inventing performance targets', () => {
    for (const mode of ['manual', 'agentic']) {
        const request = deploymentPlanningRequest(draft, DEFAULT_REQUIREMENTS, 'session-a', mode, 'provider');
        assert.equal(request.planning_mode, mode);
        assert.equal(request.max_model_len, 40960);
        assert.equal(request.planning_facts.workload_profile.mean_input_tokens, 1024);
        assert.equal(request.planning_facts.workload_profile.mean_output_tokens, 256);
        assert.equal(request.planning_facts.workload_profile.shared_prefix_ratio, 0);
        assert.equal(request.planning_facts.ttft_slo_ms, null);
        assert.ok(!request.requirements.unknowns.includes('prefixReuse'));
        assert.equal(request.requirements.workload.prefillHeavy, false);
        assert.equal(request.requirements.workload.concurrency, null);
        assert.equal(request.ai_provider_id, mode === 'manual' ? null : 'provider');
        assert.equal(request.execution_policy.mode, 'approval_required');
        assert.equal(request.planning_facts.max_candidates, mode === 'manual' ? 10 : 3);
    }
});

test('suggested request sizes respect small model contexts without changing runtime context', () => {
    for (const context of [256, 512, 1024, 1280, 40960]) {
        const values = defaultRequirements(context);
        assert.ok(Number(values.inputTokens) + Number(values.outputTokens) <= context);
        const request = deploymentPlanningRequest({ ...draft, contextLength: context }, values, 'session-a', 'manual');
        assert.equal(request.max_model_len, context);
    }
    assert.deepEqual(defaultRequirements(undefined), DEFAULT_REQUIREMENTS);
});

test('cleared values and legacy unknown evidence remain unspecified when restoring', () => {
    const values = { ...DEFAULT_REQUIREMENTS, inputTokens: '', outputTokens: '', useCase: 'unknown', prefixReuse: 'unknown' };
    const request = deploymentPlanningRequest(draft, values, 'session-a', 'manual');
    assert.equal(request.requirements.workload.inputTokens, null);
    assert.equal(request.requirements.workload.prefixReuse, null);
    assert.deepEqual(requirementsFromRun({ request }), { ...values, maxCandidates: '10' });
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
    const request = deploymentPlanningRequest(draft, values, 'session-a', 'agentic');
    assert.equal(request.requirements.hard_constraints.providerRef, 'optimized-baseline');
    assert.equal(request.planning_facts.workload_profile.mean_input_tokens, 1024);
    assert.equal(request.planning_facts.workload_profile.p95_input_tokens, 8192);
    assert.equal(request.planning_facts.end_to_end_latency_slo_ms, 2000);
    assert.deepEqual(requirementsFromRun({ request }), values);
    assert.equal(deploymentPlanningRequest(draft, DEFAULT_REQUIREMENTS, 'session-a', 'manual').planning_facts.workload_profile.p95_input_tokens, null);
    for (const invalid of [{ p95InputTokens: '-1' }, { p95InputTokens: '2.5' }, { e2eMs: '0' }, { plannerPrompt: 'x'.repeat(1801) }]) {
        assert.throws(() => deploymentPlanningRequest(draft, { ...values, ...invalid }, 'session-a', 'agentic'));
    }
});
