import test from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { DeploymentPlanReview } from './DeploymentPlanReview.jsx';
test('default review exposes an accessible overview and optional 3D, without claiming live state', () => {
    const html = renderToStaticMarkup(<DeploymentPlanReview content={'apiVersion: v1\nkind: Service\nmetadata:\n  name: endpoint'} />);
    assert.match(html, /Overview/);
    assert.match(html, /3D explorer/);
    assert.match(html, /not live cluster inventory/);
    assert.match(html, /Inspect component/);
    assert.match(html, /Service: endpoint/);
});
test('agentic review labels candidate evidence before rendering', () => {
    const html = renderToStaticMarkup(<DeploymentPlanReview mode="agentic" model="test/model" candidate={{ provider_ref: 'optimized-baseline', replicas: 1, tensor_parallel_size: 2 }} />);
    assert.match(html, /Candidate topology before YAML generation/);
    assert.doesNotMatch(html, /aria-label="Generated YAML"/);
});


test('measured optimization exposes resource budgets and an explicit deployment action', async () => {
    const { MeasuredOptimizationControls } = await import('./MeasuredOptimizationControls.jsx');
    const html = renderToStaticMarkup(<MeasuredOptimizationControls run={{ id: 'plan', status: 'awaiting_approval', candidates: [] }} onUpdate={() => {}} onReview={() => {}} />);
    assert.match(html, /Maximum accelerators/);
    assert.match(html, /Maximum minutes/);
    assert.match(html, /Repetitions per candidate/);
    assert.match(html, /Start deployment and benchmark tests/);
    assert.match(html, /best tested candidate/);
});


test('trial settings restore recorded workload and prevent restarting deployed or unclean plans', async () => {
    const { MeasuredOptimizationControls } = await import('./MeasuredOptimizationControls.jsx');
    for (const [status, optimizationStatus] of [['deploying', 'completed'], ['awaiting_approval', 'cleanup_failed']]) {
        const run = { id: 'plan', status, candidates: [], optimization: { status: optimizationStatus, request: {
            candidate_ids: ['candidate-a'], max_gpu_count: 3, repetitions: 2,
            benchmark: { matrix: [{ isl: 1024, osl: 64 }], concurrency_stages: [{ concurrency: 8, num_requests: 200 }] },
        } } };
        const html = renderToStaticMarkup(<MeasuredOptimizationControls run={run} />);
        assert.match(html, /<fieldset disabled=""/);
        assert.match(html, /Input tokens<input[^>]*value="1024"/);
        assert.match(html, /Concurrent requests<input[^>]*value="8"/);
        if (optimizationStatus === 'cleanup_failed') assert.match(html, /Clean up interrupted tests/);
    }
});
