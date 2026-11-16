import assert from 'node:assert/strict';
import test from 'node:test';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { MODELS } from '@repo/shared/catalog';
import { AuthContext } from '../../auth/AuthContext.js';
import AgenticDeploymentWorkspace from '../../deployment/ui/AgenticDeploymentWorkspace.jsx';
import ManualDeploymentWorkspace from '../../deployment/ui/ManualDeploymentWorkspace.jsx';
import { DeploymentRequirementsForm } from '../../deployment/ui/DeploymentRequirementsForm.jsx';
import { DeploymentPlanReview } from '../../deployment/ui/DeploymentPlanReview.jsx';
import ModelMarketPage, { ModelCard } from './ModelMarketPage.jsx';
import { standardRecommendation } from '../model/standardRecommendation.js';

const authenticated = child => createElement(AuthContext.Provider, { value: { can: () => true } }, child);

test('model cards retain catalog styling, natural height and full-width model type', () => {
    const html = renderToStaticMarkup(createElement(ModelCard, { model: MODELS[0], onSelect() {} }));
    assert.match(html, /type="button"/);
    assert.match(html, /\bh-auto\b/);
    assert.doesNotMatch(html, /\bh-8\b/);
    assert.match(html, /\bwhitespace-normal\b/);
    assert.match(html, /dark:bg-slate-900/);
    assert.match(html, /dark:hover:border-cyan-400\/60/);
    assert.match(html, /class="col-span-2"><dt[^>]*>Model type/);
});

test('deployment context uses the exact model card metadata for every listed context', () => {
    for (const model of MODELS.filter((model) => model.contextLength != null)) {
        assert.equal(standardRecommendation(model, null).maxModelLen, model.contextLength, model.repository);
    }
    const qwen = MODELS.find((model) => model.repository === 'Qwen/Qwen3-0.6B');
    assert.equal(standardRecommendation(qwen, null).maxModelLen, 40960);
});

test('deployment context keeps the default only when model card context is missing', () => {
    for (const contextLength of [null, undefined]) {
        assert.equal(standardRecommendation({ contextLength, sizeGiB: 1 }, null).maxModelLen, 4096);
    }
});

test('both deployment modes identify the model and cluster without redundant numbered stages', () => {
    for (const Workspace of [ManualDeploymentWorkspace, AgenticDeploymentWorkspace]) {
        const html = renderToStaticMarkup(authenticated(createElement(Workspace, { draft: { model: 'fixture/model', modelTitle: 'Fixture Model', clusterName: 'Research cluster', deploymentName: 'long-generated-name' }, embedded: true })));
        assert.match(html, /<h2[^>]*>(?:Manual|Agentic) plan<\/h2>/);
        assert.equal((html.match(/Fixture Model/g) || []).length, 1);
        assert.match(html, /Research cluster/);
        assert.doesNotMatch(html, /Deployment target details|Planning mode<\/dt>/);
        assert.doesNotMatch(html, /aria-label="Deployment workflow"|<h2[^>]*>long-generated-name/);
        assert.doesNotMatch(html, /Custom vLLM parameters|GPU capacity for planning|Template-based deployment/);
    }
});

test('workload inputs are optional hints, not assumed traffic or compatibility gates', () => {
    for (const mode of ['manual', 'agentic']) {
        const html = renderToStaticMarkup(createElement(DeploymentRequirementsForm, { mode }));
        assert.match(html, /Mean input tokens<input[^>]*value=""/);
        assert.match(html, /Mean output tokens<input[^>]*value=""/);
        assert.match(html, /Concurrent requests<input[^>]*value=""/);
        assert.match(html, /Use this or QPS, not both/);
        assert.doesNotMatch(html, />Unknown</);
        assert.match(html, /<details[^>]*><summary[^>]*>Request size/);
        assert.match(html, /No live deployment is changed/);
        assert.match(html, /does not switch optimizations on or off/);
        assert.doesNotMatch(html, /<details[^>]* open|aria-label="Requirement steps"|>Continue</);
        const visible = html.replace(/<details[\s\S]*?<\/details>/g, '');
        assert.doesNotMatch(visible, /<input|<textarea|<select/);
        assert.match(html, /No required parameters/);
        if (mode === 'agentic') assert.match(visible, />Balanced<|>Low latency<|>High throughput</);
        assert.doesNotMatch(html, /Tensor parallel|gpu-memory-utilization/);
    }
});

test('Manual review restores 3D topology and allows component editing', () => {
    const html = renderToStaticMarkup(createElement(DeploymentPlanReview, {
        candidate: { id: 'a', replicas: 1, tensor_parallel_size: 1, provider_ref: 'optimized-baseline' },
        model: 'test/model',
        onEdit() {},
        decisions: [
            { capability_id: 'prefix_cache', label: 'Prefix caching', status: 'dependency', reason: 'Required by the cache connector.' },
            { capability_id: 'autoscaling', label: 'Autoscaling', status: 'unknown', reason: 'Missing source support.' },
        ],
    }));
    assert.match(html, /Prefix caching/);
    assert.match(html, /2D overview/);
    assert.match(html, /Inference flow/);
    assert.match(html, /Verify/);
    assert.match(html, /3D explorer/);
    assert.doesNotMatch(html, /Needs information/);
});

test('model catalog preserves search and removes legacy pre-generation runtime controls', t => {
    let hasReturnDraft = false;
    const original = Object.getOwnPropertyDescriptor(globalThis, 'sessionStorage');
    Object.defineProperty(globalThis, 'sessionStorage', {
        configurable: true,
        value: {
            getItem: key => hasReturnDraft && key === 'prism_model_market_return_draft' ? JSON.stringify({ model: MODELS[0].repository }) : null,
            removeItem() {},
        },
    });
    t.after(() => { if (original) Object.defineProperty(globalThis, 'sessionStorage', original); else delete globalThis.sessionStorage; });
    const renderPage = canDeploy => renderToStaticMarkup(createElement(AuthContext.Provider, { value: { can: () => canDeploy } }, createElement(ModelMarketPage, { onNavigate() {} })));
    const catalog = renderPage(true);
    assert.match(catalog, /role="tablist" aria-label="Model categories"/);
    assert.match(catalog, /aria-label="Search models"/);
    assert.match(catalog, /class="w-52 sm:w-64"/);
    assert.match(catalog, /lg:w-auto lg:flex-1/);
    assert.match(catalog, /grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3/);
    hasReturnDraft = true;
    const html = renderPage(true);
    assert.match(html, />Model Catalog<\/button>/);
    assert.match(html, />Manual</);
    assert.match(html, />Agentic</);
    assert.match(html, /Compose, then adjust/);
    assert.match(html, /Start with your objective/);
    assert.match(html, /Choose how to build your plan/);
    assert.equal((html.match(/type="submit"/g) || []).length, 1);
    assert.ok(html.indexOf('id="deployment-mode"') < html.indexOf('id="deployment-target"'));
    assert.match(html, /1–3 Walkthrough/);
    assert.match(html, /Choose an environment/);
    assert.match(html, /Try Intel XPU mock/);
    assert.doesNotMatch(html, /Current setup|Change target|Prepare model files|No storage volumes available/);
    assert.doesNotMatch(html, /aria-current="step"|1\. Target|2\. Deployment plan/);
    assert.match(html, /Build Manual plan/);
    assert.doesNotMatch(html, /Name and notes|Notes<textarea|Complete the workload picture/);
    assert.doesNotMatch(html, />Replicas<|>Tensor parallelism<|Custom vLLM parameters|Template-based deployment/);
    assert.match(renderPage(false), /Your role cannot create deployments/);
});

test('saved constraints remain visible and unchanged when the simplified form is reopened', () => {
    const html = renderToStaticMarkup(createElement(DeploymentRequirementsForm, {
        mode: 'agentic', initialValue: { objective: 'reliability', maxAccelerators: '8', ttftMs: '500', tpotMs: '20', inputTokens: '4096', forbiddenFeatures: ['external_kv_cache'] },
    }));
    assert.match(html, /aria-pressed="true"[^>]*>Reliability/);
    assert.match(html, /value="500"/);
    assert.match(html, /value="20"/);
    assert.match(html, /value="4096"/);
    assert.match(html, /<details open="">/);
    assert.doesNotMatch(html, /Predicted TTFT|GPU rental/);
});
