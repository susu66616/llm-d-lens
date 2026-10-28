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
import ModelMarketPage from './ModelMarketPage.jsx';
import { standardRecommendation } from '../model/standardRecommendation.js';

const authenticated = child => createElement(AuthContext.Provider, { value: { can: () => true } }, child);

test('deployment context uses the exact model metadata and only defaults missing values', () => {
    for (const model of MODELS.filter(item => item.contextLength != null)) {
        assert.equal(standardRecommendation(model, null).maxModelLen, model.contextLength, model.repository);
    }
    for (const contextLength of [null, undefined]) {
        assert.equal(standardRecommendation({ contextLength, sizeGiB: 1 }, null).maxModelLen, 4096);
    }
});

test('both deployment modes use the same staged workflow without legacy parameter forms', () => {
    for (const Workspace of [ManualDeploymentWorkspace, AgenticDeploymentWorkspace]) {
        const html = renderToStaticMarkup(authenticated(createElement(Workspace, { draft: {}, embedded: true })));
        assert.match(html, /aria-label="Deployment workflow"/);
        assert.match(html, /Deployment goals/);
        assert.match(html, /Review and edit/);
        assert.match(html, /Deploy and observe/);
        assert.doesNotMatch(html, /Custom vLLM parameters|GPU capacity for planning|Template-based deployment/);
    }
});

test('both modes provide a short form with explicit defaults and collapsed optional controls', () => {
    for (const mode of ['manual', 'agentic']) {
        const html = renderToStaticMarkup(createElement(DeploymentRequirementsForm, { mode }));
        assert.match(html, /General chat \(recommended\)/);
        assert.match(html, /Do not assume shared content/);
        assert.match(html, /Mean input tokens per request<input[^>]*value="1024"/);
        assert.match(html, /Mean output tokens per request<input[^>]*value="256"/);
        assert.match(html, /Concurrent requests<input[^>]*value=""/);
        assert.match(html, /concurrency or request rate, not both/);
        assert.match(html, /<details[^>]*><summary[^>]*>.*Advanced settings/);
        assert.match(html, /No deployment until approval/);
        assert.match(html, /planning assumptions, not observed traffic/);
        assert.doesNotMatch(html, /<details[^>]* open|aria-label="Requirement steps"|>Continue</);
        assert.doesNotMatch(html, /Tensor parallel|gpu-memory-utilization/);
    }
});

test('Manual review prioritizes topology and allows component editing without requiring 3D', () => {
    const html = renderToStaticMarkup(createElement(DeploymentPlanReview, {
        candidate: { id: 'a', replicas: 1, tensor_parallel_size: 1, provider_ref: 'optimized-baseline' },
        model: 'test/model',
        onEdit() {},
        decisions: [
            { capability_id: 'prefix_cache', label: 'Prefix caching', status: 'dependency', reason: 'Required by the cache connector.' },
            { capability_id: 'autoscaling', label: 'Autoscaling', status: 'unknown', reason: 'Missing source support.' },
        ],
    }));
    assert.match(html, /View all decisions \(2\)/);
    assert.match(html, /Prefix caching/);
    assert.match(html, /Overview/);
    assert.match(html, /3D explorer/);
    assert.match(html, />Edit component</);
    assert.doesNotMatch(html, /Optimization selection checklist|Missing source support/);
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
    hasReturnDraft = true;
    const html = renderPage(true);
    assert.match(html, />Model Catalog<\/button>/);
    assert.match(html, />Manual</);
    assert.match(html, />Agentic</);
    assert.match(html, /Continue to deployment goals/);
    assert.doesNotMatch(html, />Replicas<|>Tensor parallelism<|Custom vLLM parameters|Template-based deployment/);
    assert.match(renderPage(false), /Your role cannot create deployments/);
});
