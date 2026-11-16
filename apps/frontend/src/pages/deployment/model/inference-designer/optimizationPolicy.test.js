import test from 'node:test';
import assert from 'node:assert/strict';
import { createBlueprint } from './blueprint';
import { DEFAULT_DEPLOYMENT_INTENT, deploymentIntentErrors, recommendOptimizations } from './optimizationPolicy';

const catalog = {
    guides: [{ id: 'optimized-baseline', title: 'Baseline', variables: {}, steps: { deploy: {
        main: [{ run: 'kubectl apply -k ${REPO_ROOT}/guides/optimized-baseline/modelserver/base' }],
    } } }],
    assets: [{ path: 'guides/optimized-baseline/modelserver/base', guide: 'optimized-baseline', kind: 'kustomize' }],
};
const draft = () => ({ ...createBlueprint(), selectedGuides: [], capabilityOverrides: {}, deploymentIntent: { ...DEFAULT_DEPLOYMENT_INTENT } });
const profile = { upstream_variant: 'fixture' };

test('workload inputs reject invalid bounds without inventing hardware defaults', () => {
    assert.deepEqual(deploymentIntentErrors(DEFAULT_DEPLOYMENT_INTENT), []);
    for (const value of ['0', '-1', '1.5', 'NaN', 'Infinity', '9007199254740992']) {
        assert.match(deploymentIntentErrors({ ...DEFAULT_DEPLOYMENT_INTENT, maxDevices: value }).join(' '), /positive integer/);
    }
    assert.throws(() => recommendOptimizations(catalog, draft(), null), /accelerator profile/);
});

test('recommendation preserves manual disable and does not provision external systems', () => {
    const result = recommendOptimizations(catalog, { ...draft(), capabilityOverrides: { prefix: false } }, profile);
    assert.equal(result.capabilityOverrides.prefix, false);
    for (const id of ['agentic', 'batch', 'shared-cache', 'snapshot', 'weights', 'scaling']) assert.notEqual(result.capabilityOverrides[id], true);
    assert.ok(result.optimizationDecisions.every(item => item.reason));
});

test('manual source edits protect engine parameters from automatic switches', () => {
    const result = recommendOptimizations(catalog, { ...draft(), sourceUnits: [{ path: 'fixture', edits: [{ path: [0], value: 1 }] }] }, profile);
    assert.equal(result.capabilityOverrides.chunked, undefined);
    assert.match(result.optimizationDecisions.find(item => item.id === 'chunked').reason, /manually edited/);
});

test('explicit locks survive changed optimization objectives', () => {
    const result = recommendOptimizations(catalog, {
        ...draft(), deploymentIntent: { ...DEFAULT_DEPLOYMENT_INTENT, objective: 'cost' },
        capabilityOverrides: { chunked: false }, automaticCapabilities: ['chunked'], optimizationLocks: ['chunked'],
    }, profile);
    assert.equal(result.capabilityOverrides.chunked, false);
});

test('both planning modes include compatible additions beyond workload presets', () => {
    const guide = { id: 'fixture-addon', title: 'Revision-owned addition', variables: {}, steps: { deploy: {
        main: [{ run: 'kubectl apply -k SOURCE/guides/fixture-addon/resources' }],
    } } };
    const source = { ...catalog, guides: [...catalog.guides, guide], assets: [...catalog.assets, { path: 'guides/fixture-addon/resources', guide: guide.id, kind: 'kustomize' }] };
    const manual = recommendOptimizations(source, { ...draft(), planningMode: 'manual' }, profile);
    const agentic = recommendOptimizations(source, { ...draft(), planningMode: 'agentic' }, profile);
    assert.equal(manual.capabilityOverrides['guide:fixture-addon'], true);
    assert.equal(agentic.capabilityOverrides['guide:fixture-addon'], true);
    const excluded = recommendOptimizations(source, { ...draft(), planningMode: 'manual', capabilityOverrides: { 'guide:fixture-addon': false } }, profile);
    assert.equal(excluded.capabilityOverrides['guide:fixture-addon'], false);
    assert.notEqual(manual.capabilityOverrides.tracing, true);
});
