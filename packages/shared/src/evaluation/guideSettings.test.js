import assert from 'node:assert/strict';
import test from 'node:test';
import { normalizeGuideSettings } from './guideSettings.js';

test('guide options validate values while source manifests determine variant support', () => {
    assert.throws(() => normalizeGuideSettings('optimized-baseline', '', { cacheCpuGiB: 20 }));
    assert.deepEqual(normalizeGuideSettings('pd-disaggregation', 'custom/deep', { rdmaNicCount: 2 }), { rdmaNicCount: 2 });
    assert.deepEqual(normalizeGuideSettings('tiered-prefix-cache', 'custom-cache', { cacheCpuGiB: 20 }), { cacheCpuGiB: 20 });
    assert.throws(() => normalizeGuideSettings('pd-disaggregation', 'vllm-rdma', { rdmaNicCount: 1.5 }));
    assert.deepEqual(normalizeGuideSettings('tiered-prefix-cache', 'native/cpu/base', { cacheCpuGiB: '20', routerValues: 'router: {}' }), { cacheCpuGiB: 20, routerValues: 'router: {}' });
    assert.deepEqual(normalizeGuideSettings('optimized-baseline', '', { cacheCpuGiB: '', rdmaNicCount: '' }), {});
});

test('P/D slices and router index sizes preserve integer contracts', () => {
    assert.deepEqual(normalizeGuideSettings('pd-disaggregation', '', { slices: '3' }), { slices: 3 });
    for (const slices of [0, -1, 1.5]) assert.throws(() => normalizeGuideSettings('pd-disaggregation', '', { slices }));
    assert.throws(() => normalizeGuideSettings('optimized-baseline', '', { slices: 2 }));
    assert.throws(() => normalizeGuideSettings('tiered-prefix-cache', '', { cacheLruCapacity: 1.5 }));
});


test('filesystem cache settings require tiered cache and explicit capacity', () => {
    assert.deepEqual(normalizeGuideSettings('tiered-prefix-cache', 'native/fs/base', { cacheStorageGiB: '100', cacheStorageClass: 'shared-rwx' }), { cacheStorageGiB: 100, cacheStorageClass: 'shared-rwx' });
    assert.throws(() => normalizeGuideSettings('tiered-prefix-cache', '', { cacheStorageClass: 'shared-rwx' }));
    assert.throws(() => normalizeGuideSettings('pd-disaggregation', '', { cacheStorageGiB: 100 }));
});


test('Guide adaptation does not add separate routing experiment controls', () => {
    for (const field of ['peakPrefillThroughput', 'maxPrefixTokensToMatch', 'kvLoadFailurePolicy']) {
        assert.throws(() => normalizeGuideSettings('pd-disaggregation', '', { [field]: 1 }), /Unknown Guide setting/);
    }
});
