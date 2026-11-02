import assert from 'node:assert/strict';
import test from 'node:test';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import SweepInputs, { defaultSweep } from './SweepInputs.jsx';

const render = props => renderToStaticMarkup(createElement(SweepInputs, { onChange() {}, ...props }));
test('disabled search preserves single configuration mode', () => {
    assert.doesNotMatch(render({ value: null }), /Search strategy/);
});
test('ranges expose bounds without requiring manually added configurations', () => {
    const html = render({ value: { ...defaultSweep(), dimensions: { tensorParallelSize: { minimum: 1, maximum: 8 } } }, runtime: 'vllm', provider: 'optimized-baseline' });
    assert.match(html, /Serving \/ Decode TP minimum/);
    assert.match(html, /Serving \/ Decode TP maximum/);
    assert.doesNotMatch(html, /Prefill TP/);
    assert.match(html, /Maximum batched tokens/);
});
test('runtime-specific dimensions are hidden for undeclared runtimes', () => {
    assert.doesNotMatch(render({ value: defaultSweep(), runtime: 'unknown' }), /Maximum batched tokens/);
});

test('Guide ranges are presented directly without manually entering Guide parameter names', () => {
    const tiered = render({ value: defaultSweep(), runtime: 'vllm', provider: 'tiered-prefix-cache' });
    assert.match(tiered, /CPU cache capacity/);
    assert.match(tiered, /CPU cache index blocks per server/);
    assert.doesNotMatch(tiered, /P\/D slices|RDMA NICs per pod/);
    const pd = render({ value: defaultSweep(), runtime: 'vllm', provider: 'pd-disaggregation' });
    assert.match(pd, /P\/D slices/);
    assert.match(pd, /RDMA NICs per pod/);
    assert.doesNotMatch(pd, /CPU cache index blocks per server/);
    assert.doesNotMatch(pd, /<option value="guide">/);
});
