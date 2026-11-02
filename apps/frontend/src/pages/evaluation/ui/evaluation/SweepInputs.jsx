import { useState } from 'react';
import { configurationRuntimeValues } from '../../model/configuration.js';
import { Button, Checkbox, Input, Label, Panel, Select } from '../../../../shared/ui';

const PARAMETERS = [
    ['guide:slices', 'P/D slices'],
    ['guide:cacheCpuGiB', 'CPU cache capacity (GiB)'],
    ['guide:cacheLruCapacity', 'CPU cache index blocks per server'],
    ['guide:rdmaNicCount', 'RDMA NICs per pod'],
    ['tensorParallelSize', 'Serving / Decode TP'], ['replicas', 'Serving / Decode replicas'],
    ['prefillTensorParallelSize', 'Prefill TP'], ['prefillReplicas', 'Prefill replicas'],
    ['maxNumSeqs', 'Maximum sequences'], ['maxNumBatchedTokens', 'Maximum batched tokens'],
    ['gpuMemoryUtilization', 'Memory utilization'], ['maxModelLen', 'Maximum model length'],
    ['blockSize', 'Block size'],
];

export const defaultSweep = () => ({ strategy: 'auto', dimensions: {}, resource_mode: 'maximum', accelerator_budget: 1, max_trials: 12, max_duration_seconds: 14400, objective: 'throughput', confirmation_trials: 2, seed: 0 });

export default function SweepInputs({ value, onChange, provider, runtime, baseContent = {}, onPreview, preview, busy, error }) {
    const [advanced, setAdvanced] = useState({ kind: 'argument', target: 'both', name: '', values: '' });
    const [advancedError, setAdvancedError] = useState('');
    const runtimeValues = configurationRuntimeValues(baseContent);
    const serving = baseContent.decode || baseContent.serving || {};
    const baseValues = {
        replicas: serving.replicaCount, tensorParallelSize: serving.tensorParallelSize,
        prefillReplicas: baseContent.prefill?.replicaCount, prefillTensorParallelSize: baseContent.prefill?.tensorParallelSize,
        ...runtimeValues,
        ...Object.fromEntries(Object.entries(baseContent.guideSettings || {}).map(([key, number]) => [`guide:${key}`, number])),
    };
    const visibleParameters = PARAMETERS.filter(([key]) => {
        if (key.startsWith('prefill') || ['guide:slices', 'guide:rdmaNicCount'].includes(key)) return provider === 'pd-disaggregation';
        if (key.startsWith('guide:cache')) return provider === 'tiered-prefix-cache';
        return ['replicas', 'tensorParallelSize'].includes(key) || runtime === 'vllm';
    });
    const patch = update => onChange({ ...value, ...update });
    const dimension = (key, update) => patch({ dimensions: { ...value.dimensions, [key]: { ...value.dimensions[key], ...update } } });
    return <Panel title="Deployment parameter search">
        <Label><Checkbox checked={Boolean(value)} onChange={event => onChange(event.target.checked ? defaultSweep() : null)} /> Search parameter ranges</Label>
        {value && <div className="mt-4 space-y-4 text-theme-text">
            <p className="text-sm text-theme-muted">Select one base configuration. Unselected parameters stay fixed. The server generates candidates and validates each configuration before deployment.</p>
            <div className="grid gap-4 sm:grid-cols-2">
                <Label>Search strategy<Select value={value.strategy} onChange={event => patch({ strategy: event.target.value })}><option value="auto">Automatic exploration and refinement</option><option value="grid">Complete sampled grid</option></Select></Label>
                <Label>Objective<Select value={value.objective} onChange={event => patch({ objective: event.target.value })}><option value="goodput">SLO goodput</option><option value="throughput">Output throughput</option><option value="latency">P99 time to first token</option><option value="resource_efficiency">Throughput per accelerator</option></Select></Label>
                <Label>Resource constraint<Select value={value.resource_mode} onChange={event => patch({ resource_mode: event.target.value })}><option value="maximum">Maximum accelerator count</option><option value="fixed">Fixed accelerator count</option></Select></Label>
                {[['accelerator_budget', 'Accelerator budget'], ['max_trials', 'Maximum configurations'], ['max_duration_seconds', 'Time budget (seconds)'], ['confirmation_trials', 'Confirmation runs within budget']].map(([key, label]) => <Label key={key}>{label}<Input type="number" min={1} value={value[key]} onChange={event => patch({ [key]: Number(event.target.value) })} /></Label>)}
            </div>
            <div className="space-y-3">{visibleParameters.map(([key, label]) => <div key={key} className="rounded border border-theme-border p-3">
                <Label><Checkbox checked={Boolean(value.dimensions[key])} onChange={event => { const dimensions = { ...value.dimensions }; if (event.target.checked) dimensions[key] = { minimum: Number(baseValues[key]) > 0 ? Number(baseValues[key]) : null, maximum: Number(baseValues[key]) > 0 ? Number(baseValues[key]) : null }; else delete dimensions[key]; patch({ dimensions }); }} />{label}</Label>
                {value.dimensions[key] && <div className="mt-2 grid grid-cols-3 gap-3">{['minimum', 'maximum', 'step'].map(bound => <Label key={bound}>{bound}<Input aria-label={`${label} ${bound}`} type="number" step={key === 'gpuMemoryUtilization' ? 0.01 : 1} value={value.dimensions[key][bound] ?? ''} placeholder={bound === 'step' ? 'Automatic' : 'Required'} onChange={event => dimension(key, { [bound]: event.target.value === '' ? null : Number(event.target.value) })} /></Label>)}</div>}
            </div>)}</div>
            <p className="text-sm text-theme-muted">Goodput ranking requires measured SLO goodput from a compatible rate workload. Automatic search samples ranges within the configuration budget. Grid search runs every resource-valid sampled combination. Runtime compatibility is checked during planning; a range does not imply every TP value is supported.</p>
            <details className="rounded border border-theme-border p-3"><summary>Advanced parameter options</summary>
                <p className="my-3 text-sm text-theme-muted">Use the range controls above for supported deployment parameters. Additional runtime arguments and environment variables accept explicit choices. Use only options supported by your selected runtime. Boolean choices can be written as true,false.</p>
                <div className="grid gap-3 sm:grid-cols-2">
                    <Label>Parameter type<Select value={advanced.kind} onChange={event => setAdvanced({ ...advanced, kind: event.target.value })}><option value="argument">Runtime argument</option><option value="environment">Environment variable</option></Select></Label>
                    <Label>Applies to<Select value={advanced.target} onChange={event => setAdvanced({ ...advanced, target: event.target.value })}><option value="both">All model servers</option><option value="prefill">Prefill</option><option value="decode">Decode</option></Select></Label>
                    <Label>Parameter name<Input value={advanced.name} onChange={event => setAdvanced({ ...advanced, name: event.target.value })} /></Label>
                    <Label>Choices, separated by commas<Input value={advanced.values} onChange={event => setAdvanced({ ...advanced, values: event.target.value })} /></Label>
                </div>
                <Button variant="secondary" onClick={() => {
                    const choices = advanced.values.split(',').map(item => item.trim());
                    if (!advanced.name.trim() || choices.some(item => !item)) { setAdvancedError('Enter a parameter name and nonempty choices.'); return; }
                    const key = `${advanced.kind}:${advanced.target}:${advanced.name.trim()}`;
                    const values = choices.map(item => item === 'true' ? true : item === 'false' ? false : Number.isFinite(Number(item)) ? Number(item) : item);
                    dimension(key, { values }); setAdvancedError('');
                }}>Add search parameter</Button>
                {advancedError && <p role="alert">{advancedError}</p>}
                {Object.entries(value.dimensions).filter(([key]) => key.includes(':') && !PARAMETERS.some(([name]) => name === key)).map(([key, domain]) => <div key={key} className="mt-3 flex items-center justify-between gap-3"><span>{key}: {(domain.values || []).join(', ')}</span><Button variant="ghost" onClick={() => { const dimensions = { ...value.dimensions }; delete dimensions[key]; patch({ dimensions }); }}>Remove</Button></div>)}
            </details>
            <Button variant="secondary" disabled={busy || !Object.keys(value.dimensions).length} onClick={onPreview}>Preview search</Button>
            {error && <p role="alert" className="text-theme-text">{error}</p>}
            {preview && <div aria-live="polite"><p>{preview.candidates.length} resource-valid candidates; {preview.excluded.length} excluded. At most {value.max_trials} configurations will run.</p><div className="max-h-64 overflow-auto"><table className="w-full text-left text-sm"><thead><tr><th>Parameters</th><th>Accelerators</th></tr></thead><tbody>{preview.candidates.slice(0, 30).map(item => <tr key={item.id}><td>{Object.entries(item.parameters).map(([key, number]) => `${key}=${number}`).join(', ')}</td><td>{item.accelerator_count}</td></tr>)}</tbody></table></div><p className="text-sm text-theme-muted">Showing up to 30 candidates. Later selections depend on measurements. Preview does not create deployments.</p></div>}
        </div>}
    </Panel>;
}
