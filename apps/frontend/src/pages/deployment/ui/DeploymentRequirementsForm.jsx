import { useState } from 'react';
import { ArrowRight, CheckCircle2, SlidersHorizontal, Sparkles } from 'lucide-react';
import { Badge, Button, Checkbox, Input, Panel, Select, Textarea } from '../../../shared/ui';
import { DEFAULT_REQUIREMENTS } from '../api/agenticDeploymentBackend';

const mechanisms = [
    ['pd_disaggregation', 'Prefill / Decode separation'],
    ['prefix_routing', 'Prefix-aware routing'],
    ['external_kv_cache', 'External KV cache'],
];

export function DeploymentRequirementsForm({ mode = 'manual', initialValue, onSubmit, disabled = false, children }) {
    const [value, setValue] = useState({ ...DEFAULT_REQUIREMENTS, ...initialValue });
    const patch = key => event => setValue(current => ({ ...current, [key]: event.target.value }));
    const number = (key, label, placeholder, max) => <label className="block text-sm text-theme-text">{label}
        <Input className="mt-1.5" type="number" min={key === 'requestRate' ? '0.01' : '1'} step={key === 'requestRate' ? 'any' : '1'} max={max} value={value[key]} placeholder={placeholder} onChange={patch(key)} />
    </label>;
    const control = (id, setting) => setValue(current => ({
        ...current,
        requiredFeatures: [...current.requiredFeatures.filter(item => item !== id), ...(setting === 'required' ? [id] : [])],
        forbiddenFeatures: [...current.forbiddenFeatures.filter(item => item !== id), ...(setting === 'forbidden' ? [id] : [])],
    }));
    return <form onInvalid={event => event.target.closest('details')?.setAttribute('open', '')} onSubmit={event => { event.preventDefault(); onSubmit(value); }}>
        <Panel padding="none">
            <div className="grid lg:grid-cols-[minmax(0,1fr)_18rem]">
                <fieldset disabled={disabled} className="min-w-0 space-y-6 p-6">
                    <div>
                        <Badge tone={mode === 'manual' ? 'brand' : 'violet'}>{mode === 'manual' ? 'Manual configuration' : 'Goal and constraints'}</Badge>
                        <h3 className="mt-3 text-lg font-semibold text-theme-text">{mode === 'manual' ? 'Choose your deployment configuration' : 'What should the deployment achieve?'}</h3>
                        <p className="mt-1 text-sm text-theme-muted">Start with the suggested settings. You can inspect and edit the plan before anything is deployed.</p>
                    </div>
                    {mode === 'manual' && <label className="block text-sm text-theme-text">Deployment method
                        <Select className="mt-1.5" value={value.providerRef} onChange={patch('providerRef')}>
                            <option value="">Select a compatible method automatically</option>
                            <option value="optimized-baseline">Aggregated serving</option>
                            <option value="pd-disaggregation">Prefill / Decode separation</option>
                            <option value="precise-prefix-cache-routing">Prefix-aware routing</option>
                            <option value="tiered-prefix-cache">Tiered prefix cache</option>
                        </Select>
                        <span className="mt-2 block text-xs text-theme-muted">An explicit method is a hard constraint. Lens checks workload prerequisites and current cluster support; it will not silently substitute another method. Edit role replicas, parallelism and runtime settings in the generated plan.</span>
                    </label>}
                    <div className="grid gap-5 md:grid-cols-2">
                        <label className="block text-sm text-theme-text">Workload
                            <Select className="mt-1.5" value={value.useCase} onChange={patch('useCase')}>
                                <option value="general-chat">General chat (recommended)</option>
                                <option value="long-inputs">Long-input analysis</option>
                                <option value="summarization">Document summarization</option>
                                {value.useCase === 'unknown' && <option value="unknown">Baseline only (saved plan)</option>}
                            </Select>
                            <span className="mt-2 block text-xs text-theme-muted">Choose long-input analysis for large documents or long prompts.</span>
                        </label>
                        <label className="block text-sm text-theme-text">Optimize for
                            <Select className="mt-1.5" value={value.objective} onChange={patch('objective')}>
                                <option value="throughput">More throughput (recommended)</option>
                                <option value="latency">Faster responses</option>
                                <option value="cost">Fewer resources</option>
                                <option value="reliability">Higher availability</option>
                            </Select>
                            <span className="mt-2 block text-xs text-theme-muted">Used to choose between compatible configurations, not a performance guarantee.</span>
                        </label>
                    </div>
                    <label className="block text-sm text-theme-text">Do requests share a long system prompt or document?
                        <Select className="mt-1.5" value={value.prefixReuse} onChange={patch('prefixReuse')}>
                            <option value="no">Do not assume shared content (recommended)</option>
                            <option value="yes">Yes, requests reuse the same prefix</option>
                            <option value="unknown">Decide after observing traffic</option>
                        </Select>
                        <span className="mt-2 block text-xs text-theme-muted">Choose yes to consider prefix-aware routing and reusable cache where supported.</span>
                    </label>
                    {mode === 'agentic' && children}
                    {mode === 'agentic' && <label className="block text-sm text-theme-text">Deployment preference (optional)
                        <Textarea className="mt-1.5" value={value.plannerPrompt} maxLength={1800} onChange={patch('plannerPrompt')} placeholder="Explain tradeoffs the AI planner should consider." />
                        <span className="mt-2 block text-xs text-theme-muted">AI guidance supplements the structured limits below. Built-in rules use the structured fields; preference text cannot override resource or compatibility checks.</span>
                    </label>}
                    <details className="border-t border-theme-border pt-4">
                        <summary className="cursor-pointer text-sm font-medium text-theme-text"><SlidersHorizontal size={14} className="mr-2 inline" />Advanced settings <span className="font-normal text-theme-muted">— optional</span></summary>
                        <div className="mt-5 space-y-6">
                            <section className="space-y-3">
                                <h4 className="text-sm font-semibold">Expected request size</h4>
                                <p className="text-xs text-theme-muted">Suggested token counts are planning assumptions, not observed traffic. Adjust them to your workload; clearing a value leaves it unspecified.</p>
                                <div className="grid gap-4 md:grid-cols-2">{number('inputTokens', 'Mean input tokens per request', 'Not specified')}{number('p95InputTokens', 'P95 input tokens per request', 'Unknown; not inferred from mean')}{number('outputTokens', 'Mean output tokens per request', 'Not specified')}</div>
                            </section>
                            <section className="space-y-3">
                                <h4 className="text-sm font-semibold">Resource and performance limits</h4>
                                <div className="grid gap-4 md:grid-cols-2">
                                    {number('maxAccelerators', 'Maximum accelerators', 'Use currently free capacity')}
                                    {number('concurrency', 'Concurrent requests', 'No load target')}
                                    {number('requestRate', 'Request rate (QPS)', 'No load target')}
                                    {number('ttftMs', 'P95 time to first token (ms)', 'No latency target')}
                                    {number('tpotMs', 'P95 time per output token (ms)', 'No latency target')}
                                    {number('e2eMs', 'P95 end-to-end latency (ms)', 'No latency target')}
                                </div>
                                <p className="text-xs text-theme-muted">Specify concurrency or request rate, not both. Only set a latency target if your application requires one. Targets need prediction or measurement to verify.</p>
                            </section>
                            <section className="space-y-3">
                                <h4 className="text-sm font-semibold">Optimization controls</h4>
                                <p className="text-xs text-theme-muted">Leave these on Automatic unless you need a specific architecture. Required options must pass compatibility and resource checks.</p>
                                <div className="grid gap-4 md:grid-cols-3">{mechanisms.map(([id, label]) => <label key={id} className="text-sm">{label}<Select className="mt-1.5" value={value.requiredFeatures.includes(id) ? 'required' : value.forbiddenFeatures.includes(id) ? 'forbidden' : 'auto'} onChange={event => control(id, event.target.value)}>
                                    <option value="auto">Automatic</option><option value="required">Required</option><option value="forbidden">Disabled</option>
                                </Select></label>)}</div>
                            </section>
                            {mode === 'agentic' && <section className="space-y-3">
                                <h4 className="text-sm font-semibold">Search budget</h4>
                                <div className="grid gap-4 md:grid-cols-2">{number('maxCandidates', 'Maximum retained candidates', '3', 10)}{number('maxDuration', 'Planning timeout (minutes)', '60', 180)}</div>
                                <Checkbox label="Enable optional deployment and benchmark trials after review" checked={value.allowTrial} onChange={event => setValue(current => ({ ...current, allowTrial: event.target.checked }))} />
                                <p className="text-xs text-theme-muted">Trials require a separate confirmation and budget. Enabling this does not start a deployment.</p>
                            </section>}
                        </div>
                    </details>
                </fieldset>
                <aside className="space-y-6 border-t border-theme-border bg-theme-bg/50 p-6 lg:border-l lg:border-t-0">
                    <div><h4 className="flex items-center gap-2 text-sm font-semibold"><Sparkles size={16} />Lens handles the setup</h4>
                        <ul className="mt-4 space-y-3">{['Read available cluster capacity', 'Use model metadata and hardware profiles', 'Size replicas and parallelism', 'Check compatible optimizations'].map(label => <li key={label} className="flex gap-2 text-xs text-theme-muted"><CheckCircle2 size={14} className="mt-0.5 shrink-0" />{label}</li>)}</ul>
                    </div>
                    <div className="border-t border-theme-border pt-5"><h4 className="text-sm font-semibold">What happens next</h4>
                        <p className="mt-2 text-xs leading-relaxed text-theme-muted">{mode === 'manual' ? 'Lens builds a compatible plan without an LLM. Review its topology, adjust components, then approve the deployment.' : 'Lens compares candidate plans for your goal. Review the recommendation and alternatives, then approve the deployment.'}</p>
                    </div>
                    <Badge tone="neutral">No deployment until approval</Badge>
                </aside>
            </div>
            <div className="flex flex-wrap items-center justify-between gap-3 border-t border-theme-border px-6 py-4">
                <p className="text-xs text-theme-muted">Existing deployments will not be changed.</p>
                <Button type="submit" isLoading={disabled} disabled={disabled}>{mode === 'manual' ? 'Generate plan and topology' : 'Compare deployment plans'}<ArrowRight size={15} /></Button>
            </div>
        </Panel>
    </form>;
}
