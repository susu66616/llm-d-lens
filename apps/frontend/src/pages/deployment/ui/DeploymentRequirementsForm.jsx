import { useState } from 'react';
import { ArrowRight } from 'lucide-react';
import { Badge, Button, Input, Modal, Panel, Select, Textarea, ToggleGroup } from '../../../shared/ui';
import { DEFAULT_REQUIREMENTS } from '../api/agenticDeploymentBackend';
import { OPTIMIZATION_OBJECTIVES } from '../model/inference-designer/optimizationPolicy';

export function DeploymentRequirementsForm({ mode = 'manual', initialValue, onSubmit, disabled = false, runtime, resourceEstimate, aiProviderConfigured = false, objectivesOnly = false, submitLabel, children }) {
    const [value, setValue] = useState({ ...DEFAULT_REQUIREMENTS, ...initialValue });
    const patch = key => event => setValue(current => ({ ...current, [key]: event.target.value }));
    const hasValues = keys => keys.some(key => initialValue?.[key] !== '' && initialValue?.[key] != null);
    const number = (key, label, placeholder, max) => <label className="block text-sm text-theme-text">{label}
        <Input className="mt-1.5" type="number" min={key === 'requestRate' ? '0.01' : '1'} step={key === 'requestRate' ? 'any' : '1'} max={max} value={value[key]} placeholder={placeholder} onChange={patch(key)} />
    </label>;
    const objectives = OPTIMIZATION_OBJECTIVES.filter(([id]) => ['balanced', 'latency', 'throughput', value.objective].includes(id));
    return <form id="deployment-requirements" onInvalid={event => event.target.closest('details')?.setAttribute('open', '')} onSubmit={event => { event.preventDefault(); onSubmit(value); }}>
        <Panel title={mode === 'manual' ? 'Manual setup' : 'What should Agentic prioritize?'} actions={<Badge tone="info">No required parameters</Badge>}>
            <fieldset disabled={disabled} className="space-y-5">
                {mode === 'agentic' ? <div className="space-y-3"><ToggleGroup className="flex-wrap" size="sm" options={objectives.map(([id, label]) => ({ value: id, label, disabled }))} value={value.objective} onChange={objective => setValue(current => ({ ...current, objective }))} /><p className="text-sm text-theme-muted">{objectivesOnly ? 'Mock recommendation · simulated rules' : aiProviderConfigured ? 'AI-assisted recommendation' : 'Rule-based recommendation · AI is optional'}. Performance is not verified by planning.</p></div>
                    : <p className="text-sm text-theme-muted">Combine supported compatible optimizations within discovered capacity. Generate with the defaults, then adjust the plan.</p>}
                {(runtime?.max_model_len || resourceEstimate?.available_accelerators != null) && <p className="text-xs text-theme-muted">
                    {runtime?.max_model_len ? `Context ${runtime.max_model_len.toLocaleString()} tokens` : ''}
                    {runtime?.max_model_len && resourceEstimate?.available_accelerators != null ? ' · ' : ''}
                    {resourceEstimate?.available_accelerators != null ? `${resourceEstimate.available_accelerators} accelerators available` : ''}
                </p>}
                {!objectivesOnly && mode === 'agentic' && <details open={Boolean(initialValue?.plannerPrompt)}>
                    <summary className="cursor-pointer text-sm font-medium">Describe a goal <span className="font-normal text-theme-muted">/ optional AI assistance</span></summary>
                    <div className="mt-3 space-y-4">
                        {children}
                        <label className="block text-sm">Goal for the AI planner (optional)
                            <Textarea className="mt-1.5" value={value.plannerPrompt} maxLength={1800} onChange={patch('plannerPrompt')} placeholder="For example: serve document analysis with short responses." />
                        </label>
                        <p className="text-xs text-theme-muted">A written goal requires an AI provider. Enter numeric limits explicitly in the sections below.</p>
                    </div>
                </details>}
                {!objectivesOnly && <><details open={hasValues(['maxAccelerators', 'concurrency', 'requestRate'])}>
                    <summary className="cursor-pointer text-sm font-medium">Resource limit and workload <span className="font-normal text-theme-muted">/ optional</span></summary>
                    <div className="mt-3 grid items-start gap-4 md:grid-cols-3">
                        {number('maxAccelerators', 'Accelerator limit', 'Use available capacity')}
                        {number('concurrency', 'Concurrent requests', 'Unconstrained')}
                        {number('requestRate', 'Request rate (QPS)', 'Unconstrained')}
                    </div>
                    <p className="mt-3 text-xs text-theme-muted">Concurrency refines capacity estimates. Use this or QPS, not both. The resource limit constrains the topology; traffic does not switch optimizations on or off.</p>
                </details>
                <details open={hasValues(['inputTokens', 'outputTokens', 'p95InputTokens'])}>
                    <summary className="cursor-pointer text-sm font-medium">Request size <span className="font-normal text-theme-muted">/ optional KV memory estimates</span></summary>
                    <div className="mt-3 grid gap-4 md:grid-cols-3">
                        {number('inputTokens', 'Mean input tokens', 'Optional estimate')}
                        {number('outputTokens', 'Mean output tokens', 'Optional estimate')}
                        {number('p95InputTokens', 'P95 input tokens', 'Optional estimate')}
                    </div>
                </details>
                <details open={hasValues(['ttftMs', 'tpotMs', 'e2eMs'])}>
                    <summary className="cursor-pointer text-sm font-medium">Latency targets <span className="font-normal text-theme-muted">/ optional SLOs</span></summary>
                    <div className="mt-3 grid gap-4 md:grid-cols-3">
                        {number('ttftMs', 'P95 TTFT (ms)', 'Unconstrained')}
                        {number('tpotMs', 'P95 TPOT (ms)', 'Unconstrained')}
                        {number('e2eMs', 'P95 E2E (ms)', 'Unconstrained')}
                    </div>
                    <p className="mt-2 text-xs text-theme-muted">These are requirements, not predicted results. Blank targets stay unspecified.</p>
                </details>
                </>}
                {value.providerRef && <p role="status" className="text-sm">This saved plan pins a deployment method. <Button type="button" size="xs" variant="link" onClick={() => setValue(current => ({ ...current, providerRef: '' }))}>Remove method restriction</Button></p>}
                <div id="deployment-generate" className="flex flex-wrap items-center justify-between gap-3 border-t border-theme-border pt-4">
                    <p className="text-xs text-theme-muted">No live deployment is changed.</p>
                    <Button type="submit" isLoading={disabled} disabled={disabled}>{submitLabel || (runtime ? 'Rebuild plan' : mode === 'manual' ? 'Build Manual plan' : 'Build Agentic plan')}<ArrowRight size={15} /></Button>
                </div>
            </fieldset>
        </Panel>
    </form>;
}

/** Uses the same planning inputs and provider contract as the initial recommendation. */
export function DeploymentRefinementDialog({ isOpen, onOpen, onClose, onSubmit, available = true, disabled = false, editing = false, requirements, runtime, resourceEstimate, mock = false, provider, providers = [], onProviderChange }) {
    if (!available) return null;
    return <><div className="flex flex-wrap items-center justify-between gap-3"><p className="text-sm text-theme-muted">{editing ? 'Apply or discard your edits before requesting another plan.' : 'Want a different plan?'}</p><Button variant="secondary" disabled={disabled || editing} onClick={onOpen}>Refine with Agentic<ArrowRight size={14} /></Button></div>
        <Modal isOpen={isOpen} onClose={onClose} title="Refine with Agentic" size="lg"
        subtitle="Generate a separate recommendation. You can return to your previous plan.">
        <DeploymentRequirementsForm mode="agentic" disabled={disabled || editing} initialValue={requirements} runtime={runtime} resourceEstimate={resourceEstimate} objectivesOnly={mock}
            aiProviderConfigured={Boolean(provider)} submitLabel="Suggest Agentic plan" onSubmit={onSubmit}>
            <PlanningProviderSelect provider={provider} providers={providers} onChange={onProviderChange} />
        </DeploymentRequirementsForm>
    </Modal></>;
}

export function PlanningProviderSelect({ provider, providers, onChange }) {
    return <label className="block text-sm">AI provider<Select className="mt-1" value={provider} onChange={event => onChange(event.target.value)}>
        <option value="">Rule-based recommendations (no AI)</option>
        {providers.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}
    </Select><span className="mt-2 block text-xs text-theme-muted">Natural-language goals require an AI provider. Rules use the structured traffic and limits.</span></label>;
}
