import { lazy, Suspense, useEffect, useMemo, useState } from 'react';
import { Box, Download, Pencil, Undo2 } from 'lucide-react';
import { Badge, Button, Input, LoadingState, Panel, Select } from '../../../shared/ui';
import { downloadBlob } from '../../../shared/lib/download';
import { deploymentPlan, candidatePlan } from '../model/deploymentPlan';
import { DeploymentPlanGraph } from './DeploymentPlanGraph';

const Scene = lazy(() => import('./DeploymentPlanScene'));
const tones = { enabled: 'success', dependency: 'info', excluded: 'neutral', unknown: 'warning', conflict: 'warning', user_disabled: 'neutral', rejected: 'danger' };
const decisionLabels = { enabled: 'Enabled', dependency: 'Automatic dependency', excluded: 'Not enabled', unknown: 'Needs confirmation', conflict: 'Alternative combination', user_disabled: 'User disabled' };
const runtimeFields = [
    ['max_model_len', 'Maximum context tokens', 256, 1048576, 1],
    ['gpu_memory_utilization', 'Accelerator memory utilization', 0.5, 0.98, 0.01],
    ['max_num_seqs', 'Maximum sequences', 1, 4096, 1],
    ['max_num_batched_tokens', 'Maximum batched tokens', 256, 131072, 1],
];

export function DeploymentPlanReview({ content = '', candidate, model, mode = 'manual', decisions = [], onEdit, onEditingChange, disabled = false, validation, revision = 1, maxAccelerators, runtime }) {
    const [view, setView] = useState('2d'), [selected, setSelected] = useState('');
    const [editing, setEditing] = useState(false), [replicas, setReplicas] = useState(''), [tp, setTp] = useState('');
    const [error, setError] = useState(''), [applying, setApplying] = useState(false);
    const [runtimeDraft, setRuntimeDraft] = useState({});
    useEffect(() => { onEditingChange?.(editing); }, [editing, onEditingChange]);
    const parsed = useMemo(() => {
        try { return { plan: content ? deploymentPlan(content) : candidatePlan(candidate, model) }; }
        catch (failure) { return { error: failure.message, plan: { nodes: [], edges: [] } }; }
    }, [content, candidate, model]);
    const plan = parsed.plan;
    const node = plan.nodes.find(item => item.id === selected) || plan.nodes.find(item => item.tier === 1 && item.role !== 'encode') || plan.nodes[0];
    const selectedId = node?.id || '';
    const select = id => {
        if (editing && dirty) { setError('Apply or cancel the pending edit before selecting another component.'); return; }
        setSelected(id); setEditing(false); setError('');
    };
    const prefix = node?.role === 'prefill' ? 'prefill_' : '';
    const previousReplicas = candidate?.[`${prefix}replicas`] ?? node?.document?.spec?.replicas;
    const previousTp = candidate?.[`${prefix}tensor_parallel_size`];
    const dirty = (replicas !== '' && Number(replicas) !== previousReplicas)
        || (tp !== '' && Number(tp) !== previousTp)
        || Object.entries(runtimeDraft).some(([key, value]) => value !== '' && Number(value) !== runtime?.[key]);
    const changedResources = previousReplicas && previousTp && (replicas || tp)
        ? Number(replicas || previousReplicas) * Number(tp || previousTp) - previousReplicas * previousTp : null;
    const currentResources = candidate ? candidate.replicas * candidate.tensor_parallel_size + (candidate.prefill_replicas || 0) * (candidate.prefill_tensor_parallel_size || 0) : null;
    const overBudget = maxAccelerators != null && changedResources != null && currentResources + changedResources > maxAccelerators;
    const apply = async () => {
        const patch = {};
        for (const [key, value] of [['replicas', replicas], ['tensorParallelSize', tp]]) {
            if (value === '' || Number(value) === (key === 'replicas' ? previousReplicas : previousTp)) continue;
            if (!Number.isSafeInteger(Number(value)) || Number(value) < 1 || Number(value) > 128) {
                setError('Enter whole numbers between 1 and 128.'); return;
            }
            patch[key] = Number(value);
        }
        for (const [key, label, min, max, step] of runtimeFields) {
            if (runtimeDraft[key] == null || runtimeDraft[key] === '' || Number(runtimeDraft[key]) === runtime?.[key]) continue;
            const value = Number(runtimeDraft[key]);
            if (!Number.isFinite(value) || value < min || value > max || (step === 1 && !Number.isSafeInteger(value))) {
                setError(`${label} must be ${step === 1 ? 'a whole number ' : ''}between ${min} and ${max}.`); return;
            }
            patch[key] = value;
        }
        if (!Object.keys(patch).length || overBudget) return;
        setApplying(true); setError('');
        try {
            await onEdit({ node, patch });
            setEditing(false); setReplicas(''); setTp(''); setRuntimeDraft({});
        } catch (failure) { setError(failure.message || 'Unable to update configuration.'); }
        finally { setApplying(false); }
    };
    const beginEdit = () => {
        setReplicas(String(previousReplicas ?? '')); setTp(String(previousTp ?? ''));
        setRuntimeDraft(Object.fromEntries(runtimeFields.map(([key]) => [key, String(runtime?.[key] ?? '')])));
        setEditing(true); setError('');
    };
    return <Panel title="Deployment workbench" actions={<Badge tone="neutral">Revision {revision} / {mode === 'manual' ? 'Manual' : 'Agentic'}</Badge>} className="min-w-0">
        <p className="mb-4 text-sm text-theme-muted">{content
            ? 'The topology and YAML show the same generated resources, not live cluster inventory. Planned placement is not actual placement.'
            : 'Candidate topology before YAML generation. Resource relationships appear after rendering; performance remains unverified.'}</p>
        <div className="mb-4 flex flex-wrap gap-2" role="group" aria-label="Configuration view">
            {[['2d', 'Overview'], ['reason', 'Configuration basis'], ['validation', 'Validation'], ...(content ? [['yaml', 'YAML']] : []), ['3d', '3D explorer']].map(([id, label]) => <Button key={id} size="sm" variant={view === id ? 'primary' : 'secondary'} aria-pressed={view === id} onClick={() => setView(id)}>{label}</Button>)}
            {content && <Button size="sm" variant="ghost" className="ml-auto" onClick={() => downloadBlob(new Blob([content], { type: 'application/yaml' }), `deployment-v${revision}.yaml`)}><Download size={14} />Download YAML</Button>}
        </div>
        {parsed.error && <p role="alert">Unable to display resources: {parsed.error}. Do not use the graph as a configuration reference.</p>}
        {view === 'reason' && <div className="mb-5 space-y-3">
            <h3 className="text-sm font-semibold">Optimization selection checklist</h3>
            <p className="text-sm text-theme-muted">Mechanism decisions describe the selected configuration, not measured benefits. Unsupported combinations are not silently enabled.</p>
            {decisions.length ? <div className="overflow-auto"><table className="w-full text-left text-sm"><thead><tr className="border-b border-theme-border"><th className="p-2">Optimization</th><th className="p-2">Status</th><th className="p-2">Reason</th></tr></thead><tbody>{decisions.map((item, index) => <tr key={item.capability_id || item.id || index} className="border-b border-theme-border">
                <td className="p-2 font-medium">{item.label || item.capability_id || item.id}</td><td className="p-2"><Badge tone={tones[item.status] || 'neutral'}>{decisionLabels[item.status] || item.status}</Badge></td><td className="p-2 text-theme-muted">{item.reason}</td>
            </tr>)}</tbody></table></div> : <p className="text-sm text-theme-muted">No mechanism evidence was recorded for this configuration.</p>}
            <p className="text-xs text-theme-muted">Validation: {validation?.status || 'Not supplied'}. Generated YAML alone is not a passed check.</p>
        </div>}
        {view === 'validation' && <section aria-label="Configuration validation" className="space-y-4">
            <Badge tone={validation?.status === 'resources_checked' ? 'success' : 'warning'}>{validation?.status || 'Not checked'}</Badge>
            <div className="grid gap-4 md:grid-cols-2"><div><h3 className="font-semibold">Completed checks</h3>{validation?.checks?.length ? <ul className="mt-3 list-inside list-disc space-y-2 text-sm">{validation.checks.map(check => <li key={check}>{check}</li>)}</ul> : <p className="mt-3 text-sm text-theme-muted">No completed checks recorded.</p>}</div><div><h3 className="font-semibold">Pending verification</h3><ul className="mt-3 list-inside list-disc space-y-2 text-sm">{(validation?.pending || ['Render resources', 'Cluster admission', 'Runtime readiness', 'Workload performance']).map(check => <li key={check}>{check}</li>)}</ul></div></div>
            {validation?.warnings?.map(warning => <p key={warning} className="text-sm">{warning}</p>)}
            <p className="text-xs text-theme-muted">These checks apply to revision {revision}. Editing creates a new revision and requires new resource checks and approval. A resource check does not establish measured SLO compliance.</p>
        </section>}
        {view === 'reason' || view === 'validation' ? null : view === 'yaml' ? <pre tabIndex={0} aria-label="Generated YAML" className="max-h-[36rem] overflow-auto rounded-lg border border-theme-border bg-theme-bg p-5 text-xs leading-relaxed">{content}</pre> : <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_20rem]">
            <div className="min-w-0">{view === '3d' ? <Suspense fallback={<LoadingState label="Loading 3D explorer" />}><Scene plan={plan} selected={selectedId} onSelect={select} /></Suspense> : <DeploymentPlanGraph plan={plan} selected={selectedId} onSelect={select} />}</div>
            <aside className="min-w-0 space-y-4 rounded-lg border border-theme-border bg-theme-bg/50 p-4 text-sm">
                <h3 className="flex items-center gap-2 font-semibold"><Box size={16} />{editing ? 'Edit component' : 'Component inspector'}{editing && <Badge tone="warning" size="xs">Unsaved</Badge>}</h3>
                <label className="block">Component<Select className="mt-1" aria-label="Inspect component" value={selectedId} onChange={event => select(event.target.value)}>{!plan.nodes.length && <option value="">No components</option>}{plan.nodes.map(item => <option key={item.id} value={item.id}>{item.kind}: {item.name}</option>)}</Select></label>
                {node ? <>
                    <div><h3 className="break-all font-semibold">{node.name}</h3><p className="mt-1 text-theme-muted">{node.detail}</p><p className="mt-1 text-xs text-theme-muted">Source: {content ? 'Generated resource' : mode === 'manual' ? 'Manual rule selection' : 'Planner recommendation'}</p></div>
                    {node.namespace && <p>Namespace: {node.namespace}</p>}
                    {node.containers?.map(container => <p key={container.name} className="break-all text-xs">{container.name}: {container.image || 'Image not declared'}</p>)}
                    <details><summary className="cursor-pointer font-medium">Dependencies</summary>{plan.edges.filter(edge => edge.from === node.id || edge.to === node.id).map((edge, index) => <p key={index} className="mt-1 break-words text-xs text-theme-muted">{edge.label}: {plan.nodes.find(item => item.id === (edge.from === node.id ? edge.to : edge.from))?.name}</p>)}</details>
                    {onEdit && node.tier === 1 && node.role !== 'encode' && (!content || node.containers?.some(container => container.name === 'modelserver')) && (!editing
                        ? <Button size="sm" disabled={disabled} onClick={beginEdit}><Pencil size={14} />Edit component</Button>
                        : <fieldset disabled={disabled || applying} className="space-y-3">
                            <legend className="font-semibold">Pending topology edit</legend>
                            <label className="block">Replicas<Input type="number" min="1" max="128" step="1" placeholder={String(previousReplicas || '')} value={replicas} onChange={event => setReplicas(event.target.value)} /></label>
                            <label className="block">Tensor parallel size<Input type="number" min="1" max="128" step="1" placeholder={String(previousTp || '')} value={tp} onChange={event => setTp(event.target.value)} /></label>
                            <p className="text-xs text-theme-muted">Replicas run separate copies. Tensor parallel size is the accelerator count used by each copy.</p>
                            {runtime && <details><summary>Runtime parameters (all serving roles)</summary><div className="mt-3 space-y-3">{runtimeFields.map(([key, label, min, max, step]) => <label key={key} className="block">{label}<Input type="number" min={min} max={max} step={step} placeholder={String(runtime[key] ?? '')} value={runtimeDraft[key] ?? ''} onChange={event => setRuntimeDraft(current => ({ ...current, [key]: event.target.value }))} /></label>)}</div></details>}
                            <p className="text-xs text-theme-muted">{changedResources === null ? 'Resource impact will be checked by the server.' : `Accelerator change: ${changedResources >= 0 ? '+' : ''}${changedResources}. Planned total: ${currentResources + changedResources}.`}</p>
                            {overBudget && <p role="alert">This edit exceeds the {maxAccelerators}-accelerator limit.</p>}
                            <p className="text-xs text-theme-muted">Applying creates a user revision, preserves the original and invalidates previous YAML checks and performance evidence. Regenerate resources before deployment.</p>
                            <div className="flex gap-2"><Button size="sm" onClick={apply} isLoading={applying} disabled={!dirty || overBudget}>Apply revision</Button><Button size="sm" variant="secondary" onClick={() => { setEditing(false); setReplicas(''); setTp(''); setRuntimeDraft({}); setError(''); }}><Undo2 size={14} />Cancel</Button></div>
                        </fieldset>)}
                    {node.document && <details><summary>Component YAML</summary><p className="mt-2 text-xs text-theme-muted">Inspect the complete source in the YAML tab. This component is part of the same configuration revision.</p></details>}
                </> : <p className="text-theme-muted">Select a component to inspect it. Selection never changes the configuration; editing requires an explicit action.</p>}
                {error && <p role="alert">{error}</p>}
            </aside>
        </div>}
        {view !== 'reason' && decisions.length > 0 && <div className="mt-4 flex flex-wrap items-center gap-2 border-t border-theme-border pt-4">
            <span className="mr-1 text-xs text-theme-muted">Optimizations</span>
            {decisions.filter(item => ['enabled', 'dependency'].includes(item.status)).map((item, index) => <Badge key={item.capability_id || item.id || index} tone={tones[item.status]}>{item.label || item.capability_id || item.id}</Badge>)}
            <Button size="xs" variant="link" onClick={() => setView('reason')}>View all decisions ({decisions.length})</Button>
        </div>}
    </Panel>;
}
