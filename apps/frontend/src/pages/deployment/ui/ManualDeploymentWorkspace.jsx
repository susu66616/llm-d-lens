import { DeploymentRefinementDialog } from './DeploymentRequirementsForm';
import { DeploymentWorkflow } from './AgenticDeploymentWorkspace';
import { useEffect, useMemo, useState } from 'react';
import { ArrowLeft, ArrowRight, CheckCircle2, Circle, Download } from 'lucide-react';
import { Badge, Button, Input, Modal, Panel, Select, SummaryStrip, Walkthrough, WorkspaceHeader } from '../../../shared/ui';
import { downloadBlob } from '../../../shared/lib/download';
import { createMockDeploymentPlan, checkMockDeployment } from '../api/agenticDeploymentBackend';
import { createBlueprint } from '../model/inference-designer/blueprint';
import { inferenceScene } from '../model/inference-designer/sceneGraph';
import { InferenceFlowMap2D } from './inference-designer/InferenceFlowMap2D';

export default function ManualDeploymentWorkspace(props) {
    return <DeploymentWorkflow {...props} mode="manual" />;
}

/** Uses the production serving-flow UI, with local fixtures and no deployment transport. */
export function MockDeploymentWorkspace({ draft, mode, onBack, onSuggestAgentic, onRestorePlan }) {
    const objective = draft.initialObjective || 'balanced';
    const [step, setStep] = useState('review');
    const [overrides, setOverrides] = useState(draft.initialMockSettings || {});
    const [selected, setSelected] = useState(null);
    const [scenario, setScenario] = useState('ready');
    const [dialog, setDialog] = useState(null), [form, setForm] = useState({}), [error, setError] = useState('');
    const [progress, setProgress] = useState(0), [tourStep, setTourStep] = useState(null);
    const plan = useMemo(() => createMockDeploymentPlan(draft, mode, objective, overrides), [draft, mode, objective, overrides]);
    const check = checkMockDeployment(plan, draft.preview, scenario);
    const graph = useMemo(() => {
        const result = inferenceScene({ ...createBlueprint(), model: draft.model, replicas: String(plan.settings.replicas), tp: String(plan.settings.tp),
            prefixCaching: plan.settings.prefixCaching ? 'enabled' : 'disabled', chunkedPrefill: plan.settings.chunkedPrefill ? 'enabled' : 'disabled' }, draft.clusterName);
        return { ...result, capabilities: result.capabilities.map(cap => ({ ...cap, inherited: false,
            enabled: plan.enabled.includes(cap.id), required: cap.id === 'parallel', unavailable: !['parallel', 'prefix', 'chunked'].includes(cap.id),
            reason: ['parallel', 'prefix', 'chunked'].includes(cap.id) ? 'Included in the mock workload excerpt.' : 'Outside this mock snapshot; live Guide compatibility has not been checked.',
        })) };
    }, [plan, draft.model, draft.clusterName]);
    const selectedCapability = graph.capabilities.find(cap => cap.id === selected) || graph.nodes.find(node => node.id === selected);
    const blocked = Object.fromEntries(graph.capabilities.filter(cap => cap.unavailable).map(cap => [cap.id, cap.reason]));
    const canConfigure = item => ['parallel', 'prefix', 'chunked', 'serving-worker', 'prefill', 'decode'].includes(item.id);
    const update = values => { setOverrides(current => ({ ...current, ...values })); setDialog(null); setError(''); };
    const edit = () => { setForm(plan.settings); setError(''); setDialog('settings'); };
    const commitSettings = () => {
        try { createMockDeploymentPlan(draft, mode, objective, form); update(Object.fromEntries(Object.entries(form).filter(([key, value]) => value !== plan.settings[key]))); return true; }
        catch (failure) { setError(failure.message); return false; }
    };
    useEffect(() => {
        if (step !== 'execution' || progress >= 3) return undefined;
        const timer = setTimeout(() => setProgress(value => value + 1), 750);
        return () => clearTimeout(timer);
    }, [step, progress]);
    const execution = ['Configuration accepted', 'Model server started', 'Service ready'];
    const numericFields = <>{[['replicas', 'Replicas', 16], ['tp', 'Tensor parallel size', 2], ['maxModelLen', 'Max model length', 40960], ['maxNumSeqs', 'Max sequences', 4096]].map(([key, label, max]) => <label key={key} className="block text-sm">{label}<Input className="mt-1" type="number" min={key === 'maxModelLen' ? 256 : 1} max={max} required value={form[key] ?? ''} onChange={event => setForm(current => ({ ...current, [key]: Number(event.target.value) }))} /></label>)}</>;
    const settingKey = selected === 'prefix' ? 'prefixCaching' : selected === 'chunked' ? 'chunkedPrefill' : null;
    const configurable = Boolean(selectedCapability && canConfigure(selectedCapability) && !selectedCapability.unavailable);
    const changed = configurable && Object.entries(form).some(([key, value]) => value !== plan.settings[key]);
    const flag = settingKey ? `--${form[settingKey] ? 'enable' : 'no-enable'}-${selected === 'prefix' ? 'prefix-caching' : 'chunked-prefill'}` : '';
    const inspector = ({ close }) => selectedCapability ? <div className="space-y-5">
        <Badge tone={configurable ? 'info' : 'neutral'}>{configurable ? 'Editable mock settings' : 'Not available in this mock'}</Badge>
        <p className="text-sm text-theme-muted">{selectedCapability.unavailable ? selectedCapability.reason : selectedCapability.detail}</p>
        {configurable && <form id="mock-optimization-form" className="space-y-5" onSubmit={event => { event.preventDefault(); if (commitSettings()) close(); }}>
            {settingKey ? <>
                <label className="block text-sm font-medium">{selectedCapability.label}<Select className="mt-2" aria-label={`${selectedCapability.label} state`} value={form[settingKey] ? 'enabled' : 'disabled'} onChange={event => { setForm(current => ({ ...current, [settingKey]: event.target.value === 'enabled' })); setError(''); }}>
                    <option value="enabled">Enabled</option><option value="disabled">Disabled</option>
                </Select></label>
                <p className="text-xs text-theme-muted">Current plan: {plan.settings[settingKey] ? 'Enabled' : 'Disabled'}. This mock supports enabling or disabling this optimization.</p>
                <div className="space-y-2"><p className="text-xs font-medium text-theme-muted">Argument preview</p><code aria-label="Argument preview" className="block break-all rounded-lg bg-theme-bg px-3 py-3 text-sm">{flag}</code></div>
            </> : numericFields}
            {error && <p role="alert" className="text-sm">{error}</p>}
            <p role="status" className="text-xs text-theme-muted">{changed ? 'Unapplied changes. Apply to update the plan and YAML.' : 'Change a value, then apply it to the plan.'}</p>
        </form>}
    </div> : null;
    const inspectorFooter = ({ close }) => configurable ? <><Button variant="secondary" onClick={close}>Cancel</Button><Button type="submit" form="mock-optimization-form" disabled={!changed}>Apply changes</Button></> : null;
    return <section className="space-y-5" aria-label="Mock deployment workflow">
        <WorkspaceHeader title={`${mode === 'manual' ? 'Manual' : 'Agentic'} plan`} eyebrow={draft.modelTitle}
            subtitle={`${draft.clusterName} · ${draft.preview.devices} × ${draft.preview.deviceModel}`}
            badge={<Badge tone="warning" title="Browser-only preview; no Kubernetes writes or measured performance.">Mock</Badge>}
            backAction={<Button variant="ghost" size="sm" onClick={onBack}><ArrowLeft size={14} />Change setup</Button>}
            actions={<>{step === 'review' && onRestorePlan && <Button variant="secondary" size="sm" onClick={onRestorePlan}>Return to previous plan</Button>}<Button variant="ghost" size="sm" onClick={() => setTourStep(0)}>1–3 Walkthrough</Button>{step === 'review' && <Button id="mock-review" size="md" onClick={() => { setTourStep(null); setDialog('review'); }}>Review &amp; simulate<ArrowRight size={15} /></Button>}</>} />
        <SummaryStrip id="mock-summary" items={[
            { label: 'Topology', value: `${plan.settings.replicas} replicas · TP ${plan.settings.tp}`, detail: 'Combined Prefill + Decode', onClick: edit, disabled: step !== 'review' },
            { label: 'Resources', value: `${plan.required} / ${check.available} devices`, detail: check.passed ? 'Within mock capacity' : 'Insufficient capacity' },
            { label: 'Optimizations', value: `${plan.enabled.length} enabled`, detail: mode === 'manual' ? 'Compatible mock defaults' : `${objective === 'latency' ? 'Low latency' : objective === 'throughput' ? 'High throughput' : 'Balanced'} objective` },
            { label: 'Validation', value: check.passed ? 'Ready for review' : 'Action required', detail: 'Mock checks only', onClick: () => setDialog('review'), disabled: step !== 'review' },
        ]} />
        {step === 'review' ? <>
            <DeploymentRefinementDialog available={Boolean(onSuggestAgentic)} onOpen={() => setDialog('refine')} isOpen={dialog === 'refine'} onClose={() => setDialog(null)} mock requirements={{ objective }} onSubmit={requirements => onSuggestAgentic?.({ requirements, mockSettings: overrides,
            mockRuntime: Object.fromEntries(Object.entries(overrides).filter(([key]) => ['maxModelLen', 'maxNumSeqs'].includes(key))) })} />
            {!check.passed && <Panel padding="sm" className="shadow-none"><p role="alert" className="text-sm">{check.message}</p></Panel>}
            <Panel id="mock-plan" className="shadow-none">
                <div className="inference-preview-flow"><InferenceFlowMap2D graph={graph} selected={selected} onSelect={setSelected} blocked={blocked} canConfigure={canConfigure} onConfigure={id => { setSelected(id); setForm(plan.settings); setError(''); }} inspector={inspector} inspectorFooter={inspectorFooter} onCloseInspector={() => { setForm({}); setError(''); }} /></div>
            </Panel>
        </> : <Panel title={progress === 3 ? 'Mock deployment complete' : 'Simulating deployment'} actions={<Badge tone={progress === 3 ? 'success' : 'info'}>{progress === 3 ? 'Mock ready' : 'Simulating'}</Badge>}>
            <ol className="my-6 grid gap-4 sm:grid-cols-3" aria-live="polite">{execution.map((label, index) => <li key={label} className="flex items-center gap-3 rounded-lg border border-theme-border p-4">{progress > index ? <CheckCircle2 size={20} /> : <Circle size={20} />}<span className="text-sm">{label}</span></li>)}</ol>
            <Button variant="secondary" onClick={() => { setProgress(0); setStep('review'); }}>Return to plan</Button>
        </Panel>}

        <Modal isOpen={dialog === 'settings'} onClose={() => setDialog(null)} title="Model server configuration" size="sm">
            <form onSubmit={event => { event.preventDefault(); commitSettings(); }} className="space-y-4">{numericFields}{error && <p role="alert" className="text-sm">{error}</p>}<Button type="submit" className="w-full">Apply changes</Button></form>
        </Modal>
        <Modal isOpen={dialog === 'review'} onClose={() => setDialog(null)} title="Review & simulate" size="xl" footer={<>
            <Button variant="secondary" onClick={() => downloadBlob(new Blob([plan.yaml], { type: 'text/yaml' }), `${draft.deploymentName}.mock.yaml`)}><Download size={15} />Download mock YAML</Button>
            <Button id="mock-confirm" disabled={!check.passed} onClick={() => { if (!checkMockDeployment(plan, draft.preview, scenario).passed) return; setProgress(0); setStep('execution'); setDialog(null); }}>Confirm simulation<ArrowRight size={15} /></Button>
        </>}>
            <div className="grid items-start gap-5 md:grid-cols-[minmax(0,1fr)_240px]">
                <section className="min-w-0"><h3 className="mb-3 text-sm font-semibold">Generated configuration</h3><pre className="max-h-[55vh] overflow-auto rounded-lg bg-theme-bg p-4 text-xs">{plan.yaml}</pre></section>
                <section className="space-y-4"><h3 className="text-sm font-semibold">Deployment checks</h3><div className="flex items-center gap-2 text-sm"><CheckCircle2 size={16} />Model cache ready</div><Badge tone={check.passed ? 'success' : 'danger'}>{check.passed ? 'Capacity available' : 'Insufficient capacity'}</Badge><p className="text-sm">{plan.required} required / {check.available} available</p>{!check.passed && <p role="alert" className="text-sm">{check.message}</p>}
                    <details className="border-t border-theme-border pt-3"><summary className="cursor-pointer text-xs text-theme-muted">Mock scenario</summary><Select className="mt-3" aria-label="Mock scenario" value={scenario} onChange={event => setScenario(event.target.value)}><option value="ready">Ready</option><option value="capacity">Insufficient capacity</option></Select>{scenario === 'capacity' && <Button className="mt-3" variant="secondary" size="sm" onClick={() => setScenario('ready')}>Restore mock capacity</Button>}</details>
                    <p className="text-xs text-theme-muted">Local simulation only. This YAML is a workload excerpt, not a complete llm-d release.</p>
                </section>
            </div>
        </Modal>
        <Walkthrough step={tourStep} onStepChange={setTourStep} onClose={() => setTourStep(null)} steps={step === 'review' ? [
            { target: '#mock-summary', title: 'Read the plan at a glance', description: 'Topology, device use and checks are summarized once. Click the topology to change replicas and engine limits.' },
            { target: '#deployment-serving-stages', title: 'Inspect the request path', description: 'Expand a group to see its optimizations. Select a setting to open its configuration directly from the graph, then return to the flow.' },
            { target: '#mock-review', title: 'Review and confirm', description: 'Review YAML and capacity together, try the failure scenario, then confirm the simulation.' },
        ] : []} />
    </section>;
}
