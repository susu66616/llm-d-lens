import { lazy, Suspense, useEffect, useEffectEvent, useRef, useState } from 'react';
import { ArrowLeft, ArrowRight, CheckCircle2, Download, ChevronDown, Circle } from 'lucide-react';
import { Badge, Button, EmptyState, LoadingState, Modal, ModulePage, Panel, Select, Spinner, SummaryStrip, Walkthrough, WorkspaceHeader } from '../../../shared/ui';
import { downloadBlob } from '../../../shared/lib/download';
import { FormError } from '../../../shared/ui/FormError';
import { requestJson } from '../../../shared/api/httpClient';
import { usePolling } from '../../../shared/lib/usePolling';
import { historicalBenchmarkLabel, publicRagMatchSummary, retrievalBranchSummaries } from '../lib/agenticPresentation';
import { useAuth } from '../../auth';
import { getLocalDeploymentRun } from '../../optimization';
import { DeploymentPodsLogsModal } from './DeploymentManagement/DeploymentPodsLogsModal';
import { formatTimestamp } from '../../../shared/lib/formatTimestamp';
import { listAIProviders } from '../../ai-providers';
import { DeploymentRequirementsForm, DeploymentRefinementDialog, PlanningProviderSelect } from './DeploymentRequirementsForm';
import { DeploymentPlanReview } from './DeploymentPlanReview';
import { MeasuredOptimizationControls } from './MeasuredOptimizationControls';
import { candidateEvidence, candidateLabel, defaultRequirements, deploymentPlanningRequest, requirementsFromRun, streamAgenticRecommendation } from '../api/agenticDeploymentBackend';
import { deploymentEditorSnapshot, renderDeploymentPlan } from '../model/inference-designer/deploymentPlanning';

const InferenceDesigner = lazy(() => import('./inference-designer/InferenceDesigner').then(module => ({ default: module.InferenceDesigner })));

const api = (path, body) => requestJson(path, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
});

export function RetrievalEvidenceSources({ run }) {
    const usedIds = new Set(run.decision_evidence_ids || []);
    const usage = (ids) => ids.some((id) => usedIds.has(id)) ? 'used for selection' : 'context only';
    const caseReferences = (item) => [
        item.case_id,
        ...(item.measurements || []).map((measurement) => measurement.evidence_id),
        ...(item.provenance || []).map((citation) => `benchmark:${citation.benchmark_id}`),
    ];
    return retrievalBranchSummaries(run).map((branch) => {
        const details = branch.evidence?.details || {};
        const observations = details.response?.evidence || [];
        const matches = new Map((details.matches || []).map((match) => [match.evidence_id, match]));
        const sourceUsage = usage([
            branch.evidence?.id,
            ...observations.map((observation) => `public-rag:${observation.item_id}`),
            ...(branch.cases || []).flatMap(caseReferences),
        ]);
        return <details key={branch.id} className="group p-3 text-theme-text">
            <summary className="cursor-pointer list-none">
                <span className="flex items-center justify-between gap-2">
                    <span className="flex items-center gap-2 text-xs"><ChevronDown size={14} className="-rotate-90 group-open:rotate-0" />{branch.label}</span>
                    <Badge tone={branch.status === 'available' ? 'info' : 'neutral'} size="xs">{branch.status.replaceAll('_', ' ')}</Badge>
                </span>
                <span className="mt-1 block text-[11px] text-theme-muted">{branch.summary}</span>
            </summary>
            <div className="mt-3 space-y-2 text-[11px] text-theme-muted">
                <Badge tone={sourceUsage === 'used for selection' ? 'success' : 'neutral'} size="xs">{sourceUsage}</Badge>
                {branch.warning && <p>{branch.warning}</p>}
                <p>Historical measurements apply to their reported conditions, not current deployment SLO compliance. Compare model, hardware, runtime and workload before using them alongside AIC estimates.</p>
                {details.message && <p>{details.message}</p>}
                {details.retrieval?.mode && <p>Retrieval: {details.retrieval.mode.replaceAll('_', ' ')}.</p>}
                {details.retrieval?.model_identity_aliases && <p>Query model name aliases: {details.retrieval.model_identity_aliases.join(', ')}. Stored names use the same rule; source labels and citations are unchanged.</p>}
                {branch.id === 'public_sql' && <ul className="space-y-3">
                    {observations.map((observation) => <li key={observation.item_id}>
                        <p>{observation.metric} {observation.statistic}: {observation.value} {observation.unit}</p>
                        <Badge tone={usedIds.has(`public-rag:${observation.item_id}`) ? 'success' : 'neutral'} size="xs">{usage([`public-rag:${observation.item_id}`])}</Badge>
                        {publicRagMatchSummary(matches.get(`public-rag:${observation.item_id}`)).map((line) => <p key={line}>{line}</p>)}
                        <p className="break-all">{observation.citation?.source_path} @ {observation.citation?.source_revision}</p>
                        <p className="break-all">SHA-256: {observation.citation?.sha256}</p>
                    </li>)}
                </ul>}
                {branch.id === 'internal_sql' && <ul className="space-y-3">
                    {branch.cases.map((item) => <li key={item.case_id}>
                        <p className="font-semibold text-theme-text">{item.model} / {item.hardware} / {item.experiment_id}</p>
                        <Badge tone={usage(caseReferences(item)) === 'used for selection' ? 'success' : 'neutral'} size="xs">{usage(caseReferences(item))}</Badge>
                        <p>Applicability: {item.applicability}. Historical latency target status: {item.latency_status}.</p>
                        <p>Configuration: {JSON.stringify(item.configuration)}</p>
                        <p>Workload: {JSON.stringify(item.workload)}</p>
                        {item.known_differences?.length > 0 && <p>Known differences: {item.known_differences.join(', ')}.</p>}
                        {item.unknown_fields?.length > 0 && <p>Unknown fields: {item.unknown_fields.join(', ')}.</p>}
                        {item.ranking?.queue === 'fallback' && <p>Fallback reference only, not low-latency evidence.</p>}
                        <ul>{item.measurements?.map((measurement) => <li key={measurement.evidence_id}>{measurement.metric} {measurement.statistic}: {measurement.value} {measurement.unit} ({measurement.quality})</li>)}</ul>
                        {item.provenance?.map((citation, index) => <p key={index} className="break-all">Benchmark: {citation.benchmark_id}; {citation.source_path} @ {citation.source_revision}</p>)}
                        {item.validation_required?.length > 0 && <p>Validation required: {item.validation_required.join(', ')}.</p>}
                    </li>)}
                </ul>}
                {details.response?.truncated && <p>More observations matched; only the bounded retrieved subset was supplied to planning.</p>}
                {!observations.length && !branch.cases?.length && <p>No retained evidence details for this source.</p>}
            </div>
        </details>;
    });
}

export function DeploymentSummary({ run, draft, deploymentRun, busy, locked, editing, cleanupRequired, onDeploy, onCheck, onRequirements, onOpen, endpoint, onEndpointChange, confirmLabel = 'Deploy' }) {
    const plan = run.configuration_plan;
    const checked = Boolean(plan?.configuration?.content?.officialGuide?.renderedManifest) && plan?.validation?.status === 'resources_checked';
    const submitted = Boolean(run.deployment_run_id);
    const ready = deploymentRun?.status === 'succeeded' && Boolean(deploymentRun.cases?.length) && deploymentRun.cases.every(item => item.status === 'ready');
    const needsAttention = ['failed', 'partially_succeeded', 'cancelled', 'cleaned'].includes(deploymentRun?.status);
    const title = submitted ? ready ? 'Deployment ready' : needsAttention ? 'Review deployment status' : 'Deployment submitted'
        : cleanupRequired ? 'Clean up test resources' : busy ? 'Updating your plan' : editing ? 'Finish editing to continue' : checked ? 'Ready for your confirmation' : 'Resource checks required';
    const candidate = run.selected_candidate;
    const resources = plan?.resource_estimate || {};
    return <Panel id="deployment-summary" title="Deployment summary" className="min-w-0 shadow-none" aria-label="Deployment summary">
        <p role="status" className="text-lg font-semibold text-theme-text">{title}</p>
        <p className="mt-2 text-sm leading-relaxed text-theme-muted">{submitted
            ? ready ? 'The serving components report ready. Open the deployment for endpoints, monitoring and evaluation.'
                : 'Follow the reported component states and inspect logs below. Submission alone does not mean the model is ready.'
            : 'Confirm the resource footprint and checks. Capacity is checked again when you deploy.'}</p>
        <dl className="my-5 divide-y divide-theme-border text-sm">
            {[
                ['Cluster', draft.clusterName || draft.clusterId],
                ['Accelerators required', resources.accelerators ?? (candidate ? candidate.replicas * candidate.tensor_parallel_size + (candidate.prefill_replicas || 0) * (candidate.prefill_tensor_parallel_size || 0) : 'Unknown')],
                ['Available at planning', resources.available_accelerators ?? 'Unknown'],
                ['Serving replicas / TP', candidate ? `${candidate.replicas} / ${candidate.tensor_parallel_size}` : 'Unknown'],
                ...(candidate?.prefill_replicas ? [['Prefill replicas / TP', `${candidate.prefill_replicas} / ${candidate.prefill_tensor_parallel_size}`]] : []),
                ...(candidate?.native_cpu_cache_gib ? [['CPU cache / replica', `${candidate.native_cpu_cache_gib} GiB`]] : []),
                ['Model cache', draft.storageVolumeName || draft.storageVolumeId],
            ].map(([label, value]) => <div key={label} className="flex flex-wrap justify-between gap-x-4 gap-y-1 py-3"><dt className="text-theme-muted">{label}</dt><dd className="min-w-0 break-all font-medium">{value}</dd></div>)}
        </dl>
        <ul className="space-y-3 text-sm" aria-label="Deployment readiness">
            <li className="flex gap-2">{checked ? <CheckCircle2 size={17} className="shrink-0" aria-hidden="true" /> : <Circle size={17} className="shrink-0" aria-hidden="true" />}<span>{checked ? 'Configuration and resource checks passed' : 'Configuration and resource checks pending'}</span></li>
            <li className="flex gap-2">{ready ? <CheckCircle2 size={17} className="shrink-0" aria-hidden="true" /> : <Circle size={17} className="shrink-0" aria-hidden="true" />}<span>{ready ? 'Serving components ready' : submitted ? 'See execution for startup status' : 'Cluster admission and startup: after confirmation'}</span></li>
            <li className="flex gap-2"><Circle size={17} className="shrink-0" aria-hidden="true" /><span>Readiness does not verify workload performance.</span></li>
        </ul>
        {plan?.validation?.warnings?.length > 0 && <details className="mt-4 text-sm"><summary className="cursor-pointer">Review {plan.validation.warnings.length} configuration warnings</summary><ul className="mt-2 list-disc space-y-2 pl-4 text-theme-muted">{plan.validation.warnings.map((warning, index) => <li key={index}>{warning}</li>)}</ul></details>}
        {run.error && <p role="alert" className="mt-4 text-sm">{run.error}</p>}
        {cleanupRequired && <p role="alert" className="mt-4 text-sm">Clean up interrupted benchmark resources before revising or deploying.</p>}
        <div className="mt-5 space-y-3 border-t border-theme-border pt-5">
            {!submitted && plan?.source_bundle && <label className="block text-sm">Inference Service
                <Select className="mt-1" aria-label="Inference Service" value={endpoint || ''} disabled={locked || editing} onChange={event => onEndpointChange?.(event.target.value)}>
                    <option value="">Choose the inference Service</option>{plan.endpoints?.map(item => <option key={item.value} value={item.value}>{item.value}</option>)}
                </Select><span className="mt-1 block text-xs text-theme-muted">Choose the model-serving endpoint, not EPP or metrics.</span>
            </label>}
            {submitted ? onOpen && <Button size="md" className="w-full" onClick={onOpen}>Open deployment</Button> : <>
                <Button size="md" className="w-full" disabled={locked || editing || !plan || (checked && Boolean(plan.source_bundle) && !plan.endpoints?.some(item => item.value === endpoint))} isLoading={busy} onClick={checked ? onDeploy : onCheck}>{checked ? confirmLabel : 'Retry checks'}</Button>
                <Button variant="ghost" className="w-full" disabled={locked || editing} onClick={onRequirements}>Workload and limits</Button>
                <p className="text-xs leading-relaxed text-theme-muted">You will confirm this exact configuration before any deployment starts.</p>
            </>}
        </div>
    </Panel>;
}

export function DeploymentWorkflow({ draft, mode = 'agentic', embedded = false, onBack, onDeploymentStarted, onSuggestAgentic, onRestorePlan }) {
    const { can } = useAuth();
    const permitted = can('deployment:run:create');
    const [run, setRun] = useState(null), [requirements, setRequirements] = useState(() => ({ ...defaultRequirements(), objective: draft?.initialObjective || 'balanced', ...draft?.initialRequirements }));
    const [busy, setBusy] = useState(false), [error, setError] = useState(''), [events, setEvents] = useState([]);
    const [providers, setProviders] = useState([]), [provider, setProvider] = useState(draft?.initialProvider || '');
    const [refining, setRefining] = useState(false);
    const [restoring, setRestoring] = useState(true), [approval, setApproval] = useState(null);
    const [editing, setEditing] = useState(false);
    const [endpoint, setEndpoint] = useState('');
    const [sourceProfiles, setSourceProfiles] = useState([]), [profileId, setProfileId] = useState('');
    const [workspaceView, setWorkspaceView] = useState('review');
    const [changingRequirements, setChangingRequirements] = useState(false);
    const [trialBusy, setTrialBusy] = useState(false);
    const [tourStep, setTourStep] = useState(null);
    const [deploymentRun, setDeploymentRun] = useState(null), [logsDeployment, setLogsDeployment] = useState(null);
    const requestVersion = useRef(0);
    const abort = useRef(null), mounted = useRef(true);
    const autoStarted = useRef(null);
    const key = `lens-deployment-plan:${mode}:${draft?.clusterId}:${draft?.model}:${draft?.deploymentName}:${draft?.storageVolumeId}${draft?.initialObjective ? `:${draft.initialObjective}` : ''}${draft?.planningAttemptId ? `:${draft.planningAttemptId}` : ''}`;
    useEffect(() => {
        mounted.current = true;
        return () => { mounted.current = false; abort.current?.abort(); };
    }, []);
    useEffect(() => {
        let active = true;
        requestVersion.current += 1; abort.current?.abort();
        setRun(null); setDeploymentRun(null); setLogsDeployment(null); setApproval(null); setRestoring(true); setBusy(false); setError(''); setEvents([]); setChangingRequirements(false);
        setRequirements({ ...defaultRequirements(), objective: draft?.initialObjective || 'balanced', ...draft?.initialRequirements }); setProvider(draft?.initialProvider || '');
        const id = draft?.resumeRunId || sessionStorage.getItem(key);
        if (!id) { setRestoring(false); return undefined; }
        api(`/api/agentic-deployments/${encodeURIComponent(id)}`).then(value => {
            if (active) {
                setRequirements(requirementsFromRun(value));
                setProvider(value.request.ai_provider_id || '');
                setRun(value);
                setWorkspaceView(value.deployment_run_id ? 'execution' : value.status === 'optimizing' ? 'trials' : 'review');
            }
        }).catch(failure => { if (active) setError(`Unable to restore the saved plan: ${failure.message}`); })
            .finally(() => { if (active) setRestoring(false); });
        return () => { active = false; };
    }, [key, draft?.contextLength, draft?.initialObjective, draft?.initialRequirements, draft?.initialProvider, draft?.resumeRunId, mode]);
    useEffect(() => {
        if (mode !== 'agentic' && !refining) return undefined;
        let active = true;
        listAIProviders().then(items => { if (active) setProviders(items); })
            .catch(failure => { if (active) setError(`Unable to load AI providers: ${failure.message}`); });
        return () => { active = false; };
    }, [mode, refining]);
    const accept = value => {
        if (!mounted.current) return;
        setRun(value); setApproval(null); setEndpoint(''); sessionStorage.setItem(key, value.id);
    };
    const active = run?.status === 'optimizing' || run?.status === 'deploying';
    usePolling(async () => {
        if (!run) return;
        const version = requestVersion.current;
        try {
            const value = await api(`/api/agentic-deployments/${encodeURIComponent(run.id)}`);
            if (mounted.current && version === requestVersion.current) setRun(value);
            if (value.deployment_run_id) {
                const execution = await getLocalDeploymentRun(value.deployment_run_id);
                if (mounted.current && version === requestVersion.current) setDeploymentRun(execution);
            }
        } catch (failure) { if (mounted.current && version === requestVersion.current) setError(failure.message); }
    }, { enabled: Boolean(run) && !busy && !trialBusy, intervalMs: active ? 5000 : 15000, immediate: true });
    const generate = async value => {
        const version = ++requestVersion.current;
        setTourStep(null);
        setBusy(true); setError(''); setApproval(null); setEvents([]);
        const controller = new AbortController();
        abort.current = controller;
        try {
            const session = await api(`/api/cluster/session?clusterId=${encodeURIComponent(draft.clusterId)}`);
            const payload = deploymentPlanningRequest(run ? {
                ...draft, contextLength: run.request.max_model_len, gpuMemoryUtilization: run.request.gpu_memory_utilization,
                maxNumSeqs: run.request.max_num_seqs, maxNumBatchedTokens: run.request.max_num_batched_tokens,
            } : draft, value, session.sessionId || session.session_id, mode, provider);
            let result;
            await streamAgenticRecommendation(payload, (type, data) => {
                if (type === 'progress' && mounted.current && version === requestVersion.current) setEvents(current => [...current.slice(-29), data]);
                if (type === 'complete') result = data;
                if (type === 'error') throw new Error(data.message || 'Planning failed.');
            }, controller.signal);
            if (!result) throw new Error('Planning ended without a validated configuration.');
            if (!mounted.current || version !== requestVersion.current) return;
            accept(result); setRequirements(requirementsFromRun(result)); setChangingRequirements(false); setWorkspaceView('review');
            {
                setEvents(current => [...current, { phase: 'resources', status: 'running', message: 'Checking the optimization combination and current cluster resources.' }]);
                try {
                    const rendered = await renderDeploymentPlan(result, { compare: true, profileId });
                    if (!mounted.current || version !== requestVersion.current) return;
                    accept(rendered);
                    if (mounted.current) setEvents(current => [...current, { phase: 'resources', status: 'complete', message: 'Configuration checked. Ready for review.' }]);
                } catch (failure) {
                    if (failure.profiles && mounted.current) setSourceProfiles(failure.profiles);
                    if (mounted.current && version === requestVersion.current) setError(`The recommendation is preserved, but source resource checks failed: ${failure.message}. Resolve the source requirements and retry; deployment remains blocked.`);
                }
            }
            return true;
        } catch (failure) {
            if (mounted.current && version === requestVersion.current) setError(controller.signal.aborted ? 'Planning cancelled. No deployment was started.' : failure.message);
            return false;
        } finally { if (mounted.current && version === requestVersion.current) { setBusy(false); abort.current = null; } }
    };
    const autoGenerate = mode === 'manual' || Boolean(draft?.initialObjective);
    const startPlanning = useEffectEvent(() => generate({ ...defaultRequirements(), objective: draft?.initialObjective || 'balanced', ...draft?.initialRequirements }));
    useEffect(() => {
        if (autoGenerate && !restoring && !run && !error && permitted && autoStarted.current !== key) {
            autoStarted.current = key;
            startPlanning();
        }
    }, [autoGenerate, restoring, run, error, permitted, key]);
    const mutate = async (action, body, checkResources = false) => {
        const version = ++requestVersion.current;
        setBusy(true); setError('');
        try {
            const value = await api(`/api/agentic-deployments/${encodeURIComponent(run.id)}/${action}`, body);
            if (!mounted.current || version !== requestVersion.current) return null;
            accept(value);
            if (checkResources) {
                try {
                    const checkedRun = value.configuration_plan?.source_bundle ? value : await renderDeploymentPlan(value, { profileId });
                    if (!mounted.current || version !== requestVersion.current) return null;
                    accept(checkedRun);
                    return checkedRun;
                } catch (failure) {
                    if (mounted.current && version === requestVersion.current) setError(`The change is saved, but resource checks failed: ${failure.message}. Deployment is blocked until checks pass.`);
                }
            }
            return value;
        } catch (failure) {
            if (mounted.current && version === requestVersion.current) setError(failure.message);
            throw failure;
        } finally { if (mounted.current && version === requestVersion.current) setBusy(false); }
    };
    const choose = async id => { try { if (await mutate('select', { candidate_id: id }, true)) setWorkspaceView('review'); } catch { /* mutate presents the error. */ } };
    const renderSource = async ({ draft: editedDraft, compare = false } = {}) => {
        const version = ++requestVersion.current;
        setBusy(true); setError(''); setApproval(null);
        try {
            const value = await renderDeploymentPlan(run, { editedDraft, compare, profileId });
            if (mounted.current && version === requestVersion.current) { accept(value); setEditing(false); setTourStep(current => current === 3 ? 4 : current); }
            return value;
        } catch (failure) {
            if (mounted.current && version === requestVersion.current) { setError(failure.message); if (failure.profiles) setSourceProfiles(failure.profiles); }
            throw failure;
        } finally { if (mounted.current && version === requestVersion.current) setBusy(false); }
    };
    const render = async () => { try { await renderSource(); } catch { /* The source error remains visible; no legacy fallback is deployed. */ } };
    const edit = ({ node, patch }) => mutate('select', {
        candidate_id: run.selected_candidate.id, role: node.role === 'prefill' ? 'prefill' : 'decode',
        replicas: patch.replicas, tensor_parallel_size: patch.tensorParallelSize,
        max_model_len: patch.max_model_len, gpu_memory_utilization: patch.gpu_memory_utilization,
        max_num_seqs: patch.max_num_seqs, max_num_batched_tokens: patch.max_num_batched_tokens,
        native_cpu_cache_gib: patch.native_cpu_cache_gib,
    }, true);
    const deploy = async () => {
        if (!approval || approval.runId !== run.id || approval.checksum !== run.configuration_plan?.configuration?.checksum) {
            setApproval(null); setError('The configuration changed. Review the current plan before deploying.'); return;
        }
        try { const result = await mutate('approve', { configuration_checksum: approval.checksum, ...(plan.source_bundle ? { endpoint: approval.endpoint } : {}) }); if (result?.deployment_run_id) setWorkspaceView('execution'); }
        catch { /* mutate presents the error. */ }
    };
    const reset = () => { setTourStep(null); setChangingRequirements(true); setApproval(null); setError(''); setEvents([]); };
    const plan = run?.configuration_plan;
    const manifest = plan?.configuration?.content?.officialGuide?.renderedManifest || '';
    const checked = Boolean(manifest) && plan?.validation?.status === 'resources_checked';
    const cleanupRequired = ['interrupted', 'cleanup_failed'].includes(run?.optimization?.status);
    const locked = busy || trialBusy || cleanupRequired || active || !permitted || (run && run.status !== 'awaiting_approval');
    const enabled = plan?.decisions?.filter(item => ['enabled', 'dependency'].includes(item.status)) || [];
    const tourSteps = !run || changingRequirements ? [
        { target: '#deployment-requirements', title: 'Adjust planning inputs', description: 'Change the objective or optional workload limits, then rebuild the plan.' },
    ] : [
        { target: '#deployment-plan-summary', title: 'Read the plan at a glance', description: 'The topology, resource use and validation state are summarized once.' },
        { target: '#deployment-serving-stages', title: 'Inspect the serving path', description: 'Expand a group and select an optimization to edit it directly. Apply changes and check resources before reviewing.' },
        { target: '#deployment-review', title: 'Review and confirm', description: 'Review YAML, resource checks and the inference endpoint together. Confirming submits this exact configuration.' },
    ];
    const content = <section className="space-y-5">
        <WorkspaceHeader title={`${mode === 'manual' ? 'Manual' : 'Agentic'} plan`}
            eyebrow={draft.modelTitle || draft.model?.split('/').at(-1) || 'New deployment'} subtitle={draft.clusterName || draft.clusterId}
            backAction={onBack && <Button variant="ghost" size="sm" disabled={busy || trialBusy || active || editing} onClick={onBack}><ArrowLeft size={14} />Change setup</Button>}
            actions={<>{onRestorePlan && !run?.deployment_run_id && <Button variant="secondary" size="sm" disabled={busy || trialBusy || active || editing || cleanupRequired} onClick={onRestorePlan}>Return to previous plan</Button>}{(!run || plan?.source_bundle) && !run?.deployment_run_id && <Button variant="ghost" size="sm" disabled={restoring || busy || trialBusy || editing} onClick={() => { setWorkspaceView('review'); setTourStep(0); }}>1–3 Walkthrough</Button>}
                {run?.deployment_run_id ? onDeploymentStarted && <Button onClick={() => onDeploymentStarted(run)}>Open deployment</Button> : run && <Button id="deployment-review" size="md" disabled={locked || editing || !plan} onClick={() => { setTourStep(null); setApproval({ runId: run.id, checksum: plan.configuration?.checksum, endpoint }); }}>Review &amp; deploy<ArrowRight size={15} /></Button>}</>} />
        {run && <SummaryStrip id="deployment-plan-summary" items={[
            { label: 'Topology', value: run.selected_candidate?.prefill_replicas ? 'Prefill / Decode' : 'Combined serving', detail: `${run.selected_candidate?.replicas || 1} serving replicas · TP ${run.selected_candidate?.tensor_parallel_size || 1}` },
            { label: 'Resources', value: `${plan?.resource_estimate?.accelerators ?? '?'} / ${plan?.resource_estimate?.available_accelerators ?? '?'} accelerators` },
            { label: 'Optimizations', value: `${enabled.length} configured`, detail: mode === 'manual' ? 'Compatible source settings' : requirements.objective },
            { label: 'Validation', value: editing ? 'Unapplied changes' : checked ? 'Ready for review' : 'Checks pending', detail: 'Performance not benchmarked' },
        ]} />}
        <DeploymentRefinementDialog available={Boolean(run && !run.deployment_run_id && !changingRequirements && onSuggestAgentic)} disabled={locked} editing={editing} onOpen={() => { setTourStep(null); setRefining(true); }} isOpen={refining} onClose={() => setRefining(false)} requirements={requirements} runtime={run?.request} resourceEstimate={plan?.resource_estimate}
            provider={provider} providers={providers} onProviderChange={setProvider} onSubmit={values => {
                if (locked || editing || !run) return;
                onSuggestAgentic?.({ requirements: values, provider, runId: run.id, runtime: {
                    contextLength: run.request.max_model_len, gpuMemoryUtilization: run.request.gpu_memory_utilization,
                    maxNumSeqs: run.request.max_num_seqs, maxNumBatchedTokens: run.request.max_num_batched_tokens,
                } });
            }} />
        {error && <div className="space-y-2"><p className="text-sm font-semibold text-theme-text">{run?.deployment_run_id ? 'Deployment update needs attention' : run ? 'Your saved plan is preserved' : 'Unable to prepare the deployment plan'}</p><FormError message={error} /><p className="text-sm text-theme-muted">{run?.deployment_run_id ? 'The deployment may still be running. Check its execution status before taking further action.' : run ? 'Review the message, retry resource checks or adjust the workload and limits. No deployment starts without confirmation.' : 'Check the target and inputs, then build the plan again.'}</p></div>}
        {!permitted && <EmptyState title="Deployment permission required" message="Ask an administrator for deployment-create access to this cluster." />}
        {restoring ? <LoadingState label="Restoring deployment plan" /> : ((!run && ((!autoGenerate && mode === 'agentic') || (!busy && error))) || changingRequirements) && <DeploymentRequirementsForm mode={mode} initialValue={requirements} runtime={run?.request} resourceEstimate={plan?.resource_estimate} onSubmit={generate} disabled={busy || !permitted} aiProviderConfigured={Boolean(provider)}>
            <PlanningProviderSelect provider={provider} providers={providers} onChange={setProvider} />
        </DeploymentRequirementsForm>}
        {changingRequirements && <Button variant="secondary" disabled={busy} onClick={() => { setRequirements(requirementsFromRun(run)); setProvider(run.request.ai_provider_id || ''); setChangingRequirements(false); }}>Keep existing plan</Button>}
        {events.length > 0 && <div className="text-sm">
            {busy && <div className="flex flex-wrap items-center gap-3 rounded-lg border border-theme-border p-4">
                {busy && <Spinner size="sm" />}
                <div className="min-w-0 flex-1" role="status"><p className="font-medium text-theme-text">{busy ? 'Building your deployment plan' : error ? 'Planning stopped' : 'Planning complete'}</p><p className="mt-1 text-theme-muted">{events.at(-1)?.message}</p></div>
                {busy && !run && <Button size="sm" variant="secondary" onClick={() => abort.current?.abort()}>Cancel</Button>}
            </div>}
            <details className="mt-3"><summary className="cursor-pointer text-theme-muted">Planning log ({events.length})</summary>
                <ol className="mt-3 max-h-48 space-y-2 overflow-auto border-l border-theme-border pl-4 text-theme-muted">
                    {events.map((event, index) => <li key={index}><span className="font-medium text-theme-text">{event.phase.replaceAll('_', ' ')}</span><span className="ml-3">{event.message}</span></li>)}
                </ol>
            </details>
        </div>}
        {run && !changingRequirements && <>
            {run.request.requirements?.workload?.interpretationSource === 'ai_goal' && <p className="text-sm text-theme-muted">AI workload interpretation: shared prompts {run.request.requirements.workload.prefixReuse == null ? 'unknown' : run.request.requirements.workload.prefixReuse ? 'yes' : 'no'}; long inputs with short answers {run.request.requirements.workload.prefillHeavy == null ? 'unknown' : run.request.requirements.workload.prefillHeavy ? 'yes' : 'no'}. Correct these in Deployment settings.</p>}
            {cleanupRequired && <Panel title="Test resources require cleanup"><p role="alert" className="text-sm">Resolve the interrupted or failed cleanup in Real benchmark tests before revising or deploying this plan.</p><Button className="mt-3" variant="secondary" disabled={busy || trialBusy || editing} onClick={() => setWorkspaceView('trials')}>Open test cleanup</Button></Panel>}
            {(mode === 'agentic' || run.deployment_run_id) && <details open={workspaceView !== 'review' || Boolean(run.deployment_run_id)}><summary className="cursor-pointer text-sm text-theme-muted">{run.deployment_run_id ? 'Deployment views' : 'More planning tools / optional'}</summary><nav className="mt-3 flex flex-wrap gap-2" aria-label="Planning workspace">
                {[["review", run.deployment_run_id ? 'Submitted plan' : 'Plan'], ...(mode === 'agentic' && !run.deployment_run_id && run.candidates.length > 1 ? [["compare", 'Alternatives']] : []), ...(mode === 'agentic' && (!run.deployment_run_id || run.optimization) ? [["trials", 'Benchmark tests']] : []), ...(run.deployment_run_id ? [["execution", 'Deployment execution']] : [])].map(([id, label]) => <Button key={id} variant={workspaceView === id ? 'secondary' : 'ghost'} aria-pressed={workspaceView === id} disabled={editing || busy || trialBusy} onClick={() => setWorkspaceView(id)}>{label}</Button>)}
            </nav></details>}
            <div className="grid min-w-0 items-start gap-5">
            <div className="min-w-0 space-y-4">
            {workspaceView === 'review' && (plan?.source_bundle && plan.source_draft
                ? <Suspense fallback={<LoadingState label="Loading deployment designer" />}><InferenceDesigner key={`${run.id}:${run.revision}:${plan.configuration?.checksum}`} initialDraft={deploymentEditorSnapshot(run).draft} initialResult={deploymentEditorSnapshot(run).result} clusterName={draft.clusterName || draft.clusterId} disabled={locked} onReview={renderSource} onEditingChange={setEditing} /></Suspense>
                : <>
                    <Panel title="Source-backed Guide editor">
                        <p className="text-sm text-theme-muted">Load this cluster's pinned source catalog to open the 2D optimization flow and edit its parameters. A legacy plan is not evidence that every Guide can be composed.</p>
                        {sourceProfiles.length > 0 && <label className="mt-3 block text-sm">Accelerator profile<Select value={profileId} onChange={event => setProfileId(event.target.value)}><option value="">Choose discovered hardware</option>{sourceProfiles.map(item => <option key={item.id} value={item.id}>{item.name || item.id}</option>)}</Select></label>}
                        <Button className="mt-3" disabled={locked || editing} onClick={render}>Load Guides and build source configuration</Button>
                    </Panel>
                    <details><summary className="cursor-pointer text-sm text-theme-muted">Legacy plan preview</summary><DeploymentPlanReview content={manifest} candidate={run.selected_candidate} model={draft.model} mode={mode} decisions={plan?.decisions || []} onEdit={edit} onEditingChange={setEditing} onRequirementsChange={value => generate({ ...requirements, ...value })} onCheck={render} onBenchmark={mode === 'agentic' ? () => setWorkspaceView('trials') : undefined} onWorkload={reset} requirements={requirements} disabled={locked} validation={plan?.validation} revision={run.revision} runtime={run.request} maxAccelerators={plan?.requirements?.hard_constraints?.maxAccelerators} resourceEstimate={plan?.resource_estimate} evidence={candidateEvidence(run)} rationale={run.decision?.rationale} /></details>
                </>)}
            {(run.generator_fallback_reason || run.planner_fallback_reason) && <details className="text-sm text-theme-muted"><summary className="cursor-pointer">Rules were used for part of this recommendation</summary><p role="status" className="mt-2">An AI step was unavailable. Resource checks and confirmation are still required; this is not a performance guarantee.</p><p className="mt-2">{run.generator_fallback_reason || run.planner_fallback_reason}</p></details>}
            {workspaceView === 'compare' && <section aria-label="Candidate comparison" className="space-y-4">
                <div><h3 className="text-lg font-semibold">Compare deployment plans</h3><p className="mt-1 text-sm text-theme-muted">Compare resource needs and evidence, not just the number of optimizations. Selecting a plan automatically reruns its resource checks.</p></div>
                <fieldset disabled={locked || editing} className="grid min-w-0 gap-4 lg:grid-cols-3">
                    {run.candidates.map((item, index) => <Panel key={item.id} title={`Plan ${index + 1}`} className="flex min-w-0 flex-col shadow-none" actions={run.selected_candidate.id === item.id && <Badge tone="info">Selected</Badge>}>
                        <h4 className="text-base font-semibold">{plan?.source_candidates?.[item.id] ? 'Source-composed configuration' : candidateLabel(item)}</h4>
                        <p className="my-4 text-2xl font-semibold">{item.required_gpus} <span className="text-sm font-normal text-theme-muted">accelerators</span></p>
                        <p className="text-sm text-theme-muted">{item.replicas} serving replicas / TP {item.tensor_parallel_size}{item.prefill_replicas ? `; prefill ${item.prefill_replicas} / TP ${item.prefill_tensor_parallel_size}` : ''}</p>
                        {item.native_cpu_cache_gib && <p className="mt-2 text-sm text-theme-muted">CPU KV cache: {item.native_cpu_cache_gib} GiB per replica</p>}
                        <p className="my-4 text-sm">{item.slo_status === 'satisfied' ? 'Targets supported by prediction or history; not verified for this deployment.' : item.slo_status === 'not_satisfied' ? 'Available evidence does not meet your targets.' : 'Performance for your targets is not established.'}</p>
                        <details className="mb-5 text-sm"><summary className="cursor-pointer text-theme-muted">Technical details and evidence</summary>
                            <p className="mt-3 break-all">{plan?.source_candidates?.[item.id] ? 'Identity and optimizations come from the reviewed source resources.' : item.provider_ref}</p>
                            <p className="mt-2 text-theme-muted">{item.performance_estimate_source || 'No performance prediction'}</p>
                            {item.performance_estimate && <dl className="mt-2 space-y-2">{[['TTFT (ms)', 'ttft_ms'], ['TPOT (ms)', 'tpot_ms'], ['Throughput (tokens/s)', 'throughput_tokens_per_sec']].map(([label, field]) => <div key={field}><dt className="inline text-theme-muted">{label}: </dt><dd className="inline">{Number.isFinite(item.performance_estimate[field]) ? item.performance_estimate[field].toFixed(2) : 'Unknown'}</dd></div>)}</dl>}
                            {item.historical_benchmarks?.length > 0 && <ul aria-label="Historical measurements" className="mt-3 space-y-2">{item.historical_benchmarks.map(sample => <li key={sample.benchmark_id}>{historicalBenchmarkLabel(sample)}</li>)}</ul>}
                        </details>
                        <Button variant="secondary" className="mt-auto w-full" disabled={locked || run.selected_candidate.id === item.id} onClick={() => choose(item.id)}>{run.selected_candidate.id === item.id ? 'Selected' : `Review plan ${index + 1}`}</Button>
                    </Panel>)}
                </fieldset>
                {run.rejected_candidates?.length > 0 && <details className="mt-3 text-sm"><summary>Excluded candidates ({run.rejected_candidates.length})</summary><ul className="mt-2 space-y-2">{run.rejected_candidates.map(item => <li key={item.id}>{item.provider_ref}: {item.rejection_reasons.join('; ')}</li>)}</ul></details>}
                <details className="mt-4"><summary className="cursor-pointer text-sm font-semibold">Planning evidence ({run.planning_evidence?.length || 0})</summary>{retrievalBranchSummaries(run).length > 0 && <div className="mt-3 divide-y divide-theme-border rounded border border-theme-border"><RetrievalEvidenceSources run={run} /></div>}<ul className="mt-3 space-y-3">{run.planning_evidence?.map(item => <li key={item.id} className="rounded border border-theme-border p-3 text-sm"><div className="flex flex-wrap gap-2"><strong>{item.source}</strong><Badge tone="neutral">{item.status}</Badge><Badge tone="neutral">{run.decision_evidence_ids?.includes(item.id) ? 'Used in decision' : 'Context'}</Badge></div><p className="mt-1 break-all text-xs text-theme-muted">{item.id}</p></li>)}</ul></details>
                <div className="mt-4 flex justify-end"><Button disabled={locked || editing} onClick={() => setWorkspaceView('review')}>Review selected configuration</Button></div>
            </section>}
            {workspaceView === 'execution' && run.deployment_run_id && <Panel title="Live deployment execution" actions={<Badge tone="neutral">{deploymentRun?.status || 'Loading execution'}</Badge>}>
                {deploymentRun ? <>
                    <dl className="mb-4 grid gap-3 text-sm md:grid-cols-3">{[['Created', deploymentRun.created_at], ['Started', deploymentRun.started_at], ['Finished', deploymentRun.finished_at]].map(([label, value]) => <div key={label}><dt className="text-xs text-theme-muted">{label}</dt><dd>{value ? formatTimestamp(value) : 'Not yet reported'}</dd></div>)}</dl>
                    <div className="overflow-auto"><table className="w-full text-left text-sm"><thead><tr className="border-b border-theme-border"><th className="p-2">Component</th><th className="p-2">Provider</th><th className="p-2">Reported status</th><th className="p-2">Attempt</th><th className="p-2">Diagnostics</th></tr></thead><tbody>{deploymentRun.cases?.map(item => <tr key={item.id} className="border-b border-theme-border"><td className="p-2">{item.component}</td><td className="p-2">{item.provider_ref}</td><td className="p-2"><Badge tone={item.status === 'ready' ? 'success' : item.status === 'failed' ? 'danger' : 'neutral'}>{item.status}</Badge></td><td className="p-2">{item.attempt}</td><td className="p-2">{item.failure && <p role="alert">{item.failure.title}: {item.failure.detail}</p>}{item.execution_id ? <Button size="sm" variant="secondary" onClick={() => setLogsDeployment({ execution_id: item.execution_id, name: run.request.deployment_name, cluster_id: draft.clusterId })}>View Pods &amp; Logs</Button> : <span className="text-theme-muted">Waiting for execution record</span>}</td></tr>)}</tbody></table></div>
                    <p className="mt-3 text-xs text-theme-muted">States are reported by Deploy and refreshed automatically. Readiness does not establish workload performance. Open the deployment to monitor metrics or run Evaluate.</p>
                </> : <LoadingState label="Reading deployment execution" />}
            </Panel>}
            {mode === 'agentic' && workspaceView === 'trials' && !run.request.execution_policy.allow_trials && !run.optimization && <Panel title="Optional benchmark tests">
                <p className="text-sm text-theme-muted">Prepare a test-enabled plan, then choose the workload and resource budget. Tests require a separate confirmation before using cluster resources.</p>
                <Button className="mt-3" disabled={locked || editing} onClick={async () => { if (await generate({ ...requirements, allowTrial: true })) setWorkspaceView('trials'); }}>Prepare benchmark plan</Button>
            </Panel>}
            {mode === 'agentic' && (run.request.execution_policy.allow_trials || run.optimization) && <div hidden={workspaceView !== 'trials'}><MeasuredOptimizationControls key={run.id} onBusyChange={setTrialBusy} run={run} onUpdate={value => { requestVersion.current += 1; accept(value); }} onReview={choose} onCheckCandidates={async () => { try { await renderSource({ compare: true }); } catch { /* Workflow reports source errors. */ } }} disabled={busy || !permitted || editing} /></div>}
            </div>

            </div>
        </>}
        <Modal isOpen={Boolean(approval)} onClose={() => setApproval(null)} closeOnBackdrop={!busy} closeOnEscape={!busy} title="Review & deploy" size="xl" footer={manifest && <Button variant="secondary" onClick={() => downloadBlob(new Blob([manifest], { type: 'text/yaml' }), `${draft.deploymentName}.yaml`)}><Download size={15} />Download YAML</Button>}>
            {run && <div className="grid items-start gap-5 md:grid-cols-[minmax(0,1fr)_300px]">
                <section className="min-w-0"><h3 className="mb-3 text-sm font-semibold">Generated configuration</h3><pre className="max-h-[65vh] overflow-auto rounded-lg bg-theme-bg p-4 text-xs">{manifest || 'Generate and check the source configuration first.'}</pre></section>
                <DeploymentSummary run={run} draft={draft} deploymentRun={deploymentRun} busy={busy} locked={locked || approval?.checksum !== plan?.configuration?.checksum} editing={editing} cleanupRequired={cleanupRequired} endpoint={endpoint}
                    onEndpointChange={value => { setEndpoint(value); setApproval(current => current ? { ...current, endpoint: value } : null); }}
                    onDeploy={deploy} confirmLabel="Confirm deployment" onCheck={render} onRequirements={reset} />
            </div>}
            {error && <p role="alert" className="mt-3 text-sm">{error}</p>}
        </Modal>
        {logsDeployment && <DeploymentPodsLogsModal deployment={logsDeployment} onClose={() => setLogsDeployment(null)} />}
        <Walkthrough steps={tourSteps} step={tourStep} onStepChange={setTourStep} onClose={() => setTourStep(null)} />
    </section>;
    return embedded ? content : <ModulePage>{content}</ModulePage>;
}

export default function AgenticDeploymentWorkspace(props) {
    return <DeploymentWorkflow {...props} mode="agentic" />;
}
