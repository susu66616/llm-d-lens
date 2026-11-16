import { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowLeft, ArrowRight, Box, CheckCircle2, Cpu, Database, Rocket, Search, Server, SlidersHorizontal, Sparkles } from 'lucide-react';
import { CATEGORIES, MODELS } from '@repo/shared/catalog';
import { requestJson } from '../../../shared/api/httpClient';
import { usePolling } from '../../../shared/lib/usePolling';
import { Badge, Button, EmptyState, Input, Modal, ModuleHeader, ModulePage, Panel, Select, StorageVolumeSelect, Walkthrough } from '../../../shared/ui';
import { cn } from '../../../shared/lib/cn';
import { FormError } from '../../../shared/ui/FormError';
import { listStorageVolumes } from '../../storage';
import { listModelCacheEntries } from '../../model-cache';
import { AgenticDeploymentWorkspace, ManualDeploymentWorkspace } from '../../deployment';
import { useAuth } from '../../auth';
import { consumePendingModelMarketCluster } from '../model/transfer';
import { standardRecommendation } from '../model/standardRecommendation';
import { DEPLOYMENT_PREVIEW, mockEnvironmentFromProfiles } from '../../deployment/api/agenticDeploymentBackend';
import { MockDeploymentWorkspace } from '../../deployment/ui/ManualDeploymentWorkspace';
import { getHardwareProfiles } from '../../clusters/ui/ClusterMonitoringStack/hardwareProfilesBackend';

function deploymentName(model) {
    const suffix = crypto.randomUUID().slice(0, 8);
    return `${String(model.id || 'model').slice(0, 54)}-${suffix}`;
}

export function ModelCard({ model, onSelect }) {
    return (
        <Button type="button" variant="catalogCard" onClick={() => onSelect(model)}
            className="group flex h-auto min-h-52 flex-col items-stretch justify-start gap-0 whitespace-normal rounded-lg p-5 text-left font-normal">
            <div className="flex items-start justify-between gap-4">
                <div>
                    <p className="text-xs font-medium uppercase text-[var(--brand-accent)] dark:text-cyan-300">{model.family}</p>
                    <h2 className="mt-2 text-lg font-semibold text-slate-900 dark:text-white">{model.name}</h2>
                </div>
                <Box size={19} className="shrink-0 text-[var(--brand-accent)] transition dark:text-slate-500 dark:group-hover:text-cyan-300" aria-hidden="true" />
            </div>
            <dl className="mt-5 grid grid-cols-2 gap-x-3 gap-y-4 text-xs">
                <div><dt className="text-slate-500">Released</dt><dd className="mt-1 text-slate-700 dark:text-slate-200">{model.releaseDate}</dd></div>
                <div><dt className="text-slate-500">Precision</dt><dd className="mt-1 text-slate-700 dark:text-slate-200">{model.dtype}</dd></div>
                <div><dt className="text-slate-500">Weights</dt><dd className="mt-1 text-slate-700 dark:text-slate-200">{model.weightGiB ? `${model.weightGiB} GiB` : 'Not listed'}</dd></div>
                <div><dt className="text-slate-500">Context</dt><dd className="mt-1 text-slate-700 dark:text-slate-200">{model.contextLength ? model.contextLength.toLocaleString() : 'Not listed'}</dd></div>
                <div className="col-span-2"><dt className="text-slate-500">Model type</dt><dd className="mt-1 text-slate-700 dark:text-slate-200">{model.modelType}</dd></div>
            </dl>
        </Button>
    );
}

export default function ModelMarketPage({ onNavigate, isActive = true }) {
    const { can } = useAuth();
    const [previewRequested, setPreviewRequested] = useState(() => typeof window !== 'undefined' && new URLSearchParams(window.location.search).get('preview') === DEPLOYMENT_PREVIEW.id);
    const [preview, setPreview] = useState(null), [targetOpen, setTargetOpen] = useState(false);
    const [previewReload, setPreviewReload] = useState(0);
    const [selected, setSelected] = useState(() => {
        if (typeof window !== 'undefined' && new URLSearchParams(window.location.search).get('preview') === DEPLOYMENT_PREVIEW.id) return MODELS.find(model => model.repository === DEPLOYMENT_PREVIEW.model);
        const saved = sessionStorage.getItem('prism_model_market_return_draft');
        if (!saved) return null;
        try { return MODELS.find(model => model.repository === JSON.parse(saved)?.model) || null; }
        catch { return null; }
    });
    const [category, setCategory] = useState('All'), [search, setSearch] = useState('');
    const [name, setName] = useState(''), [description, setDescription] = useState('');
    const [cluster, setCluster] = useState(''), [clusters, setClusters] = useState([]);
    const [storage, setStorage] = useState(''), [volumes, setVolumes] = useState([]);
    const [loadingStorage, setLoadingStorage] = useState(false), [cacheReady, setCacheReady] = useState(false);
    const [checkingCache, setCheckingCache] = useState(false), [error, setError] = useState('');
    const [mode, setMode] = useState('manual'), [draft, setDraft] = useState(null);
    const [objective, setObjective] = useState('balanced');
    const [previousPlans, setPreviousPlans] = useState([]);
    const [pendingCluster] = useState(() => consumePendingModelMarketCluster());
    const [hardware, setHardware] = useState(null);
    const [tourStep, setTourStep] = useState(null);
    const cacheRequest = useRef(0);
    useEffect(() => {
        if (!previewRequested) return undefined;
        let active = true;
        getHardwareProfiles().then(payload => { if (active) { setPreview(mockEnvironmentFromProfiles(payload.profiles || [])); setError(''); } })
            .catch(failure => { if (active) setError(failure.message); });
        return () => { active = false; };
    }, [previewRequested, previewReload]);
    useEffect(() => {
        if (!selected) return;
        const saved = sessionStorage.getItem('prism_model_market_return_draft_values');
        sessionStorage.removeItem('prism_model_market_return_draft');
        sessionStorage.removeItem('prism_model_market_return_draft_values');
        if (!saved) { setName(deploymentName(selected)); return; }
        try {
            const values = JSON.parse(saved);
            setName(values.deployName || deploymentName(selected));
            setDescription(values.description || '');
            setCluster(values.cluster || '');
        } catch { setError('The saved model draft could not be restored. Choose deployment requirements again.'); }
    }, [selected]);
    const loadClusters = useCallback(async () => {
        if (previewRequested) return;
        try {
            const payload = await requestJson('/api/cluster/clusters');
            const items = payload.items || [];
            setClusters(items);
            setCluster(current => current || (items.some(item => item.id === pendingCluster) ? pendingCluster : items.find(item => item.ready)?.id) || '');
        } catch (failure) { setError(failure.message); }
    }, [pendingCluster, previewRequested]);
    useEffect(() => { if (isActive) loadClusters(); }, [loadClusters, isActive]);
    usePolling(loadClusters, { enabled: isActive && !previewRequested });
    useEffect(() => {
        let active = true;
        setHardware(null); setVolumes([]); setStorage(''); setCacheReady(false);
        if (!cluster || !isActive || previewRequested) return undefined;
        setLoadingStorage(true);
        requestJson(`/api/cluster/overview?clusterId=${encodeURIComponent(cluster)}`)
            .then(overview => { if (active) setHardware(overview.kubernetes?.hardware || null); })
            .catch(failure => { if (active) setError(failure.message); });
        listStorageVolumes({ clusterId: cluster, purpose: 'model-cache' }).then(items => {
            if (!active) return;
            setVolumes(items);
            setStorage(items.find(item => item.status === 'ready')?.id || '');
        }).catch(failure => { if (active) setError(failure.message); })
            .finally(() => { if (active) setLoadingStorage(false); });
        return () => { active = false; };
    }, [cluster, isActive, previewRequested]);
    const checkCache = useCallback(async () => {
        const request = ++cacheRequest.current;
        if (previewRequested || !selected || !storage || !cluster) { setCacheReady(false); return; }
        setCheckingCache(true);
        try {
            const entries = await listModelCacheEntries({ clusterId: cluster, storageVolumeId: storage });
            if (request === cacheRequest.current) setCacheReady(entries.some(entry => entry.status === 'ready' && entry.source?.kind === 'huggingface' && String(entry.source.huggingface.repoId || '').toLowerCase() === selected.repository.toLowerCase()));
        } catch (failure) { if (request === cacheRequest.current) { setCacheReady(false); setError(failure.message); } }
        finally { if (request === cacheRequest.current) setCheckingCache(false); }
    }, [selected, storage, cluster, previewRequested]);
    useEffect(() => { setCacheReady(false); if (isActive) checkCache(); return () => { cacheRequest.current += 1; }; }, [checkCache, isActive]);
    usePolling(checkCache, { enabled: isActive && !previewRequested && Boolean(selected && storage && cluster), intervalMs: 10000 });
    const changePreview = enabled => {
        setPreviewRequested(enabled); setError('');
        const url = new URL(window.location.href);
        if (enabled) url.searchParams.set('preview', DEPLOYMENT_PREVIEW.id);
        else url.searchParams.delete('preview');
        window.history.replaceState({}, '', url);
    };
    const choose = model => { setSelected(model); setDescription(''); setDraft(null); setMode('manual'); changePreview(false); };
    const openPreview = () => { setSelected(MODELS.find(model => model.repository === DEPLOYMENT_PREVIEW.model)); setDraft(null); changePreview(true); setTargetOpen(false); };
    const begin = event => {
        event.preventDefault(); setError(''); setPreviousPlans([]);
        if (previewRequested) {
            if (!preview) { setError('Wait for the hardware profile to load.'); return; }
            setTourStep(null);
            setDraft({ model: selected.repository, modelTitle: selected.name, deploymentName: name || 'qwen3-mock', clusterId: preview.clusterId, clusterName: preview.clusterName, initialObjective: mode === 'manual' ? 'balanced' : objective, preview });
            return;
        }
        if (!cacheReady) { setError('The selected model must be ready in the selected model cache.'); return; }
        setTourStep(null);
        const defaults = standardRecommendation(selected, hardware);
        setDraft({
            model: selected.repository, modelTitle: selected.name, modelWeightGiB: selected.weightGiB || selected.sizeGiB,
            deploymentName: name, description, initialObjective: mode === 'manual' ? 'balanced' : objective, clusterId: cluster, clusterName: clusters.find(item => item.id === cluster)?.name, storageVolumeId: storage, storageVolumeName: volumes.find(item => item.id === storage)?.name,
            contextLength: defaults.maxModelLen, gpuMemoryUtilization: 0.9,
            maxNumSeqs: 256, maxNumBatchedTokens: 8192, vllmArguments: selected.vllmArguments || [],
        });
    };
    const words = search.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const models = MODELS.filter(model => (category === 'All' || model.category === category) && words.every(word => [model.name, model.family, model.repository, model.modelType, model.category, model.dtype].some(value => String(value || '').toLowerCase().includes(word))));
    if (!selected) return <ModulePage contentClassName="flex flex-col gap-4 py-4 lg:py-5">
        <ModuleHeader icon={Database} title="Model Catalog" />
        <div className="flex flex-wrap items-center justify-between gap-3"><p className="text-sm text-theme-muted">Choose a model to deploy.</p><Button variant="secondary" onClick={openPreview}><Cpu size={16} />Try {DEPLOYMENT_PREVIEW.label} mock</Button></div>
        <div className="flex flex-wrap items-center gap-3">
            <div className="flex min-w-0 w-full flex-wrap gap-2 lg:w-auto lg:flex-1" role="tablist" aria-label="Model categories">{CATEGORIES.map(item => <Button type="button" key={item} role="tab" size="sm" variant="catalogTab" className="h-auto rounded-md px-3 py-1.5 font-normal" aria-selected={category === item} onClick={() => setCategory(item)}>{item}</Button>)}</div>
            <div className="ml-auto flex max-w-full flex-wrap items-center gap-2">
                <span role="status" className="text-xs text-theme-muted">{models.length} models</span>
                <div className="w-52 sm:w-64">
                    <Input type="search" icon={Search} aria-label="Search models" value={search} onChange={event => setSearch(event.target.value)} placeholder="Search models, publishers, repositories, types, or precision..." />
                </div>
                {(search || category !== 'All') && <Button type="button" size="sm" variant="secondary" onClick={() => { setSearch(''); setCategory('All'); }}>Clear filters</Button>}
            </div>
        </div>
        <section className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">{models.map(model => <ModelCard key={model.id} model={model} onSelect={choose} />)}{!models.length && <EmptyState className="col-span-full" icon={<Search size={22} aria-hidden="true" />} title="No matching models" message="Try another search or change the model category." />}</section>
    </ModulePage>;
    const allowed = can('deployment:run:create');
    const suggestAgentic = ({ requirements, provider, runId, mockSettings, mockRuntime, runtime }) => {
        setPreviousPlans(current => [...current, { mode, draft: { ...draft, resumeRunId: runId, initialMockSettings: mockSettings } }]);
        setMode('agentic');
        setDraft({ ...draft, ...runtime, planningAttemptId: crypto.randomUUID(), resumeRunId: undefined,
            initialObjective: requirements.objective, initialRequirements: requirements, initialProvider: provider,
            initialMockSettings: mockRuntime });
    };
    const restorePlan = () => {
        const previous = previousPlans.at(-1);
        if (!previous) return;
        setMode(previous.mode); setDraft(previous.draft); setPreviousPlans(current => current.slice(0, -1));
    };
    const Workspace = draft?.preview ? MockDeploymentWorkspace : mode === 'manual' ? ManualDeploymentWorkspace : AgenticDeploymentWorkspace;
    return <ModulePage contentClassName="mx-auto flex max-w-7xl flex-col gap-5">
        {!draft && <ModuleHeader icon={Rocket} title="Create deployment" />}
        {!draft && <Button className="self-start" variant="ghost" onClick={() => { setSelected(null); setDraft(null); }}><ArrowLeft size={15} />Model Catalog</Button>}
        {!allowed ? <EmptyState title="Your role cannot create deployments" message="Use an existing deployment or request deployment-create permission." action={<Button onClick={() => onNavigate('optimization-deployments')}>Open Deployments</Button>} /> : draft ? <>
            <Workspace key={`${mode}:${draft.clusterId}:${draft.model}:${draft.deploymentName}:${draft.planningAttemptId || 'initial'}`} mode={mode} embedded draft={draft} onBack={() => { setDraft(null); setPreviousPlans([]); }} onSuggestAgentic={suggestAgentic} onRestorePlan={previousPlans.length ? restorePlan : undefined} onDeploymentStarted={run => onNavigate('optimization-deployments', { clusterId: draft.clusterId, deploymentSearch: run?.deployment_execution_id || run?.deployment_run_id || draft.deploymentName })} />
        </> : <form onSubmit={begin} onInvalid={event => event.target.closest('details')?.setAttribute('open', '')} className="space-y-5">
            <section id="deployment-mode" aria-label="Planning mode" className="space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-3"><div><p className="mb-2 text-xs font-semibold uppercase tracking-wider text-theme-muted">{selected.name} · Deployment</p><h2 className="text-3xl font-semibold tracking-tight">Choose how to build your plan</h2></div><Button type="button" variant="ghost" onClick={() => setTourStep(0)}>1–3 Walkthrough</Button></div>
                <div className="grid gap-4 md:grid-cols-2">{[
                    ['manual', 'Manual', 'Compose, then adjust', 'Include supported, compatible settings.', ['Compatibility', 'Configuration', 'Your review'], <SlidersHorizontal key="manual" size={25} />],
                    ['agentic', 'Agentic', 'Start with your objective', 'Recommend a configuration for your goal.', ['Your objective', 'Recommendation', 'Your review'], <Sparkles key="agentic" size={25} />],
                ].map(([id, label, subtitle, detail, path, icon]) => <Button type="button" key={id} variant="outline" size="md" className={cn('deployment-mode-card', mode === id && 'is-selected')} aria-pressed={mode === id} onClick={() => setMode(id)}>
                    <span className="flex w-full items-center justify-between"><span className="deployment-mode-icon">{icon}</span><Badge tone={mode === id ? 'brand' : 'neutral'}>{mode === id ? 'Selected' : 'Choose'}</Badge></span>
                    <span className="text-3xl font-semibold tracking-tight text-theme-text">{label}</span><span className="text-base font-medium">{subtitle}</span><span className="text-sm font-normal text-theme-muted">{detail}</span>
                    <span className="deployment-mode-path">{path.map((part, index) => <span key={part}>{index > 0 && <ArrowRight size={12} aria-hidden="true" />}{part}</span>)}</span>
                </Button>)}</div>
            </section>
            <Panel id="deployment-continue" tone="subtle" className="shadow-none">
                <div className="flex flex-wrap items-center justify-between gap-5">
                    <div><p className="mb-2 text-xs font-semibold uppercase tracking-wider text-theme-muted">{mode === 'manual' ? 'Manual' : 'Agentic'}</p><h3 className="text-lg font-semibold">{mode === 'manual' ? 'Start with compatible defaults' : 'What should the plan prioritize?'}</h3>
                        {mode === 'manual' ? <p className="mt-2 text-sm text-theme-muted">Review the generated plan and adjust any supported setting.</p> : <div className="mt-3 flex flex-wrap gap-2" aria-label="Planning objective">{[['balanced', 'Balanced'], ['latency', 'Low latency'], ['throughput', 'High throughput']].map(([id, label]) => <Button key={id} type="button" variant={objective === id ? 'primary' : 'secondary'} aria-pressed={objective === id} onClick={() => setObjective(id)}>{label}</Button>)}</div>}
                    </div>
                    <Button type="submit" size="md" disabled={previewRequested ? !preview : !cluster || !name || !cacheReady || checkingCache || !selected.vllmSupported}>{mode === 'manual' ? 'Build Manual plan' : 'Build Agentic plan'}<ArrowRight size={15} /></Button>
                </div>
            </Panel>
            <Panel id="deployment-target" padding="sm">
                {previewRequested ? <>
                    <div className="flex flex-wrap items-center justify-between gap-3"><div className="flex items-center gap-3"><Cpu size={22} /><div><h3 className="font-semibold">{DEPLOYMENT_PREVIEW.label} sandbox</h3><p className="text-sm text-theme-muted">{DEPLOYMENT_PREVIEW.nodes} nodes · {DEPLOYMENT_PREVIEW.devices} × {DEPLOYMENT_PREVIEW.deviceModel} · {DEPLOYMENT_PREVIEW.memoryGiB} GiB / device</p></div></div><Badge tone="warning" size="md">Mock</Badge></div>
                    <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-theme-border pt-3"><span className="flex items-center gap-2 text-sm">{preview && <CheckCircle2 size={16} />}{preview ? 'Model cache ready · simulated' : error ? 'Hardware profile unavailable' : 'Loading hardware profile…'}</span><Button type="button" variant="ghost" size="sm" onClick={() => changePreview(false)}>Use a real cluster</Button></div>
                    {!preview && error && <Button className="mt-3" type="button" variant="secondary" onClick={() => { setError(''); setPreviewReload(value => value + 1); }}>Retry hardware profile</Button>}
                </> : cluster ? <>
                    <div className="flex flex-wrap items-center justify-between gap-3"><div className="flex items-center gap-3"><Server size={22} /><div><h3 className="font-semibold">{clusters.find(item => item.id === cluster)?.name || cluster}</h3><p role="status" className="text-sm text-theme-muted">{loadingStorage || (checkingCache && !cacheReady) ? 'Checking model cache…' : cacheReady ? 'Model cache ready' : 'Model cache needs preparation'}</p></div></div><Button type="button" size="sm" variant="secondary" onClick={() => setTargetOpen(true)}>Change setup</Button></div>
                    {!cacheReady && !loadingStorage && <Button type="button" className="mt-3" size="sm" variant="secondary" onClick={() => onNavigate('model-cache', { downloadModel: { clusterId: cluster, repoId: selected.repository, storageVolumeId: storage } })}>Prepare model cache<ArrowRight size={14} /></Button>}
                </> : <div className="flex flex-wrap items-center justify-between gap-3"><div className="flex items-center gap-3"><Server size={22} /><div><h3 className="font-semibold">Choose an environment</h3><p className="text-sm text-theme-muted">Connect a cluster, or explore with a ready mock.</p></div></div><div className="flex gap-2"><Button type="button" variant="secondary" onClick={() => clusters.length ? setTargetOpen(true) : onNavigate('clusters')}>Choose cluster</Button><Button type="button" onClick={openPreview}>Try {DEPLOYMENT_PREVIEW.label} mock</Button></div></div>}
            </Panel>
            <FormError message={error} />
            {!selected.vllmSupported && <p role="alert">This model requires a runtime not supported by this workflow.</p>}
        </form>}
        <Modal isOpen={targetOpen} onClose={() => setTargetOpen(false)} title="Deployment setup" size="sm" footer={<Button onClick={() => setTargetOpen(false)}>Done</Button>}>
            <div className="space-y-4">
                <label className="block text-sm">Cluster<Select className="mt-1" value={cluster} onChange={event => setCluster(event.target.value)}><option value="">Select cluster</option>{clusters.map(item => <option key={item.id} value={item.id} disabled={!item.ready}>{item.name}{item.ready ? '' : ' (not ready)'}</option>)}</Select></label>
                <label className="block text-sm">Deployment name<Input className="mt-1" value={name} onChange={event => setName(event.target.value)} /></label>
                {cluster && <label className="block text-sm">Model cache<StorageVolumeSelect className="mt-1" value={storage} onChange={setStorage} volumes={volumes} loading={loadingStorage} placeholder="Select model cache" /></label>}
            </div>
        </Modal>
        {!draft && <Walkthrough step={tourStep} onStepChange={setTourStep} onClose={() => setTourStep(null)} steps={[
            { target: '#deployment-mode', title: 'Choose a planning mode', description: 'Manual combines supported compatible optimizations. Agentic recommends a plan for your objective. Both remain editable.' },
            { target: '#deployment-target', title: 'Reuse the current setup', description: 'The model is already selected. Lens discovers cluster resources and checks model files. Use Change setup only when needed. The mock includes simulated model files.' },
            { target: '#deployment-continue', title: 'Build, then review', description: 'Continue with this mode. Numeric workload and latency targets are optional. Deployment requires a separate confirmation.' },
        ]} />}
    </ModulePage>;
}
