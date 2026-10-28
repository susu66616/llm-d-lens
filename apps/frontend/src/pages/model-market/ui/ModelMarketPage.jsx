import { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowLeft, ArrowRight, Box, Database, Rocket, Search, SlidersHorizontal, Sparkles } from 'lucide-react';
import { CATEGORIES, MODELS } from '@repo/shared/catalog';
import { requestJson } from '../../../shared/api/httpClient';
import { usePolling } from '../../../shared/lib/usePolling';
import { Badge, Button, EmptyState, Input, ModuleHeader, ModulePage, Panel, Select, StorageVolumeSelect, Textarea } from '../../../shared/ui';
import { listStorageVolumes } from '../../storage';
import { listModelCacheEntries } from '../../model-cache';
import { AgenticDeploymentWorkspace, ManualDeploymentWorkspace } from '../../deployment';
import { useAuth } from '../../auth';
import { consumePendingModelMarketCluster } from '../model/transfer';
import { standardRecommendation } from '../model/standardRecommendation';

function deploymentName(model) {
    const suffix = crypto.randomUUID().slice(0, 8);
    return `${String(model.id || 'model').slice(0, 54)}-${suffix}`;
}

export function ModelCard({ model, onSelect }) {
    return <Button variant="secondary" onClick={() => onSelect(model)} className="min-h-52 flex-col items-stretch p-5 text-left">
        <div className="flex items-start justify-between gap-4"><div><p className="text-xs text-theme-muted">{model.family}</p><h2 className="mt-2 text-lg font-semibold">{model.name}</h2></div><Box size={19} aria-hidden="true" /></div>
        <dl className="mt-5 grid grid-cols-2 gap-3 text-xs">
            {[['Released', model.releaseDate], ['Precision', model.dtype], ['Weights', model.weightGiB ? `${model.weightGiB} GiB` : 'Not listed'], ['Context', model.contextLength?.toLocaleString() || 'Not listed'], ['Model type', model.modelType]].map(([label, value]) => <div key={label}><dt className="text-theme-muted">{label}</dt><dd className="mt-1">{value}</dd></div>)}
        </dl>
    </Button>;
}

export default function ModelMarketPage({ onNavigate, isActive = true }) {
    const { can } = useAuth();
    const [selected, setSelected] = useState(() => {
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
    const [pendingCluster] = useState(() => consumePendingModelMarketCluster());
    const [hardware, setHardware] = useState(null);
    const cacheRequest = useRef(0);
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
        try {
            const payload = await requestJson('/api/cluster/clusters');
            const items = payload.items || [];
            setClusters(items);
            setCluster(current => current || (items.some(item => item.id === pendingCluster) ? pendingCluster : items.find(item => item.ready)?.id) || '');
        } catch (failure) { setError(failure.message); }
    }, [pendingCluster]);
    useEffect(() => { if (isActive) loadClusters(); }, [loadClusters, isActive]);
    usePolling(loadClusters, { enabled: isActive });
    useEffect(() => {
        let active = true;
        setHardware(null); setVolumes([]); setStorage(''); setCacheReady(false);
        if (!cluster || !isActive) return undefined;
        setLoadingStorage(true);
        Promise.all([
            requestJson(`/api/cluster/overview?clusterId=${encodeURIComponent(cluster)}`),
            listStorageVolumes({ clusterId: cluster, purpose: 'model-cache' }),
        ]).then(([overview, items]) => {
            if (!active) return;
            setHardware(overview.kubernetes?.hardware || null); setVolumes(items);
            setStorage(items.find(item => item.status === 'ready')?.id || '');
        }).catch(failure => { if (active) setError(failure.message); })
            .finally(() => { if (active) setLoadingStorage(false); });
        return () => { active = false; };
    }, [cluster, isActive]);
    const checkCache = useCallback(async () => {
        const request = ++cacheRequest.current;
        if (!selected || !storage || !cluster) { setCacheReady(false); return; }
        setCheckingCache(true);
        try {
            const entries = await listModelCacheEntries({ clusterId: cluster, storageVolumeId: storage });
            if (request === cacheRequest.current) setCacheReady(entries.some(entry => entry.status === 'ready' && entry.source?.kind === 'huggingface' && String(entry.source.huggingface.repoId || '').toLowerCase() === selected.repository.toLowerCase()));
        } catch (failure) { if (request === cacheRequest.current) { setCacheReady(false); setError(failure.message); } }
        finally { if (request === cacheRequest.current) setCheckingCache(false); }
    }, [selected, storage, cluster]);
    useEffect(() => { setCacheReady(false); if (isActive) checkCache(); return () => { cacheRequest.current += 1; }; }, [checkCache, isActive]);
    usePolling(checkCache, { enabled: isActive && Boolean(selected && storage && cluster), intervalMs: 10000 });
    const choose = model => { setSelected(model); setDescription(''); setDraft(null); setMode('manual'); setError(''); };
    const begin = event => {
        event.preventDefault(); setError('');
        if (!cacheReady) { setError('The selected model must be ready in the selected model cache.'); return; }
        const defaults = standardRecommendation(selected, hardware);
        setDraft({
            model: selected.repository, modelWeightGiB: selected.weightGiB || selected.sizeGiB,
            deploymentName: name, description, clusterId: cluster, clusterName: clusters.find(item => item.id === cluster)?.name, storageVolumeId: storage,
            contextLength: defaults.maxModelLen, gpuMemoryUtilization: 0.9,
            maxNumSeqs: 256, maxNumBatchedTokens: 8192, vllmArguments: selected.vllmArguments || [],
        });
    };
    const words = search.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const models = MODELS.filter(model => (category === 'All' || model.category === category) && words.every(word => [model.name, model.family, model.repository, model.modelType, model.category, model.dtype].some(value => String(value || '').toLowerCase().includes(word))));
    if (!selected) return <ModulePage contentClassName="flex flex-col gap-4">
        <ModuleHeader icon={Database} title="Model Catalog" />
        <div className="flex flex-wrap items-center gap-3">
            <div className="flex flex-wrap gap-2" role="tablist" aria-label="Model categories">{CATEGORIES.map(item => <Button key={item} role="tab" size="sm" variant={category === item ? 'primary' : 'secondary'} aria-selected={category === item} onClick={() => setCategory(item)}>{item}</Button>)}</div>
            <Input type="search" icon={Search} aria-label="Search models" value={search} onChange={event => setSearch(event.target.value)} placeholder="Search models, publishers, repositories, types, or precision..." />
            <span role="status" className="text-xs text-theme-muted">{models.length} models</span>
            {(search || category !== 'All') && <Button size="sm" variant="secondary" onClick={() => { setSearch(''); setCategory('All'); }}>Clear filters</Button>}
        </div>
        <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">{models.map(model => <ModelCard key={model.id} model={model} onSelect={choose} />)}{!models.length && <EmptyState title="No matching models" message="Try another search or category." />}</section>
    </ModulePage>;
    const allowed = can('deployment:run:create');
    const Workspace = mode === 'manual' ? ManualDeploymentWorkspace : AgenticDeploymentWorkspace;
    return <ModulePage contentClassName="flex flex-col gap-5">
        <ModuleHeader icon={Rocket} title="Create deployment" />
        <Button className="self-start" variant="secondary" onClick={() => { setSelected(null); setDraft(null); }}><ArrowLeft size={15} />Model Catalog</Button>
        {!allowed ? <EmptyState title="Your role cannot create deployments" message="Use an existing deployment or request deployment-create permission." action={<Button onClick={() => onNavigate('optimization-deployments')}>Open Deployments</Button>} /> : draft ? <>
            <Panel title={selected.name} actions={<Badge tone="neutral">{mode === 'manual' ? 'Manual' : 'Agentic'}</Badge>}><p className="text-sm text-theme-muted">{name} / {clusters.find(item => item.id === cluster)?.name || cluster}</p></Panel>
            <Workspace key={`${mode}:${draft.clusterId}:${draft.model}:${draft.deploymentName}`} embedded draft={draft} onBack={() => setDraft(null)} onDeploymentStarted={run => onNavigate('optimization-deployments', { clusterId: draft.clusterId, deploymentSearch: run?.deployment_execution_id || run?.deployment_run_id || draft.deploymentName })} />
        </> : <form onSubmit={begin} className="space-y-5">
            <Panel title="Deployment target" actions={<Badge tone="neutral">{selected.family}</Badge>}>
                <h2 className="mb-4 text-lg font-semibold">{selected.name}</h2>
                <div className="grid gap-4 md:grid-cols-2">
                    <label className="text-sm">Cluster<Select className="mt-1" required value={cluster} onChange={event => setCluster(event.target.value)}><option value="">Select cluster</option>{clusters.map(item => <option key={item.id} value={item.id} disabled={!item.ready}>{item.name}{item.ready ? '' : ' (not ready)'}</option>)}</Select></label>
                    <label className="text-sm">Model cache storage<StorageVolumeSelect className="mt-1" value={storage} onChange={setStorage} volumes={volumes} loading={loadingStorage} disabled={!cluster} placeholder="Select ready model storage" /><p className="mt-2 text-xs text-theme-muted">{checkingCache ? 'Checking model cache...' : cacheReady ? 'Model weights are ready.' : 'Download this model into the selected storage before continuing.'}</p></label>
                </div>
                <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
                    <p className="text-xs text-theme-muted">Lens fills in the deployment name and model runtime settings.</p>
                    <Button type="button" size="sm" variant="link" onClick={() => onNavigate('model-cache', { downloadModel: { clusterId: cluster, repoId: selected.repository, storageVolumeId: storage } })}>Manage model cache</Button>
                </div>
                <details className="mt-4 border-t border-theme-border pt-4"><summary className="cursor-pointer text-sm font-medium">Name and description <span className="font-normal text-theme-muted">(optional)</span></summary>
                    <div className="mt-4 grid gap-4 md:grid-cols-2">
                        <label className="text-sm">Deployment name<Input className="mt-1" required value={name} onChange={event => setName(event.target.value)} /></label>
                        <label className="text-sm">Description<Textarea className="mt-1" rows={2} value={description} maxLength={1000} onChange={event => setDescription(event.target.value)} placeholder="What is this deployment for?" /></label>
                    </div>
                </details>
            </Panel>
            <Panel title="How would you like to build your plan?">
                <div className="grid gap-3 md:grid-cols-2">{[
                    ['manual', 'Manual', 'Start with a compatible configuration. Inspect and edit it yourself. No LLM required.', <SlidersHorizontal key="manual" size={18} />],
                    ['agentic', 'Agentic', 'Compare plans for your goals, with optional AI assistance and benchmark trials.', <Sparkles key="agentic" size={18} />],
                ].map(([id, label, detail, icon]) => <Button type="button" key={id} variant={mode === id ? 'primary' : 'secondary'} className="h-auto flex-col items-start gap-3 whitespace-normal p-5 text-left" aria-pressed={mode === id} onClick={() => setMode(id)}><span className="flex items-center gap-2 font-semibold">{icon}{label}</span><span className="text-xs leading-relaxed">{detail}</span></Button>)}</div>
                <p className="mt-4 text-xs text-theme-muted">Both modes include interactive 2D / 3D editing and YAML review. Nothing deploys without your approval.</p>
            </Panel>
            {error && <p role="alert" className="text-sm text-theme-text">{error}</p>}
            {!selected.vllmSupported && <p role="alert">This model requires a runtime not supported by this workflow.</p>}
            <div className="flex justify-end"><Button type="submit" disabled={!cluster || !name || !cacheReady || checkingCache || !selected.vllmSupported}>Continue to deployment goals<ArrowRight size={15} /></Button></div>
        </form>}
    </ModulePage>;
}
