import { useCallback, useEffect, useMemo, useState } from 'react';
import { Settings2, X } from 'lucide-react';
import { Badge, Button, LoadingState, Modal, Panel } from '../../../../shared/ui';
import { cn } from '../../../../shared/lib/cn';
import { useAuth } from '../../../auth';
import { getHardwareProfiles } from '../../../clusters/ui/ClusterMonitoringStack/hardwareProfilesBackend';
import { loadClusterOverview } from '../../../optimization/ui/OptimizationWorkspace/clusterBackend';
import { createBlueprint, blueprintClusterProfiles } from '../../model/inference-designer/blueprint';
import { capabilityTogglePatch, deploymentSelection, nodeParameterPaths, selectionConflicts } from '../../model/inference-designer/deploymentSelection';
import { compositionDisplayDraft } from '../../model/inference-designer/compositionPolicy';
import { inferenceScene, TIERS } from '../../model/inference-designer/sceneGraph';
import { nodeResourceKinds } from '../../model/inference-designer/nodeParameters';
import { recommendOptimizations } from '../../model/inference-designer/optimizationPolicy';
import { loadSourceCatalog } from './sourceCompositionBackend';
import { InferenceFlowMap } from './InferenceFlowMap';
import { InferenceFlowMap2D } from './InferenceFlowMap2D';
import { InferenceGuidePresets } from './InferenceGuidePresets';
import { InferenceSourceComposer } from './InferenceSourceComposer';
import { InferenceDeploymentParameters } from './InferenceDeploymentParameters';
import { InferenceParameterInput } from './InferenceParameterInput';
import './inferenceDesigner.css';

/** Edits one source-backed workflow snapshot; deployment stays approval-gated. */
export function InferenceDesigner({ initialDraft, initialResult, clusterName, disabled, onReview, onEditingChange }) {
    const { can } = useAuth();
    const [intent, setIntent] = useState(() => ({ ...createBlueprint(), ...initialDraft }));
    const [catalog, setCatalog] = useState(null), [profile, setProfile] = useState(null);
    const [loading, setLoading] = useState(true), [error, setError] = useState('');
    const [dirty, setDirty] = useState(false), [saving, setSaving] = useState(false);
    const [mapView, setMapView] = useState('2d');
    const [selected, setSelected] = useState('routing'), [parameters, setParameters] = useState(null);
    const [focus, setFocus] = useState(null), [reload, setReload] = useState(0);
    const select = useCallback(id => { setSelected(id); if (id) setFocus({ id }); else setParameters(null); }, []);
    useEffect(() => {
        let active = true;
        Promise.resolve().then(() => { if (active) { setLoading(true); setError(''); } });
        Promise.all([loadSourceCatalog(intent.clusterId), getHardwareProfiles(), loadClusterOverview(intent.clusterId)])
            .then(([source, hardware, overview]) => {
                if (!active) return;
                setCatalog(source);
                const profiles = blueprintClusterProfiles(overview, hardware.profiles || []);
                setProfile(profiles.find(item => item.id === intent.profileId) || null);
            }).catch(reason => { if (active) setError(reason.message); })
            .finally(() => { if (active) setLoading(false); });
        return () => { active = false; };
    }, [intent.clusterId, intent.profileId, reload]);
    useEffect(() => { onEditingChange?.(dirty || Boolean(parameters)); return () => onEditingChange?.(false); }, [dirty, parameters, onEditingChange]);
    const plan = useMemo(() => catalog ? deploymentSelection(catalog, intent.selectedGuides, {
        model: intent.model, variables: intent.sourceVariables, namespace: intent.sourceNamespace,
        capabilities: intent.capabilityOverrides, guideChoices: intent.guideChoices, mode: intent.sourceRouter.mode,
        accelerator: intent.accelerator, units: intent.sourceUnits, routerValues: intent.sourceRouter.values,
        finalConfiguration: intent.finalConfiguration,
    }) : initialDraft.sourcePlan, [catalog, intent, initialDraft.sourcePlan]);
    const draft = useMemo(() => !plan ? intent : ({ ...intent, sourcePlan: plan,
        sourceGuides: catalog?.guides.map(item => ({ id: item.id, title: item.title, disposition: item.disposition, hasAssets: catalog.assets.some(asset => asset.guide === item.id || asset.guide.startsWith(`${item.id}/`)) })) || intent.sourceGuides,
        sourceUnits: plan.units.map(unit => ({ ...unit, ...intent.sourceUnits.find(item => item.path === unit.path) })),
        sourceRouter: { ...intent.sourceRouter, ...plan.router, values: plan.router.values.map(layer => ({ ...layer, ...intent.sourceRouter.values.find(item => item.path === layer.path) })) },
    }), [intent, plan, catalog]);
    const result = dirty ? null : initialResult;
    const display = useMemo(() => compositionDisplayDraft(draft, result), [draft, result]);
    const graph = useMemo(() => inferenceScene(display, clusterName), [display, clusterName]);
    const entity = [...graph.nodes, ...graph.capabilities].find(item => item.id === selected);
    const stale = catalog && catalog.commit !== draft.sourceCommit;
    const locked = disabled || saving || loading || !catalog || stale;
    const canConfigure = item => nodeParameterPaths(item, draft).length > 0
        || plan?.choices?.some(choice => choice.guide === item?.guide) || ['prefix', 'chunked'].includes(item?.id)
        || (Boolean(draft.runtimeConfiguration) && (item?.enabled || item?.inPlan)
            && (item?.kind === 'capability' ? Boolean(nodeResourceKinds(item)) || item.id === 'cache'
                : ['prefill', 'decode', 'prefill-worker', 'decode-worker', 'serving-worker', 'pods'].includes(item?.id)));
    const blocked = Object.fromEntries(graph.capabilities.map(cap => {
        const patch = capabilityTogglePatch(cap, draft, catalog);
        return [cap.id, !catalog ? 'Source catalog unavailable.' : cap.id === 'cache' && !cap.enabled && draft.nativeCpuCacheGiB && draft.capabilityOverrides.prefix === false ? 'Enable prefix caching before restoring CPU KV cache.'
            : cap.required ? 'Required by the selected configuration. Change the dependent optimization or source recipe first.'
            : cap.selection === 'recipe' || cap.selection === 'parameter' ? 'Controlled by source parameters; edit this node or its Guide in Technical details.'
                : ['external-operation', 'requires-adapter', 'unavailable'].includes(cap.generation)
                    ? cap.reason : !cap.enabled && patch ? selectionConflicts(catalog, draft, patch).join(' ') : ''];
    }));
    const change = (patch, automatic = false) => {
        if (locked) return false;
        const previous = selectionConflicts(catalog, draft);
        const conflicts = selectionConflicts(catalog, draft, patch).filter(message => !previous.includes(message));
        if (conflicts.length) { setError(conflicts.join(' ')); return false; }
        if (!automatic && patch.capabilityOverrides) patch = { ...patch, optimizationLocks: [...new Set([...(draft.optimizationLocks || []),
            ...Object.keys({ ...draft.capabilityOverrides, ...patch.capabilityOverrides }).filter(id => draft.capabilityOverrides[id] !== patch.capabilityOverrides[id])])] };
        if (patch.selectedGuides?.includes('tiered-prefix-cache')) patch = { ...patch, nativeCpuCacheGiB: undefined };
        setError(''); setDirty(true);
        setIntent(current => ({ ...current, ...patch }));
        return true;
    };
    const activate = id => {
        select(id);
        const cap = graph.capabilities.find(item => item.id === id);
        if (!cap || locked) return;
        if (blocked[id]) { setError(blocked[id]); return; }
        const patch = capabilityTogglePatch(cap, draft, catalog);
        if (patch) change(patch);
    };
    const configure = id => {
        select(id);
        const item = [...graph.nodes, ...graph.capabilities].find(item => item.id === id);
        setParameters(!locked && item && !item.unavailable && canConfigure(item) ? 'node' : null);
    };
    const save = async () => {
        if (locked) return;
        setSaving(true); setError('');
        try { await onReview({ draft }); }
        catch (reason) { setError(reason.message); }
        finally { setSaving(false); }
    };
    const applyActions = dirty && <div id="deployment-apply" className="flex flex-wrap items-center justify-between gap-3">
            <p className="text-sm text-theme-muted">{dirty ? 'Changes need resource checks; previous performance evidence will not carry over.' : 'Reviewed configuration.'}</p>
            <div className="flex gap-2">
                {dirty && <Button variant="ghost" disabled={disabled || saving} onClick={() => { setIntent({ ...createBlueprint(), ...initialDraft }); setDirty(false); setParameters(null); setError(''); }}>Discard changes</Button>}
                <Button disabled={locked || !can('guide:plan:execute') || !can('configuration:artifact:render')} isLoading={saving} onClick={save}>{dirty ? 'Apply changes and check resources' : 'Recheck source configuration'}</Button>
            </div>
        </div>;
    const inspector = <aside id="deployment-optimization-settings" className={cn(mapView === '3d' && 'inference-inspector inference-glass')} aria-label="Optimization settings">
                {mapView === '3d' && <div className="inference-panel-heading"><span>{entity?.label || 'Optimization settings'}</span>{entity && <Button variant="ghost" size="icon" aria-label="Close optimization settings" onClick={() => { select(null); setParameters(null); }}><X size={14} /></Button>}</div>}
                <div className="space-y-4 p-4">
                    {mapView === '2d' && error && <p role="alert" className="text-sm">{error}</p>}
                    {!entity ? <p className="text-sm text-theme-muted">Select an optimization in the flow to see why it is enabled, change its parameters or turn it off. Supporting optimizations connect to the stage they affect, not additional request hops.</p> : <>
                        <p className="text-sm text-theme-muted">{entity.detail}</p>
                        {entity.kind === 'capability' && <>
                            <Badge tone={entity.configurationState === 'mixed' ? 'warning' : entity.enabled ? 'success' : 'neutral'}>{entity.configurationState === 'mixed' ? 'Partially configured' : entity.enabled ? 'Enabled' : 'Not enabled'}</Badge>
                            <p className="text-sm">{blocked[entity.id] || entity.sourceReason || draft.optimizationDecisions?.find(item => item.id === entity.id)?.reason || entity.reason}</p>
                            {!(parameters === 'node' && ['prefix', 'chunked'].includes(entity.id)) && <Button variant="secondary" disabled={locked || Boolean(blocked[entity.id]) || !capabilityTogglePatch(entity, draft)} onClick={() => activate(entity.id)}>{entity.enabled ? 'Disable' : 'Enable'} {entity.label}</Button>}
                        </>}
                        {parameters === 'node' && <fieldset disabled={locked} className="min-w-0 space-y-4">
                            {entity.id === 'cache' && draft.nativeCpuCacheGiB != null && <label className="block text-sm">CPU KV cache per replica (GiB)
                                <InferenceParameterInput key={draft.nativeCpuCacheGiB} leaf={{ key: 'nativeCpuCacheGiB', title: 'CPU KV cache per replica (GiB)', value: draft.nativeCpuCacheGiB, type: 'number' }} onCommit={raw => {
                                    const value = Number(raw);
                                    if (!Number.isFinite(value) || value <= 0 || value > 1024) { setError('CPU KV cache must be greater than zero and at most 1024 GiB per replica.'); return; }
                                    change({ nativeCpuCacheGiB: value });
                                }} />
                                <span className="mt-1 block text-xs text-theme-muted">Total host memory is checked across all serving replicas when you apply.</span>
                            </label>}
                            <InferenceSourceComposer key={entity.id} draft={draft} change={change} catalog={catalog} entity={entity} assetPaths={nodeParameterPaths(entity, draft)} loading={loading} error={error} controlsOnly={Boolean(draft.runtimeConfiguration)} />
                            {draft.runtimeConfiguration && nodeResourceKinds(entity)?.[0] !== 'values' && <InferenceDeploymentParameters key={entity.id} draft={draft} change={change} entity={entity} />}
                            {mapView === '3d' && <Button variant="ghost" onClick={() => setParameters(null)}>Finish editing</Button>}
                        </fieldset>}
                        {parameters !== 'node' && !entity.unavailable && canConfigure(entity) && <Button variant="secondary" disabled={locked} onClick={() => setParameters('node')}><Settings2 size={14} />Edit parameters</Button>}
                    </>}
                </div>
            </aside>;
    return <section aria-label="Source deployment configuration" className="space-y-3">
        {loading && <LoadingState label="Loading revision-owned Guides and hardware" />}
        {error && <p role="alert" className="text-sm">{error} {!catalog && <Button size="xs" variant="secondary" onClick={() => setReload(value => value + 1)}>Retry source catalog</Button>}</p>}
        {stale && <div role="alert" className="space-y-2 text-sm"><p>The cluster source revision changed. This saved configuration remains visible, but edits cannot be applied against a different revision.</p><Button variant="secondary" disabled={disabled || saving} onClick={() => {
            setIntent(current => ({ ...current, sourceCommit: catalog.commit, sourceUnits: [], finalConfiguration: undefined, sourceRouter: { ...current.sourceRouter, values: [] } }));
            setDirty(true);
        }}>Use current revision and discard source edits</Button></div>}
        {dirty && <p role="status" className="text-sm text-theme-muted">Unapplied changes. Deployment is blocked until this exact configuration passes resource checks.</p>}
        <div className={cn('inference-designer inference-workflow', mapView === '2d' && 'inference-workflow-2d')}>
            <div className="inference-view-switch inference-glass" role="group" aria-label="Configuration view">
                {['2d', '3d'].map(view => <Button key={view} size="xs" variant="ghost" aria-pressed={mapView === view} onClick={() => setMapView(view)}>{view === '2d' ? 'Serving flow' : '3D technical view'}</Button>)}
                <Badge size="xs">{dirty ? 'Draft topology' : 'Reviewed source topology'}</Badge>
            </div>
            {mapView === '3d' ? <InferenceFlowMap draft={display} clusterName={clusterName} selected={selected} onSelect={configure} visible={TIERS.map(tier => tier.id)} focus={focus} profile={profile} result={result} catalog={catalog} blockedReason={blocked[selected]} onActivate={activate} canConfigure={canConfigure} onConfigure={() => setParameters('node')} onTarget={() => setParameters('all')} />
                : <InferenceFlowMap2D graph={graph} selected={selected} onSelect={select} onActivate={activate} onConfigure={configure} canConfigure={canConfigure} blocked={blocked} intent={draft.deploymentIntent} hardware={result?.plan?.validation?.hardware} disabled={locked} onCloseInspector={() => setParameters(null)} inspectorFooter={parameters === 'node' ? applyActions : null} inspector={entity ? inspector : null} />}
            {mapView === '3d' && inspector}
        </div>
        {(mapView === '3d' || parameters !== 'node') && applyActions}
        <details><summary className="cursor-pointer text-sm text-theme-muted">Technical details: source Guides and all parameters</summary>
            <Panel className="mt-3" title="Advanced source configuration">
                <fieldset disabled={locked} className="space-y-3">
                    <div className="flex flex-wrap gap-2"><Button variant="secondary" onClick={() => setParameters('all')}><Settings2 size={14} />All deployment parameters</Button>
                        <Button variant="secondary" onClick={() => { try { change(recommendOptimizations(catalog, draft, profile), true); } catch (reason) { setError(reason.message); } }}>Select compatible optimizations</Button></div>
                    {draft.finalConfiguration && <Button variant="dangerOutline" onClick={() => change({ finalConfiguration: undefined })}>Discard final parameter edits</Button>}
                    <InferenceGuidePresets catalog={catalog} draft={draft} change={change} plan={plan} includedGuides={[...new Set([...(draft.selectedGuides || []), ...(plan?.guides || [])])]} onFocus={id => { const cap = graph.capabilities.find(item => item.guide === id); if (cap) configure(cap.id); }} />
                </fieldset>
            </Panel>
        </details>
        <Modal isOpen={parameters === 'all'} onClose={() => setParameters(null)} title="All deployment parameters" subtitle="Edits must be applied and checked before deployment." size="xl">
            <fieldset disabled={locked} className="space-y-4">
                {parameters === 'all' && <InferenceDeploymentParameters draft={draft} change={change} />}
            </fieldset>
        </Modal>
    </section>;
}
