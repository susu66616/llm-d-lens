import { useState } from 'react';
import { Badge, Button, Modal } from '../../../../shared/ui';
import { DeploymentPlanGraph } from '../DeploymentPlanGraph';
import { OPTIMIZATION_STATES } from '../../model/deploymentPlan';
import { servingFlowDiagram } from '../../model/inference-designer/sceneGraph';

/** The diagram owns progressive disclosure and entry into the source parameter editor. */
export function InferenceFlowMap2D({ graph, selected, onSelect, onActivate, onConfigure, canConfigure, blocked = {}, intent, hardware, disabled = false, inspector, inspectorFooter, onCloseInspector }) {
    const [expandedGroup, setExpandedGroup] = useState(null), [pages, setPages] = useState({}), [inspecting, setInspecting] = useState(false);
    const diagram = servingFlowDiagram(graph, blocked, canConfigure, { expandedGroup, pages });
    const capability = diagram.groups.flatMap(group => group.capabilities).find(cap => cap.id === selected);
    const entity = capability || graph.nodes.find(node => node.id === selected);
    const targets = diagram.edges.filter(edge => edge.hidden && edge.from === selected).map(edge => diagram.nodes.find(node => node.id === edge.to)?.name);
    const close = () => { setInspecting(false); onCloseInspector?.(); };
    const inspectorContent = typeof inspector === 'function' ? inspector({ close }) : inspector;
    const footerContent = typeof inspectorFooter === 'function' ? inspectorFooter({ close }) : inspectorFooter;
    const inspect = id => {
        const cap = diagram.groups.flatMap(group => group.capabilities).find(item => item.id === id);
        const stage = graph.nodes.find(node => node.id === id);
        onSelect?.(id);
        if (!cap && stage && !disabled && canConfigure?.(stage)) onConfigure?.(id);
        if (cap && !disabled && !cap.editDisabled && cap.status !== 'unsupported') onConfigure?.(id);
        setInspecting(true);
    };
    return <section id="deployment-serving-flow" className="inference-map inference-map-2d" aria-label="Model serving flow">
        <header className="flex flex-wrap items-center justify-between gap-2">
            <div><p className="mb-1 text-xs font-semibold uppercase tracking-wider text-theme-muted">Serving flow</p><h3 className="text-lg font-semibold">Explore and configure your plan</h3></div>
            <Badge tone="neutral">{diagram.configuredCount} configured · {diagram.totalCount} settings</Badge>
        </header>
        <p className="my-2 text-xs text-theme-muted">Expand a group → select an optimization → edit its configuration.</p>
        <div id="deployment-serving-stages"><DeploymentPlanGraph plan={diagram} selected={selected} onSelect={inspect}
            onToggleGroup={id => { setExpandedGroup(current => current === id ? null : id); onSelect?.(id); }}
            onGroupPage={(id, page) => setPages(current => ({ ...current, [id]: page }))} disabled={disabled} /></div>
        <p className="mt-2 text-xs text-theme-muted">{diagram.serving} · Logical processing and configuration relationships, not live traffic.</p>
        <Modal isOpen={inspecting && Boolean(entity)} variant="drawer" size="md" title={entity?.label} onClose={close}
            subtitle={targets.length ? `Applies to ${targets.join(' + ')}` : undefined} footer={footerContent || <Button variant="secondary" onClick={close}>Back to flow</Button>}>
            {inspectorContent || (entity && <div className="space-y-4">
                {capability && <Badge tone={OPTIMIZATION_STATES[capability.status]?.tone || 'neutral'}>{OPTIMIZATION_STATES[capability.status]?.label || capability.status}</Badge>}
                <p className="text-sm">{blocked[selected] || entity.reason || entity.detail}</p>
                <div className="flex gap-2"><Button variant="secondary" disabled={disabled || entity.editDisabled || entity.status === 'unsupported'} onClick={() => onConfigure?.(selected)}>Edit parameters</Button>
                    {capability && <Button variant="secondary" disabled={disabled || capability.toggleDisabled || capability.status === 'unsupported'} onClick={() => onActivate?.(selected)}>{capability.enabled ? 'Disable' : 'Enable'} {capability.label}</Button>}</div>
            </div>)}
        </Modal>
        {(intent || hardware) && <details className="mt-2"><summary className="cursor-pointer text-xs text-theme-muted">Planning inputs and checks</summary><p className="mt-2 text-xs text-theme-muted">{intent && `Objective: ${intent.objective}. Device limit: ${intent.maxDevices || 'discovered capacity'}. `}{hardware ? 'Device counts checked; placement and performance unverified.' : 'Hardware checks pending generation.'}</p></details>}
    </section>;
}
