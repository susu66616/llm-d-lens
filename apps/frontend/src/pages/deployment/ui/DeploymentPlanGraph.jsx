import { useId, useState } from 'react';
import { layoutPlan } from '../model/deploymentPlan';
import { Button, Input } from '../../../shared/ui';
import { CHART_SERIES } from '../../../shared/ui/charts/palette';

export function DeploymentPlanGraph({ plan, selected, onSelect }) {
    const [zoom, setZoom] = useState(1), [query, setQuery] = useState(''), [edgeIndex, setEdgeIndex] = useState(null);
    const edge = plan.edges[edgeIndex];
    const marker = useId().replaceAll(':', '');
    const nodes = layoutPlan(plan), height = Math.max(250, ...nodes.map(n => n.y + 80));
    const byId = new Map(nodes.map(n => [n.id, n]));
    const legend = [
        ['Routing / model', 0],
        ['Runtime components', 1],
        ['Supporting resources', 2],
    ];
    return <div className="rounded-lg border border-theme-border bg-theme-bg">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-theme-border px-4 py-3">
            <div><p className="text-xs font-semibold text-theme-text">{plan.content ? 'Deployment resources' : 'Planned architecture'}</p><p className="mt-0.5 text-[11px] text-theme-muted">{plan.content ? 'Select a component to inspect or edit it. Arrows follow declared resource references.' : 'Select a workload to adjust replicas and parallelism. Generate YAML to inspect the full resource graph.'}</p></div>
            <div className="flex flex-wrap gap-2" aria-label="Topology legend">{legend.map(([label, tier]) => <span key={label} className="inline-flex items-center gap-1.5 text-[10px] text-theme-muted"><span className="h-2 w-2 rounded-full" style={{ backgroundColor: CHART_SERIES[tier] }} />{label}</span>)}</div>
        </div>
        <div className="flex flex-wrap items-center gap-2 border-b border-theme-border p-3">
            <Input type="search" aria-label="Find topology component" placeholder="Find a component" value={query} onChange={event => setQuery(event.target.value)} />
            <Button size="xs" variant="secondary" aria-label="Zoom out" disabled={zoom <= 0.5} onClick={() => setZoom(value => value - 0.25)}>−</Button>
            <span className="text-xs text-theme-muted">{Math.round(zoom * 100)}%</span>
            <Button size="xs" variant="secondary" aria-label="Zoom in" disabled={zoom >= 2} onClick={() => setZoom(value => value + 0.25)}>+</Button>
            <Button size="xs" variant="ghost" onClick={() => { setZoom(1); setQuery(''); setEdgeIndex(null); }}>Reset view</Button>
        </div>
        {query && <div className="flex flex-wrap gap-2 p-3" aria-label="Matching components">{nodes.filter(node => `${node.name} ${node.kind}`.toLowerCase().includes(query.toLowerCase())).map(node => <Button key={node.id} size="xs" variant="secondary" onClick={() => onSelect(node.id)}>{node.name}</Button>)}{!nodes.some(node => `${node.name} ${node.kind}`.toLowerCase().includes(query.toLowerCase())) && <p className="text-xs text-theme-muted">No matching components.</p>}</div>}
        <div className="overflow-auto"><svg role="group" aria-label="Deployment resource relationships" viewBox={`0 0 940 ${height}`} style={{ minWidth: 700 * zoom, width: `${100 * zoom}%` }}>
        <defs><marker id={marker} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" /></marker></defs>
        {plan.edges.map((edge, i) => { const a = byId.get(edge.from), b = byId.get(edge.to); if (!a || !b) return null; const direction = Math.sign(b.x - a.x) || 1, ax = a.x + direction * 115, bx = b.x - direction * 115; return <g key={i} className="text-theme-muted" role="button" tabIndex={0} aria-label={`${a.name} ${edge.label} ${b.name}`} aria-pressed={edgeIndex === i} onClick={() => setEdgeIndex(i)} onKeyDown={event => { if (['Enter', ' '].includes(event.key)) { event.preventDefault(); setEdgeIndex(i); } }}><path d={`M ${ax} ${a.y} C ${(ax + bx) / 2} ${a.y}, ${(ax + bx) / 2} ${b.y}, ${bx} ${b.y}`} fill="none" stroke="currentColor" markerEnd={`url(#${marker})`} strokeWidth={edgeIndex === i ? 3 : 1.5} opacity={edgeIndex === i ? 1 : 0.5} /><text x={(a.x + b.x) / 2} y={(a.y + b.y) / 2 - 12} textAnchor="middle" fontSize="11" fill="currentColor">{edge.label}</text></g>; })}
        {nodes.map(node => <g key={node.id} role="button" tabIndex={0} aria-label={`${node.kind}: ${node.name}`} aria-pressed={selected === node.id} onClick={() => onSelect(node.id)} onKeyDown={event => { if (['Enter', ' '].includes(event.key)) { event.preventDefault(); onSelect(node.id); } }} style={{ cursor: 'pointer' }}>
            <rect x={node.x - 115} y={node.y - 30} width="230" height="70" rx="8" fill="var(--bg-card)" stroke={CHART_SERIES[node.tier]} strokeWidth={selected === node.id ? 3 : 1} />
            <text x={node.x} y={node.y - 9} textAnchor="middle" fontSize="11" fill={CHART_SERIES[node.tier]}>{node.kind}</text><text x={node.x} y={node.y + 12} textAnchor="middle" fontSize="12" fill="var(--text-primary)">{node.name.length > 28 ? `${node.name.slice(0, 25)}…` : node.name}</text><title>{`${node.name}: ${node.detail}`}</title>
        </g>)}
        </svg></div>
        {edge && <aside className="border-t border-theme-border p-3 text-sm" aria-label="Relationship details"><strong>{byId.get(edge.from)?.name} → {byId.get(edge.to)?.name}</strong><p className="mt-1 text-theme-muted">{edge.label}. This is a declared or planned relationship; no live traffic measurement is attached.</p><Button size="xs" variant="link" onClick={() => onSelect(edge.to)}>Inspect destination</Button></aside>}
        <div className="flex flex-wrap gap-x-4 gap-y-1 border-t border-theme-border px-4 py-2 text-[10px] text-theme-muted"><span><strong className="text-theme-text">solid arrow</strong> declared resource relationship</span><span><strong className="text-theme-text">highlighted border</strong> selected component</span></div>
    </div>;
}
