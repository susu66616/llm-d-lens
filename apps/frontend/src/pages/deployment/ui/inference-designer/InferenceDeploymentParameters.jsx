import yaml from 'js-yaml';
import { useEffect, useRef, useState } from 'react';
import { Button, Input, Textarea, LoadingState } from '../../../../shared/ui/index';
import { generateSourceBundle } from './sourceCompositionBackend';
import { sourceParameterLeaves, sourceParameterEdit, rebaseFinalConfiguration } from '../../model/inference-designer/sourceParameters';
import { InferenceParameterField } from './InferenceParameterField';
import { InferenceParameterInput } from './InferenceParameterInput';
import { nodeSourceFields } from '../../model/inference-designer/nodeParameters';

/** Edit the actual rendered deployment, tied to the current selection checksum. */
export function InferenceDeploymentParameters({ draft, change, entity }) {
    const [snapshot, setSnapshot] = useState(null), [busy, setBusy] = useState(true), [error, setError] = useState('');
    const [query, setQuery] = useState(''), [tab, setTab] = useState('parameters'), [reload, setReload] = useState(0);
    const latest = useRef({ draft, change }); useEffect(() => { latest.current = { draft, change }; });
    const selection = JSON.stringify({ ...draft, finalConfiguration: undefined });
    useEffect(() => {
        let active = true;
        Promise.resolve().then(() => {
            if (!active) return null;
            setBusy(true); setError(''); setSnapshot(null);
            return generateSourceBundle({ ...latest.current.draft, finalConfiguration: undefined, sourceValidate: false });
        }).then(result => {
            if (!active || !result) return;
            const saved = latest.current.draft.finalConfiguration;
            setSnapshot(rebaseFinalConfiguration(saved, result));
        }).catch(reason => { if (active) setError(reason.message); }).finally(() => { if (active) setBusy(false); });
        return () => { active = false; };
    }, [selection, reload]);
    const update = content => {
        const next = { ...snapshot, content }; setSnapshot(next);
        latest.current.change({ finalConfiguration: next });
    };
    let documents = [], fields = [], parseError = '';
    try { if (snapshot) { documents = yaml.loadAll(snapshot.content).filter(Boolean); fields = entity ? nodeSourceFields(documents, entity).filter(field => !field.readOnly) : sourceParameterLeaves(documents); } } catch (reason) { parseError = reason.message; }
    return <section className="space-y-4" aria-label="Final deployment configuration">
        <p className="text-sm text-theme-muted">{entity ? 'Parameters for this optimization or serving stage.' : 'Final configuration for the selected optimizations.'} Unrelated edits are retained when optimizations change; conflicting edits must be resolved before applying.</p>
        {busy && <LoadingState label="Rendering selected deployment" />}
        {error && <p role="alert">{error} <Button size="xs" variant="secondary" onClick={() => setReload(value => value + 1)}>Reload configuration</Button></p>}
        {snapshot && <>{!entity && <div role="tablist" aria-label="Final configuration view">{['parameters', 'yaml'].map(value => <Button key={value} role="tab" aria-selected={tab === value} variant="ghost" onClick={() => setTab(value)}>{value === 'yaml' ? 'Final YAML' : 'All parameters'}</Button>)}</div>}
        {parseError && <p role="alert">{parseError}</p>}
        {tab === 'yaml' && !entity ? <Textarea aria-label="Final deployment YAML" rows={24} value={snapshot.content} onChange={event => update(event.target.value)} /> : <>{!entity && <Input aria-label="Search deployment parameters" placeholder="Find a parameter or resource" value={query} onChange={event => setQuery(event.target.value)} />}
        {entity && !fields.length && <p className="text-sm text-theme-muted">No editable parameters are declared for this selection. Enable it and resolve its prerequisites first, or inspect Technical details.</p>}
        {fields.filter(field => `${documents[field.path[0]]?.kind} ${documents[field.path[0]]?.metadata?.name} ${field.key}`.toLowerCase().includes(query.toLowerCase())).map(field => <InferenceParameterField key={field.key} field={{ ...field, title: field.title || field.path.at(-1), section: field.section || `${documents[field.path[0]]?.kind} / ${documents[field.path[0]]?.metadata?.name}`, hint: field.hint || field.path.slice(1).join(' › ') }}><InferenceParameterInput key={JSON.stringify([field.key, field.value])} leaf={field} onCommit={raw => {
            try { const edit = sourceParameterEdit(field, field.encode ? field.encode(raw) : raw); const next = structuredClone(documents); let owner = next; for (const key of edit.path.slice(0, -1)) owner = owner[key]; owner[edit.path.at(-1)] = edit.value; update(next.map(doc => yaml.dump(doc, { noRefs: true, lineWidth: -1 })).join('---\n')); setError(''); } catch (reason) { setError(reason.message); }
        }} /></InferenceParameterField>)}</>}</>}
    </section>;
}
