import { useEffect, useRef, useState } from 'react';
import { Badge, Button, Panel, Input, Select, Checkbox, Textarea } from '../../../shared/ui';
import { requestJson } from '../../../shared/api/httpClient';

export function MeasuredOptimizationControls({ run, onUpdate, onReview, disabled, onBusyChange }) {
    const [selected, setSelected] = useState(() => run.optimization?.request?.candidate_ids || []);
    const [options, setOptions] = useState(() => {
        const saved = run.optimization?.request;
        const planned = run.request?.requirements;
        const capacity = run.planning_evidence?.find(item => item.source === 'cluster-overview')?.details?.selected_gpu_count;
        return { max_gpu_count: planned?.hard_constraints?.maxAccelerators ?? capacity ?? '', max_duration_minutes: 60, max_benchmark_runs: 4, repetitions: 2, objective: 'throughput', max_error_rate: 0, native_cpu_cache_gib: '', ...saved,
            isl: saved?.benchmark?.matrix?.[0]?.isl ?? planned?.workload?.inputTokens ?? 512,
            osl: saved?.benchmark?.matrix?.[0]?.osl ?? planned?.workload?.outputTokens ?? 128,
            concurrency: saved?.benchmark?.concurrency_stages?.[0]?.concurrency ?? planned?.workload?.concurrency ?? 4,
            requests: saved?.benchmark?.concurrency_stages?.[0]?.num_requests ?? 100 };
    });
    const [workload, setWorkload] = useState(run.optimization?.request?.benchmark?.workload_yaml || ''), [error, setError] = useState(''), [busy, setBusy] = useState(false);
    const mounted = useRef(true);
    useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
    useEffect(() => { onBusyChange?.(busy); return () => onBusyChange?.(false); }, [busy, onBusyChange]);
    const active = run.status === 'optimizing';
    const state = run.optimization;
    const variantCount = options.native_cpu_cache_gib ? (run.candidates || []).filter(candidate => selected.includes(candidate.id) && !candidate.native_cpu_cache_gib && !selected.includes(`${candidate.id}-cpu-${Number(options.native_cpu_cache_gib)}`) && ['optimized-baseline', 'precise-prefix-cache-routing', 'pd-disaggregation'].includes(candidate.provider_ref)).length : 0;
    const requiredRuns = (selected.length + variantCount) * options.repetitions;
    const cleanupRequired = ['interrupted', 'cleanup_failed'].includes(state?.status);
    const locked = active || disabled || busy || cleanupRequired || run.status !== 'awaiting_approval';
    const overCapacity = run.candidates?.some(candidate => selected.includes(candidate.id) && candidate.required_gpus > Number(options.max_gpu_count));
    const send = async (cancel = false) => {
        setBusy(true); setError('');
        try {
            const { isl, osl, concurrency, requests, ...budget } = options;
            delete budget.benchmark; delete budget.candidate_ids;
            budget.native_cpu_cache_gib = budget.native_cpu_cache_gib ? Number(budget.native_cpu_cache_gib) : null;
            const benchmark = workload.trim() ? { workload_yaml: workload } : { matrix: [{ isl, osl }], concurrency_stages: [{ concurrency, num_requests: requests }] };
            let cleanupQuery = '';
            if (cancel && ['interrupted', 'cleanup_failed'].includes(state?.status) && state.cluster_id) {
                const session = await requestJson(`/api/cluster/session?clusterId=${encodeURIComponent(state.cluster_id)}`);
                cleanupQuery = `?cluster_session_id=${encodeURIComponent(session.sessionId || session.session_id)}`;
            }
            const result = await requestJson(`/api/agentic-deployments/${encodeURIComponent(run.id)}/optimize${cancel ? '/cancel' : ''}${cleanupQuery}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: cancel ? undefined : JSON.stringify({ ...budget, candidate_ids: selected, benchmark }) });
            if (mounted.current) onUpdate(result);
        } catch (failure) { if (mounted.current) setError(failure.message); }
        finally { if (mounted.current) setBusy(false); }
    };
    const number = (name, label, min, max, step = 1) => <label className="text-sm">{label}<Input type="number" required={name !== 'native_cpu_cache_gib'} min={min} max={max} step={step} value={options[name] ?? ''} onChange={event => setOptions(value => ({ ...value, [name]: event.target.value === '' ? '' : Number(event.target.value) }))} /></label>;
    return <Panel title="Verify with real benchmarks">
        <p className="text-sm text-theme-muted mb-3">Deploy selected candidates one at a time, repeat the same workload, then remove the test deployments. Only complete measurements meeting your SLOs and error limit qualify. The winner is the best tested candidate, not a global optimum.</p>
        <form onInvalid={event => event.target.closest('details')?.setAttribute('open', '')} onSubmit={event => { event.preventDefault(); if (!locked && selected.length && !overCapacity && requiredRuns <= options.max_benchmark_runs) send(); }}><fieldset disabled={locked} className="space-y-3">
            <div>{run.candidates?.filter(c => c.deployable).map(candidate => <Checkbox key={candidate.id} label={`${candidate.provider_ref} · ${candidate.required_gpus} accelerators · ${candidate.id}`} checked={selected.includes(candidate.id)} onChange={event => setSelected(ids => event.target.checked ? [...ids, candidate.id] : ids.filter(id => id !== candidate.id))} />)}</div>
            <div className="grid gap-3 md:grid-cols-3">{number('max_gpu_count', 'Maximum accelerators', 1, 1024)}{number('max_duration_minutes', 'Maximum minutes', 1, 180)}{number('max_benchmark_runs', 'Maximum benchmark runs', 1, 30)}{number('repetitions', 'Repetitions per candidate', 1, 3)}{number('native_cpu_cache_gib', 'Optional CPU cache composition (GiB/replica)', 0.001, 1024, 0.001)}{number('max_error_rate', 'Maximum error fraction (0–1)', 0, 1, 0.001)}<label className="text-sm">Objective<Select value={options.objective} onChange={event => setOptions(value => ({ ...value, objective: event.target.value }))}><option value="throughput">Highest token throughput</option><option value="latency">Lowest P95 end-to-end latency</option><option value="throughput_per_gpu">Highest throughput per GPU</option></Select></label></div>
            <details><summary>Test workload</summary><fieldset disabled={Boolean(workload.trim())} className="grid gap-3 md:grid-cols-2 mt-2">{number('isl', 'Input tokens', 1, 1000000)}{number('osl', 'Output tokens', 1, 1000000)}{number('concurrency', 'Concurrent requests', 1, 4096)}{number('requests', 'Requests per repetition', 1, 100000)}</fieldset><label className="block text-sm mt-3">Optional inference-perf workload YAML (replaces synthetic workload)<Textarea className="mt-1 min-h-24" value={workload} onChange={event => setWorkload(event.target.value)} /></label></details>
            <p className="text-xs text-theme-muted">Selected: {selected.length} candidates + {variantCount} composition variants, each × {options.repetitions} repetitions = {requiredRuns} runs. CPU cache adds one trial variant per compatible selected base; include those repetitions in the run budget. Cleanup may continue after the time budget expires.</p>
            {overCapacity && <p role="alert">A selected candidate exceeds the accelerator ceiling.</p>}
            {requiredRuns > options.max_benchmark_runs && <p role="alert">Increase the benchmark budget or reduce candidates, composition variants or repetitions.</p>}
            <Button type="submit" isLoading={busy} disabled={!selected.length || overCapacity || requiredRuns > options.max_benchmark_runs}>Start deployment and benchmark tests</Button>
        </fieldset></form>
        {active && <div className="flex gap-3 items-center mt-3"><span role="status">{state?.phase || 'Starting'} · {state?.candidate_id} · repetition {state?.repetition || 0}</span><Button variant="secondary" disabled={disabled || busy} onClick={() => send(true)} isLoading={busy}>Stop and clean up</Button></div>}
        {cleanupRequired && <Button variant="secondary" disabled={disabled || busy} onClick={() => send(true)} isLoading={busy}>Clean up interrupted tests</Button>}
        {state && <div className="mt-4 flex flex-wrap gap-2" aria-label="Trial progress">{['deploying', 'benchmarking', 'cleaning', 'finished'].map(phase => <Badge key={phase} tone={state.phase === phase ? 'info' : 'neutral'}>{phase}{state.phase === phase ? ' (current)' : ''}</Badge>)}</div>}
        {state && <div className="mt-3 space-y-2"><p>Status: {state.status} · trial records: {state.samples?.length || 0}</p>{state.error && <p role="alert">{state.error}</p>}{state.ranking?.map((item, i) => <div key={item.candidate_id} className="flex items-center gap-3"><span>{i + 1}. {item.candidate_id} · {item.metric}: {item.score.toFixed(2)}</span><Button size="sm" variant="secondary" disabled={locked} onClick={() => onReview(item.candidate_id)}>Review measured candidate</Button></div>)}<details><summary>Trial evidence and failures</summary>{state.samples?.map((sample, index) => <div key={index} className="text-sm border-b border-theme-border py-2"><p>{sample.candidate_id} · {sample.status}</p>{sample.benchmark_id && <p className="text-xs break-all">Benchmark: {sample.benchmark_id}</p>}{sample.error && <p>{sample.error}</p>}{[['TTFT P95 (ms)', 'ttft_p95_ms'], ['TPOT P95 (ms)', 'tpot_p95_ms'], ['E2E P95 (ms)', 'e2e_p95_ms'], ['Throughput (tokens/s)', 'throughput_tokens_per_s'], ['Error fraction', 'error_rate']].map(([label, key]) => <p key={key}>{label}: {Number.isFinite(sample.metrics?.[key]) ? sample.metrics[key].toFixed(3) : 'Unknown'}</p>)}</div>)}</details>{!active && !state.ranking?.length && <p>No candidate has complete measurements satisfying the constraints.</p>}</div>}
        {error && <p role="alert" className="mt-2">{error}</p>}
    </Panel>;
}
