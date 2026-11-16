/** Enumerate the actual rendered resource fields; no stale engine flag schema. */
export function sourceParameterLeaves(documents) {
    const leaves = [];
    const visit = (value, path) => {
        if (value && typeof value === 'object') {
            for (const [key, child] of Object.entries(value)) {
                if (['__proto__', 'constructor', 'prototype'].includes(key)) throw new Error('Unsafe parameter key.');
                visit(child, [...path, Array.isArray(value) ? Number(key) : key]);
            }
        } else if (['string', 'number', 'boolean'].includes(typeof value)) leaves.push({ path, key: path.join('.'), value, type: typeof value });
    };
    visit(documents, []);
    return leaves;
}
export function sourceParameterEdit(leaf, raw) {
    const value = leaf.type === 'number' ? Number(raw) : leaf.type === 'boolean' ? raw === 'true' : raw;
    if (leaf.type === 'number' && (!raw.trim() || !Number.isFinite(value))) throw new Error('Enter a finite number.');
    return { path: leaf.path, value };
}

const same = (left, right) => JSON.stringify(left) === JSON.stringify(right);
const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);

function namedItems(values, path) {
    if (values.every(item => object(item) && (item.metadata?.name || item.name))) {
        const entries = values.map(item => [item.metadata?.name
            ? [item.apiVersion, item.kind, item.metadata.namespace, item.metadata.name].join('/')
            : item.name, item]);
        return new Set(entries.map(([key]) => key)).size === entries.length ? entries : null;
    }
    if (['args', 'command'].includes(path.at(-1)) && values.every(item => typeof item === 'string')) {
        const entries = [];
        for (let index = 0; index < values.length; index += 1) {
            const token = values[index];
            const key = token.match(/^(--[\w-]+)(?:=|$)/)?.[1];
            const tokens = [token];
            if (key && !token.includes('=') && index + 1 < values.length && !values[index + 1].startsWith('--')) tokens.push(values[++index]);
            entries.push([key || `position:${index}`, tokens]);
        }
        return new Set(entries.map(([key]) => key)).size === entries.length ? entries : null;
    }
    return null;
}

function mergeParameters(base, edited, next, path = []) {
    if (same(base, edited)) return next;
    if (same(base, next) || same(edited, next)) return edited;
    if ([base, edited, next].every(object)) {
        const result = {};
        for (const key of new Set([...Object.keys(next), ...Object.keys(edited), ...Object.keys(base)])) {
            const value = mergeParameters(base[key], edited[key], next[key], [...path, key]);
            if (value !== undefined) result[key] = value;
        }
        return result;
    }
    if ([base, edited, next].every(Array.isArray)) {
        const groups = [base, edited, next].map(items => namedItems(items, path));
        if (groups.every(Boolean)) {
            const [before, changes, generated] = groups.map(items => new Map(items));
            const result = [];
            for (const key of new Set([...generated.keys(), ...changes.keys(), ...before.keys()])) {
                // Argument groups are atomic: a changed connector must not be partially merged.
                const value = ['args', 'command'].includes(path.at(-1))
                    ? same(before.get(key), changes.get(key)) ? generated.get(key)
                        : same(before.get(key), generated.get(key)) || same(changes.get(key), generated.get(key)) ? changes.get(key)
                            : null
                    : mergeParameters(before.get(key), changes.get(key), generated.get(key), [...path, key]);
                if (value === null) throw new Error(`Parameter edit conflicts with the changed optimization at ${[...path, key].join('.')}. Your edits are preserved; undo the conflicting change before applying.`);
                if (value !== undefined) result.push(value);
            }
            return ['args', 'command'].includes(path.at(-1)) ? result.flat() : result;
        }
        if (base.length === edited.length && base.length === next.length) {
            return base.map((value, index) => mergeParameters(value, edited[index], next[index], [...path, index]));
        }
    }
    throw new Error(`Parameter edit conflicts with the changed optimization at ${path.join('.') || 'deployment'}. Your edits are preserved; undo the conflicting change before applying.`);
}

/** Rebase explicit edits only; incompatible changes fail instead of discarding user values. */
export function rebaseFinalConfiguration(saved, result) {
    const baseContent = result.files.find(file => file.name === 'deployment.yaml')?.content;
    if (!baseContent) throw new Error('The rendered deployment is missing deployment.yaml.');
    if (!saved) return { baseChecksum: result.compositionChecksum, baseContent, content: baseContent };
    if (saved.baseChecksum === result.compositionChecksum) return { ...saved, baseContent: saved.baseContent || baseContent };
    if (!saved.baseContent) throw new Error('This saved configuration has no edit baseline. Restore its original optimization selection or explicitly discard final parameter edits before changing it.');
    const documents = [saved.baseContent, saved.content, baseContent].map(content => yaml.loadAll(content).filter(Boolean));
    for (const value of documents) sourceParameterLeaves(value);
    const merged = mergeParameters(...documents);
    return { baseChecksum: result.compositionChecksum, baseContent, content: merged.map(doc => yaml.dump(doc, { noRefs: true, lineWidth: -1 })).join('---\n') };
}
import yaml from 'js-yaml';
