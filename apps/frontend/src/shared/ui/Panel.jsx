// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import React from 'react';
import { cn } from '../lib/cn';
import { Button } from './Button';

const PADDING = {
    none: 'p-0',
    sm: 'p-4',
    md: 'p-6',
};

const TONES = {
    neutral: '',
    subtle: 'bg-theme-card/40 bg-linear-to-br from-slate-100/70 to-transparent dark:from-slate-900/50',
    soft: 'border-cyan-500/15 bg-linear-to-br from-slate-100/80 via-slate-50/70 to-cyan-50/15 dark:from-slate-900/80 dark:via-slate-950/70 dark:to-cyan-950/15',
    inset: 'bg-linear-to-br from-slate-50/70 to-cyan-50/10 dark:from-slate-950/70 dark:to-cyan-950/10',
};

// The standard card surface. Replaces hand-rolled
// "bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700" shells.
export function Panel({ title, actions, padding = 'md', tone = 'neutral', className, children, ...props }) {
    return (
        <div
            className={cn(
                'bg-theme-card border border-theme-border rounded-xl shadow-lg transition-colors',
                TONES[tone],
                PADDING[padding],
                className
            )}
            {...props}
        >
            {(title || actions) && (
                <div className={cn('flex items-center justify-between gap-4 mb-4', padding === 'none' && 'p-4 mb-0 border-b border-theme-border')}>
                    {title && (
                        <h3 className="text-sm font-bold text-theme-text">{title}</h3>
                    )}
                    {actions && <div className="flex items-center gap-2">{actions}</div>}
                </div>
            )}
            {children}
        </div>
    );
}

export function WorkspaceHeader({ title, subtitle, eyebrow, badge, actions, backAction }) {
    return <header className="space-y-4">
        {backAction}
        <div className="flex flex-wrap items-end justify-between gap-4">
            <div className="min-w-0"><div className="mb-2 flex items-center gap-2"><span className="text-xs font-semibold uppercase tracking-wider text-theme-muted">{eyebrow}</span>{badge}</div><h2 className="break-words text-2xl font-semibold tracking-tight text-theme-text">{title}</h2>{subtitle && <p className="mt-2 text-sm text-theme-muted">{subtitle}</p>}</div>
            <div className="flex flex-wrap items-center gap-2">{actions}</div>
        </div>
    </header>;
}

/** Compact context row; interactive facts open their existing editor or review. */
export function SummaryStrip({ items, ...props }) {
    return <div className="grid grid-cols-2 overflow-hidden rounded-xl border border-theme-border bg-theme-card md:grid-cols-4" {...props}>
        {items.map(item => <div key={item.label} className="min-w-0 border-b border-r border-theme-border p-4 last:border-r-0 md:border-b-0">
            <div className="text-xs text-theme-muted">{item.label}</div>
            {item.onClick ? <Button variant="link" className="mt-1 h-auto max-w-full justify-start whitespace-normal p-0 text-left text-sm font-semibold" onClick={item.onClick} disabled={item.disabled}>{item.value}</Button> : <div className="mt-1 break-words text-sm font-semibold">{item.value}</div>}
            {item.detail && <div className="mt-1 text-xs text-theme-muted">{item.detail}</div>}
        </div>)}
    </div>;
}
