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

import React, { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { X } from 'lucide-react';
import { cn } from '../lib/cn';
import { Button } from './Button';

const SIZES = {
    sm: 'max-w-md',
    md: 'max-w-lg',
    lg: 'max-w-2xl',
    xl: 'max-w-4xl',
};

// "drawer" panels use their own width scale (they're full-height, so a
// max-w meant for a centered dialog reads too narrow/too wide) -- roughly
// one size step up from the centered SIZES map.
const DRAWER_SIZES = {
    sm: 'max-w-lg',
    md: 'max-w-xl',
    lg: 'max-w-3xl',
    xl: 'max-w-5xl',
};

export function Modal({
    isOpen,
    onClose,
    title,
    subtitle,
    size = 'md',
    footer,
    closeOnBackdrop = true,
    closeOnEscape = true,
    className,
    inline = false,
    // 'center' (default) is the classic centered dialog; 'drawer' slides in
    // from the right edge of the screen as a full-height panel.
    variant = 'center',
    // Drawers normally render a full-screen dimmed backdrop.
    // Pass `overlay={false}` for a drawer that
    // should leave the rest of the page fully visible and interactive
    // (e.g. a live log panel the user keeps open while working elsewhere).
    overlay = true,
    children,
}) {
    const isDrawer = variant === 'drawer' && !inline;
    const showOverlay = !inline && (!isDrawer || overlay);
    const [entered, setEntered] = useState(false);

    useEffect(() => {
        if (!isOpen || !closeOnEscape) return undefined;
        const onKeyDown = (e) => {
            if (e.key === 'Escape') onClose?.();
        };
        window.addEventListener('keydown', onKeyDown);
        return () => window.removeEventListener('keydown', onKeyDown);
    }, [isOpen, closeOnEscape, onClose]);

    useEffect(() => {
        if (!isDrawer || !isOpen) return undefined;
        // Mount off-screen first, then flip a class on the next frame so
        // the transform transition actually animates instead of snapping in.
        const raf = window.requestAnimationFrame(() => setEntered(true));
        return () => {
            window.cancelAnimationFrame(raf);
            setEntered(false);
        };
    }, [isDrawer, isOpen]);

    if (!isOpen) return null;

    const content = (
        <div className={inline ? 'w-full' : 'fixed inset-0 z-[200] flex items-center justify-center p-4 pointer-events-none'}>
            {showOverlay && <div
                className="modal-overlay absolute inset-0 backdrop-blur-sm pointer-events-auto"
                onClick={closeOnBackdrop ? onClose : undefined}
                aria-hidden="true"
            />}
            <div
                role={inline ? undefined : 'dialog'}
                aria-modal={inline ? undefined : 'true'}
                aria-label={typeof title === 'string' ? title : undefined}
                className={cn(
                    'relative w-full bg-theme-card border border-theme-border shadow-2xl',
                    !inline && 'pointer-events-auto',
                    isDrawer
                        ? cn(
                            'fixed inset-y-0 right-0 flex h-full max-h-none flex-col rounded-none border-y-0 border-r-0 transition-transform duration-300 ease-out',
                            DRAWER_SIZES[size],
                            entered ? 'translate-x-0' : 'translate-x-full'
                        )
                        : cn(inline ? 'min-h-[calc(100vh-8rem)] rounded-2xl' : 'max-h-[85vh] flex flex-col rounded-2xl', inline ? 'max-w-none' : SIZES[size]),
                    className
                )}
            >
                {(title || onClose) && (
                    <div className="flex items-start justify-between gap-4 px-6 pt-5 pb-4 border-b border-theme-border shrink-0">
                        <div className="min-w-0">
                            {typeof title === 'string' ? (
                                <h3 className="text-base font-bold text-theme-text">{title}</h3>
                            ) : (
                                title
                            )}
                            {subtitle && <p className="text-xs text-theme-muted mt-1">{subtitle}</p>}
                        </div>
                        {onClose && (
                            <button
                                onClick={onClose}
                                aria-label="Close"
                                className="p-1.5 rounded-lg text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-200/60 dark:hover:bg-slate-800 transition-colors"
                            >
                                <X className="w-4 h-4" />
                            </button>
                        )}
                    </div>
                )}
                <div className={inline ? 'px-6 py-5' : 'flex-1 min-h-0 px-6 py-4 overflow-y-auto custom-scrollbar'}>{children}</div>
                {footer && (
                    <div className="flex items-center justify-end gap-2 px-6 py-4 border-t border-theme-border shrink-0">
                        {footer}
                    </div>
                )}
            </div>
        </div>
    );

    if (inline) return content;
    return createPortal(
        content,
        document.body
    );
}

/** A non-modal guide: the highlighted real controls remain operable. */
export function Walkthrough({ steps, step, onStepChange, onClose }) {
    const [position, setPosition] = useState(null);
    const card = useRef(null), previousFocus = useRef(null);
    const current = step == null ? null : steps[step];
    const targetSelector = current?.target;
    const open = Boolean(current);
    useEffect(() => {
        if (!open) return undefined;
        previousFocus.current = document.activeElement;
        return () => { if (previousFocus.current?.isConnected) previousFocus.current.focus(); };
    }, [open]);
    useEffect(() => {
        if (!targetSelector) return undefined;
        const target = document.querySelector(targetSelector);
        target?.scrollIntoView({ block: 'center', behavior: 'instant' });
        let frame;
        const measure = () => {
            const rect = target?.getBoundingClientRect();
            const w = window.innerWidth, h = window.innerHeight;
            const left = Math.max(0, Math.min(w, (rect?.left ?? 16) - 6));
            const right = Math.max(left, Math.min(w, (rect?.right ?? w - 16) + 6));
            const top = Math.max(0, Math.min(h, (rect?.top ?? 16) - 6));
            const bottom = Math.max(top, Math.min(h, (rect?.bottom ?? 16) + 6));
            const width = Math.min(340, w - 24), height = card.current?.offsetHeight || 230;
            const x = w - right >= width + 24 ? right + 12 : left >= width + 24 ? left - width - 12 : Math.min(left, w - width - 12);
            const y = bottom + height + 24 <= h ? bottom + 12 : top > height + 24 ? top - height - 12 : Math.max(12, h - height - 12);
            setPosition({ left, right, top, bottom, w, h, x: Math.max(12, x), y, width });
        };
        const update = () => { cancelAnimationFrame(frame); frame = requestAnimationFrame(measure); };
        measure();
        card.current?.focus();
        window.addEventListener('resize', update);
        window.addEventListener('scroll', update, true);
        const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(update);
        if (target) observer?.observe(target);
        if (card.current) observer?.observe(card.current);
        return () => { cancelAnimationFrame(frame); observer?.disconnect(); window.removeEventListener('resize', update); window.removeEventListener('scroll', update, true); };
    }, [targetSelector]);
    useEffect(() => {
        if (!open) return undefined;
        const keyboard = event => {
            if (event.key === 'Escape') { event.stopPropagation(); onClose(); return; }
            if (event.key !== 'Tab') return;
            const target = document.querySelector(targetSelector);
            const selectors = 'button:not(:disabled),a[href],input:not(:disabled),select:not(:disabled),textarea:not(:disabled),summary';
            const controls = [...(target?.querySelectorAll(selectors) || []), ...(card.current?.querySelectorAll(selectors) || [])].filter(el => el.getClientRects().length);
            if (!controls.length) return;
            if (!controls.includes(document.activeElement) || (event.shiftKey && document.activeElement === controls[0]) || (!event.shiftKey && document.activeElement === controls.at(-1))) {
                event.preventDefault(); (event.shiftKey ? controls.at(-1) : controls[0]).focus();
            }
        };
        window.addEventListener('keydown', keyboard, true);
        return () => window.removeEventListener('keydown', keyboard, true);
    }, [open, targetSelector, onClose]);
    if (!current || typeof document === 'undefined') return null;
    const p = position;
    return createPortal(<div className="pointer-events-none fixed inset-0 z-[210]" aria-label="Guided walkthrough">
        {p && <>
            {[
                { left: 0, top: 0, width: p.w, height: p.top },
                { left: 0, top: p.top, width: p.left, height: p.bottom - p.top },
                { left: p.right, top: p.top, width: p.w - p.right, height: p.bottom - p.top },
                { left: 0, top: p.bottom, width: p.w, height: p.h - p.bottom },
            ].map((style, index) => <div key={index} aria-hidden="true" className="pointer-events-auto absolute bg-black/60" style={style} />)}
            <div aria-hidden="true" className="absolute rounded-lg ring-2 ring-[var(--brand-accent)]" style={{ left: p.left, top: p.top, width: p.right - p.left, height: p.bottom - p.top }} />
        </>}
        <section ref={card} tabIndex={-1} role="dialog" aria-modal="false" aria-label={`Step ${step + 1}: ${current.title}`} className="pointer-events-auto absolute max-h-[calc(100dvh-24px)] overflow-auto rounded-xl border border-theme-border bg-theme-card p-4 text-theme-text shadow-2xl" style={{ left: p?.x ?? 12, top: p?.y ?? 12, width: p?.width ?? 320 }}>
            <p className="text-xs font-medium text-theme-muted">{step + 1} / {steps.length}</p><h3 className="mt-2 text-base font-semibold">{current.title}</h3><p className="mt-2 text-sm text-theme-muted">{current.description}</p>
            <div className="mt-4 flex items-center justify-between gap-2"><Button type="button" variant="ghost" onClick={onClose}>Skip</Button><div className="flex gap-2"><Button type="button" variant="secondary" disabled={step === 0} onClick={() => onStepChange(step - 1)}>Back</Button><Button type="button" onClick={() => step + 1 === steps.length ? onClose() : onStepChange(step + 1)}>{step + 1 === steps.length ? 'Finish' : 'Next'}</Button></div></div>
        </section>
    </div>, document.body);
}
