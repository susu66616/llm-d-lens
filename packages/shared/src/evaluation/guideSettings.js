export function normalizeGuideSettings(guide, _variant, settings = {}) {
    if (!settings || typeof settings !== 'object' || Array.isArray(settings)) throw new Error('Guide settings must be an object.');
    const supported = ['cacheCpuGiB', 'rdmaNicCount', 'slices', 'cacheLruCapacity', 'cacheStorageGiB', 'cacheStorageClass', 'routerValues'];
    if (Object.keys(settings).some(key => !supported.includes(key))) throw new Error('Unknown Guide setting.');
    const result = {};
    for (const field of ['cacheCpuGiB', 'rdmaNicCount', 'slices', 'cacheLruCapacity', 'cacheStorageGiB']) {
        const raw = settings[field];
        if (raw == null || String(raw).trim() === '') continue;
        const value = Number(raw);
        if (typeof raw === 'boolean' || !Number.isFinite(value) || value <= 0 || (['rdmaNicCount', 'slices', 'cacheLruCapacity'].includes(field) && !Number.isSafeInteger(value))) throw new Error(`${field} must be a positive ${['rdmaNicCount', 'slices', 'cacheLruCapacity'].includes(field) ? 'integer' : 'number'}.`);
        result[field] = value;
    }
    if (result.cacheLruCapacity != null && guide !== 'tiered-prefix-cache') throw new Error('CPU index capacity requires tiered caching.');
    if (result.slices != null && guide !== 'pd-disaggregation') throw new Error('Slices require a P/D Guide.');
    if (result.cacheCpuGiB != null && guide !== 'tiered-prefix-cache') throw new Error('CPU cache capacity requires a supported CPU offload variant.');
    if (result.rdmaNicCount != null && guide !== 'pd-disaggregation') throw new Error('NIC count requires the P/D RDMA variant.');
    if (result.cacheStorageGiB != null && guide !== 'tiered-prefix-cache') throw new Error('Filesystem cache storage requires a tiered Guide.');
    if (settings.cacheStorageClass != null && settings.cacheStorageClass !== '') {
        if (typeof settings.cacheStorageClass !== 'string' || !/^[a-z0-9][a-z0-9.-]*$/.test(settings.cacheStorageClass) || guide !== 'tiered-prefix-cache' || !result.cacheStorageGiB) throw new Error('Cache StorageClass requires filesystem cache capacity and a valid class name.');
        result.cacheStorageClass = settings.cacheStorageClass;
    }
    if (settings.routerValues != null && typeof settings.routerValues !== 'string') throw new Error('Router values must be YAML text.');
    if (settings.routerValues?.trim()) result.routerValues = settings.routerValues.trim();
    return result;
}