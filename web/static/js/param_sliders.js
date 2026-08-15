/**
 * param_sliders.js — Shared component to render and bind categorized parameter sliders.
 * Derives bounds, steps, defaults, and categories dynamically from /models/api/param_bounds.
 */

let cachedParamBounds = null;

async function getParamBounds() {
    if (cachedParamBounds) return cachedParamBounds;
    try {
        const res = await fetch('/models/api/param_bounds');
        cachedParamBounds = await res.json();
        return cachedParamBounds;
    } catch (e) {
        console.error('Failed to load param bounds', e);
        return null;
    }
}

function toParamBool(val) {
    if (typeof val === 'string') return ['1', 'true', 'yes', 'on'].includes(val.toLowerCase());
    return Boolean(val);
}

function formatParamValue(key, val, spec) {
    if (spec.type === 'bool') return toParamBool(val) ? 'On' : 'Off';
    if (val === null || val === undefined || isNaN(val)) return '—';
    const num = parseFloat(val);
    if (key === 'eval_games' && num === 0) return '0 (Skipped)';
    if (spec.type === 'int') return Math.round(num).toString();
    if (spec.step < 0.001) return num.toFixed(4);
    if (spec.step < 0.01) return num.toFixed(3);
    if (spec.step < 0.1) return num.toFixed(2);
    return num.toFixed(1);
}

function buildParamSlidersHTML(prefix, boundsData, values = {}) {
    if (!boundsData || !boundsData.categories || !boundsData.bounds) return '';
    const { categories, bounds } = boundsData;

    return categories.map(cat => {
        const catBounds = Object.values(bounds).filter(b => b.category === cat.key);
        if (!catBounds.length) return '';

        const slidersHTML = catBounds.map(spec => {
            const inputId = `${prefix}-${spec.key}`;
            const badgeId = `${prefix}-${spec.key}-badge`;
            const initVal = values[spec.key] !== undefined && values[spec.key] !== null
                ? values[spec.key] : spec.default;
            const displayVal = formatParamValue(spec.key, initVal, spec);

            if (spec.type === 'bool') {
                return `
                <div class="param-slider-group param-toggle-group">
                    <div class="param-slider-header">
                        <label for="${inputId}">${spec.label}</label>
                        <span class="param-slider-value" id="${badgeId}">${displayVal}</span>
                    </div>
                    <label class="toggle param-toggle">
                        <input type="checkbox"
                               id="${inputId}"
                               data-key="${spec.key}"
                               ${toParamBool(initVal) ? 'checked' : ''}>
                        <span class="toggle-slider"></span>
                        <span class="param-hint" title="${spec.hint}">${spec.hint}</span>
                    </label>
                </div>
                `;
            }

            return `
                <div class="param-slider-group">
                    <div class="param-slider-header">
                        <label for="${inputId}">${spec.label}</label>
                        <span class="param-slider-value" id="${badgeId}">${displayVal}</span>
                    </div>
                    <input type="range" 
                           id="${inputId}" 
                           class="param-slider" 
                           data-key="${spec.key}"
                           min="${spec.min}" 
                           max="${spec.max}" 
                           step="${spec.step}" 
                           value="${initVal}">
                    <div class="param-slider-meta">
                        <span>${spec.min}</span>
                        <span class="param-hint" title="${spec.hint}">${spec.hint}</span>
                        <span>${spec.max}</span>
                    </div>
                </div>
            `;
        }).join('');

        return `
            <div class="param-category-block">
                <h5 class="param-category-title">${cat.label}</h5>
                <div class="param-sliders-grid">
                    ${slidersHTML}
                </div>
            </div>
        `;
    }).join('');
}

function bindParamSliders(prefix, boundsData) {
    if (!boundsData || !boundsData.bounds) return;
    const { bounds } = boundsData;

    Object.values(bounds).forEach(spec => {
        const input = document.getElementById(`${prefix}-${spec.key}`);
        const badge = document.getElementById(`${prefix}-${spec.key}-badge`);
        if (input && badge) {
            input.addEventListener('input', () => {
                const raw = spec.type === 'bool' ? input.checked : input.value;
                badge.textContent = formatParamValue(spec.key, raw, spec);
            });
        }
    });
}

function extractParamSliderValues(prefix, boundsData) {
    const result = {};
    if (!boundsData || !boundsData.bounds) return result;

    Object.values(boundsData.bounds).forEach(spec => {
        const input = document.getElementById(`${prefix}-${spec.key}`);
        if (input) {
            if (spec.type === 'bool') {
                result[spec.key] = input.checked;
            } else {
                const num = parseFloat(input.value);
                result[spec.key] = spec.type === 'int' ? Math.round(num) : num;
            }
        }
    });
    return result;
}

function setParamSliderValues(prefix, boundsData, values = {}) {
    if (!boundsData || !boundsData.bounds) return;

    Object.values(boundsData.bounds).forEach(spec => {
        const input = document.getElementById(`${prefix}-${spec.key}`);
        const badge = document.getElementById(`${prefix}-${spec.key}-badge`);
        // A stored null means "never set for this model" — show the default.
        const val = values[spec.key] !== undefined && values[spec.key] !== null
            ? values[spec.key] : spec.default;
        if (input) {
            if (spec.type === 'bool') {
                input.checked = toParamBool(val);
            } else {
                input.value = val;
            }
        }
        if (badge) badge.textContent = formatParamValue(spec.key, val, spec);
    });
}
