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
        // Sort by the explicit `order` field, never by key: the bounds arrive as
        // a JSON object and Flask alphabetises its keys, which put Temp Final
        // ahead of Temp Init.
        const catBounds = Object.values(bounds)
            .filter(b => b.category === cat.key)
            .sort((a, b) => (a.order ?? 999) - (b.order ?? 999));
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
                    <div class="param-slider-warn" id="${prefix}-${spec.key}-warn" hidden></div>
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
                        <span>${spec.max}</span>
                    </div>
                    <div class="param-hint" title="${spec.hint}">${spec.hint}</div>
                    <div class="param-slider-warn" id="${prefix}-${spec.key}-warn" hidden></div>
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

// Phases whose games are dealt out to the shared worker pool. A phase finishes
// only when its slowest game does, so a game count that is not a multiple of
// the worker count leaves workers idle through the final wave.
const WORKER_POOL_PHASES = {
    num_self_play_games: 'Self-play',
    gate_games: 'Gate',
};

function updateWorkerBalance(prefix, boundsData) {
    const workerInput = document.getElementById(`${prefix}-num_parallel_workers`);
    if (!workerInput) return;

    // The backend also caps workers at the host's core count, so mirror that
    // here — otherwise we would report a wave layout that never happens.
    const cpuCount = boundsData.cpu_count || 8;
    const workers = Math.min(Math.round(parseFloat(workerInput.value)), cpuCount);
    const gateOn = document.getElementById(`${prefix}-gate_enabled`);

    Object.entries(WORKER_POOL_PHASES).forEach(([key, label]) => {
        const warn = document.getElementById(`${prefix}-${key}-warn`);
        const input = document.getElementById(`${prefix}-${key}`);
        if (!warn || !input) return;

        const games = Math.round(parseFloat(input.value));
        const effective = Math.min(workers, games);
        const phaseSkipped = (key === 'gate_games' && gateOn && !gateOn.checked)
            || games <= 0;

        if (phaseSkipped || effective <= 1 || games % effective === 0) {
            warn.hidden = true;
            warn.textContent = '';
            return;
        }

        const waves = Math.ceil(games / effective);
        const lastWave = games % effective;
        const idle = effective - lastWave;
        warn.hidden = false;
        warn.textContent =
            `⚠ ${games} games over ${effective} workers = ${waves} waves; ` +
            `the last runs ${lastWave} game${lastWave === 1 ? '' : 's'} ` +
            `with ${idle} worker${idle === 1 ? '' : 's'} idle.`;
    });
}

// Which phases write game records, and which game-count parameter each one is
// priced against. The bytes are the whole argument for turning them off — see
// the `storage` category in param_bounds.py.
const RECORDING_PHASES = {
    record_self_play_games: { games: 'num_self_play_games', label: 'Self-play' },
    record_gate_games: { games: 'gate_games', label: 'Gate' },
};

function numericInputValue(prefix, key, fallback = 0) {
    const el = document.getElementById(`${prefix}-${key}`);
    if (!el) return fallback;
    const val = parseFloat(el.value);
    return Number.isFinite(val) ? val : fallback;
}

/**
 * Bytes at a human scale, picking the unit from the magnitude.
 *
 * Fixed units do not work across the range this warning spans: one iteration of
 * gate games is half a megabyte and a thousand iterations of them is half a
 * gigabyte. Printing the first as "0 MB" told the reader the cost was nothing
 * in the same sentence that called it a problem.
 */
function formatBytes(bytes) {
    if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(bytes >= 1e10 ? 0 : 1)} GB`;
    if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(bytes >= 1e7 ? 0 : 1)} MB`;
    return `${Math.max(1, Math.round(bytes / 1e3))} KB`;
}

/**
 * Price the recording toggles in megabytes.
 *
 * A phase that records N games per iteration costs N x ~12 KB (on 9x9) EVERY
 * iteration, forever, and the two settings that raise N are exactly the ones a
 * machine with twenty cores wants to raise. The charts on the training page do
 * not read these records — they read games/index.jsonl, which is written
 * either way — so the only thing recording buys is the ability to replay those
 * games, and the only thing it costs is disk.
 */
function updateStorageWarnings(prefix, boundsData) {
    const storage = boundsData.storage || {};
    const bytesPerGame = storage.bytes_per_game_9x9 || 12000;
    const warnGames = storage.warn_games || 24;

    // Board size lives outside the slider set (it is a structural field on the
    // create form), so read it when present and fall back to 9x9 pricing.
    const boardEl = document.getElementById(`${prefix}-board_size`)
        || document.getElementById('new-model-board-size');
    const boardSize = boardEl
        ? (parseInt(boardEl.value, 10) || 9)
        : (window.ACTIVE_BOARD_SIZE || 9);
    const areaScale = (boardSize * boardSize) / 81;

    const gateOn = document.getElementById(`${prefix}-gate_enabled`);

    Object.entries(RECORDING_PHASES).forEach(([key, spec]) => {
        const warn = document.getElementById(`${prefix}-${key}-warn`);
        const toggle = document.getElementById(`${prefix}-${key}`);
        if (!warn || !toggle) return;

        const games = Math.round(numericInputValue(prefix, spec.games, 0));
        const phaseSkipped = key === 'record_gate_games' && gateOn && !gateOn.checked;

        if (!toggle.checked || phaseSkipped || games < warnGames) {
            warn.hidden = true;
            warn.textContent = '';
            warn.classList.remove('is-danger');
            return;
        }

        // Per iteration is what the setting costs; per 1000 iterations is the
        // number that decides whether you want it, because these files are
        // never cleaned up and a long run is where they accumulate.
        const perIter = games * bytesPerGame * areaScale;
        warn.hidden = false;
        warn.classList.add('is-danger');
        warn.textContent =
            `⚠ ${spec.label} records ${games} games every iteration — ` +
            `${formatBytes(perIter)} per iteration, ${formatBytes(perIter * 100)} per 100 ` +
            `iterations, ${formatBytes(perIter * 1000)} per 1000, and nothing ever ` +
            `deletes them. Every chart on the training page is drawn from the games ` +
            `index, not these files: turning this off keeps all the statistics and ` +
            `loses only the ability to replay those games.`;
    });
}

function bindParamSliders(prefix, boundsData) {
    if (!boundsData || !boundsData.bounds) return;
    const { bounds } = boundsData;

    const refresh = () => {
        updateWorkerBalance(prefix, boundsData);
        updateStorageWarnings(prefix, boundsData);
    };

    Object.values(bounds).forEach(spec => {
        const input = document.getElementById(`${prefix}-${spec.key}`);
        const badge = document.getElementById(`${prefix}-${spec.key}-badge`);
        if (input && badge) {
            // Checkboxes fire `change`, ranges fire `input`; binding both keeps
            // the two warning passes in sync whichever control moved.
            const evt = spec.type === 'bool' ? 'change' : 'input';
            input.addEventListener(evt, () => {
                const raw = spec.type === 'bool' ? input.checked : input.value;
                badge.textContent = formatParamValue(spec.key, raw, spec);
                refresh();
            });
        }
    });

    refresh();
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

    updateWorkerBalance(prefix, boundsData);
    updateStorageWarnings(prefix, boundsData);
}

/**
 * Recording defaults for a NEW model, given the game counts it was created with.
 *
 * A model configured for 60 games an iteration should not start life writing
 * every one of them to disk — that is a gigabyte a week for replays nobody
 * asked for. Applied only where there is no stored value to respect (the
 * create form), never on edit, and the toggles remain the user's to flip.
 */
function applyRecordingDefaults(prefix, boundsData, values = {}) {
    const warnGames = (boundsData.storage && boundsData.storage.warn_games) || 24;

    Object.entries(RECORDING_PHASES).forEach(([key, spec]) => {
        if (values[key] !== undefined && values[key] !== null) return;  // explicit choice
        const toggle = document.getElementById(`${prefix}-${key}`);
        if (!toggle) return;
        const games = Math.round(numericInputValue(prefix, spec.games, 0));
        if (games >= warnGames) {
            toggle.checked = false;
            const badge = document.getElementById(`${prefix}-${key}-badge`);
            if (badge) badge.textContent = 'Off';
        }
    });

    updateStorageWarnings(prefix, boundsData);
}
