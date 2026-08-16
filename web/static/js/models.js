/**
 * models.js — the model fleet console (/models).
 *
 * The page this replaces was a Jinja snapshot of one model plus four buttons,
 * and every action reloaded it. This one holds the whole fleet in memory from
 * a single /models/api/summary call and re-renders in place, because the two
 * things it has to show — training progress and the differences between forks
 * of one run — are both invisible in a snapshot.
 *
 * Layout of this file:
 *   state + helpers · fleet header · fleet list · detail panel · compare mode
 *   · actions · modals · socket wiring
 *
 * The live dashboard (index.html + dashboard.js) is untouched.
 */

(() => {
    'use strict';

    const SORT_KEY = 'fleet.sort';
    const SELECT_KEY = 'fleet.selected';

    const state = {
        models: [],
        byId: {},
        lineage: {},
        headToHead: {},
        training: { is_running: false, model_id: null },
        totals: {},
        selectedId: null,
        sort: localStorage.getItem(SORT_KEY) || 'lineage',
        compareMode: false,
        compareIds: [],
        charts: [],
        bounds: null,
        // Config block: whether it is expanded, and whether it lists every
        // setting or only the ones that differ from the defaults. Both persist
        // across re-renders, which happen on every training event.
        configOpen: false,
        configShowAll: false,
        matchWatch: null,   // {match_id, a, b, status, series}
        refreshTimer: null,
    };

    // ---- formatting ------------------------------------------------------

    const el = (id) => document.getElementById(id);

    function escapeHtml(str) {
        return String(str ?? '').replace(/[&<>"']/g, (c) => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
        }[c]));
    }
    const escapeAttr = escapeHtml;

    function fmtBytes(n) {
        if (!n) return '0 B';
        if (n >= 1e9) return (n / 1e9).toFixed(1) + ' GB';
        if (n >= 1e6) return (n / 1e6).toFixed(1) + ' MB';
        if (n >= 1e3) return Math.round(n / 1e3) + ' KB';
        return n + ' B';
    }

    function fmtDuration(seconds) {
        if (!seconds) return '—';
        const h = Math.floor(seconds / 3600);
        const m = Math.round((seconds % 3600) / 60);
        if (h >= 1) return `${h}h ${m}m`;
        if (m >= 1) return `${m}m`;
        return `${Math.round(seconds)}s`;
    }

    /** "3 days ago" reads faster than a timestamp when scanning a list. */
    function fmtAgo(iso) {
        if (!iso) return 'never';
        const then = new Date(iso).getTime();
        if (isNaN(then)) return 'never';
        const mins = Math.round((Date.now() - then) / 60000);
        if (mins < 1) return 'just now';
        if (mins < 60) return `${mins}m ago`;
        const hours = Math.round(mins / 60);
        if (hours < 24) return `${hours}h ago`;
        const days = Math.round(hours / 24);
        return days === 1 ? 'yesterday' : `${days} days ago`;
    }

    function fmtSigned(n) {
        if (n === null || n === undefined) return '';
        const r = Math.round(n);
        return r > 0 ? `+${r}` : `${r}`;
    }

    // ---- data ------------------------------------------------------------

    async function loadFleet() {
        let data;
        try {
            const res = await fetch('/models/api/summary');
            data = await res.json();
        } catch (e) {
            el('fleet-rows').innerHTML =
                '<p class="empty-hint">Could not load models. Is the server still running?</p>';
            return;
        }

        state.models = data.models || [];
        state.byId = Object.fromEntries(state.models.map((m) => [m.id, m]));
        state.lineage = data.lineage || {};
        state.headToHead = data.head_to_head || {};
        state.training = data.training || { is_running: false, model_id: null };
        state.totals = data.totals || {};

        // Keep the current selection if it still exists; otherwise fall back to
        // the last one this browser looked at, then the active model.
        const remembered = localStorage.getItem(SELECT_KEY);
        if (!state.byId[state.selectedId]) {
            state.selectedId = state.byId[remembered] ? remembered
                : (data.active_model_id || (state.models[0] || {}).id || null);
        }
        state.compareIds = state.compareIds.filter((id) => state.byId[id]);

        renderAll();
    }

    /**
     * Refetch after a training event, coalesced.
     *
     * Socket events arrive per game; the fleet summary walks every model's
     * games directory, so refetching on each one would spend more time in
     * os.walk than in training.
     */
    function scheduleRefresh(delay = 1500) {
        if (state.refreshTimer) return;
        state.refreshTimer = setTimeout(() => {
            state.refreshTimer = null;
            loadFleet();
        }, delay);
    }

    function renderAll() {
        renderTotals();
        renderLive();
        renderRows();
        renderRight();
    }

    /** Detail or compare, depending on mode. */
    function renderRight() {
        if (state.compareMode) renderCompare();
        else renderDetail();
    }

    // ---- fleet header ----------------------------------------------------

    function renderTotals() {
        const t = state.totals;
        const bits = [
            `${t.models || 0} model${t.models === 1 ? '' : 's'}`,
            `${t.families || 0} famil${t.families === 1 ? 'y' : 'ies'}`,
            `${(t.iterations || 0).toLocaleString()} iterations`,
            `${(t.games_on_disk || 0).toLocaleString()} games`,
            fmtBytes(t.bytes_on_disk),
            `${fmtDuration(t.train_seconds)} trained`,
        ];
        if (t.archived) bits.push(`${t.archived} archived`);
        el('fleet-totals').innerHTML = bits
            .map((b) => `<span class="fleet-total">${escapeHtml(b)}</span>`)
            .join('<span class="fleet-total-sep">·</span>');
    }

    function renderLive() {
        const box = el('fleet-live');
        const t = state.training;
        if (!t.is_running || !t.model_id) {
            box.style.display = 'none';
            box.innerHTML = '';
            return;
        }

        const model = state.byId[t.model_id];
        const pct = Math.max(0, Math.min(100, Math.round(t.percent || 0)));
        box.style.display = '';
        box.innerHTML = `
            <span class="live-dot" aria-hidden="true"></span>
            <div class="live-text">
                <strong>${escapeHtml(model ? model.name : t.model_id)}</strong>
                <span class="live-stage">${escapeHtml(t.stage_name || 'Training')}${
                    t.iteration ? ` · iteration ${t.iteration}` : ''}</span>
                ${t.detail ? `<span class="live-detail">${escapeHtml(t.detail)}</span>` : ''}
            </div>
            <div class="live-bar"><div class="live-bar-fill" style="width:${pct}%"></div></div>
            <button class="btn-small" data-act="stop-training">Stop</button>
        `;
    }

    // ---- fleet list ------------------------------------------------------

    /**
     * Row order.
     *
     * `lineage` nests forks under the run they came from — with names like
     * "hero of time" / "hero of time for danya" that grouping IS the
     * information. Every other sort is flat, because a sorted tree is a lie
     * about one of the two orderings.
     */
    function orderedRows() {
        const live = state.models.filter((m) => !m.archived);

        if (state.sort !== 'lineage') {
            // An untrained model still carries the starting Elo of 500 in its
            // config, which would rank it above a trained model that has since
            // dropped below that. Nothing has been measured about it, so it
            // sorts last whatever the column.
            const key = {
                elo: (m) => (m.iterations_logged ? -(m.elo || 0) : Infinity),
                iterations: (m) => -(m.iterations_logged || 0),
                recent: (m) => (m.last_trained ? -new Date(m.last_trained).getTime() : Infinity),
            }[state.sort];
            return live.slice().sort((a, b) => key(a) - key(b) || a.name.localeCompare(b.name))
                .map((m) => ({ model: m, depth: 0 }));
        }

        const node = (id) => state.lineage[id] || {};
        const byParent = {};
        live.forEach((m) => {
            const parent = node(m.id).parent;
            const bucket = (parent && state.byId[parent] && !state.byId[parent].archived)
                ? parent : '__root__';
            (byParent[bucket] = byParent[bucket] || []).push(m);
        });

        const rows = [];
        const walk = (parentId, depth) => {
            const kids = (byParent[parentId] || []).slice().sort((a, b) => {
                // Trunks first by activity, forks by where they split off.
                if (depth === 0) return (b.iterations_logged || 0) - (a.iterations_logged || 0);
                return (node(a.id).fork_iteration || 0) - (node(b.id).fork_iteration || 0);
            });
            kids.forEach((m) => {
                rows.push({ model: m, depth });
                walk(m.id, depth + 1);
            });
        };
        walk('__root__', 0);
        return rows;
    }

    /**
     * An Elo trajectory in 60×18. A final number cannot distinguish a run that
     * is still climbing from one that stalled twenty iterations ago.
     */
    function sparkline(series, variant = '') {
        if (!series || series.length < 2) {
            return '<span class="spark-empty" title="Not enough iterations to plot">·</span>';
        }
        const w = 60, h = 18, pad = 2;
        const min = Math.min(...series), max = Math.max(...series);
        const span = (max - min) || 1;
        const pts = series.map((v, i) => {
            const x = pad + (i / (series.length - 1)) * (w - pad * 2);
            const y = h - pad - ((v - min) / span) * (h - pad * 2);
            return `${x.toFixed(1)},${y.toFixed(1)}`;
        });
        const last = pts[pts.length - 1].split(',');
        return `
            <svg class="spark ${variant}" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}"
                 aria-hidden="true" focusable="false">
                <polyline points="${pts.join(' ')}" fill="none" stroke="currentColor"
                          stroke-width="1.4" stroke-linejoin="round" stroke-linecap="round"/>
                <circle cx="${last[0]}" cy="${last[1]}" r="1.9" fill="currentColor"/>
            </svg>`;
    }

    function healthDot(model) {
        const level = (model.health || {}).level || 'idle';
        const title = (model.health || {}).headline || '';
        return `<span class="health-dot is-${level}" title="${escapeAttr(title)}"></span>`;
    }

    function rowHTML({ model, depth }) {
        const node = state.lineage[model.id] || {};
        const classes = ['fleet-row'];
        if (model.id === state.selectedId && !state.compareMode) classes.push('selected');
        if (state.compareMode && state.compareIds.includes(model.id)) classes.push('compare-picked');
        if (model.is_training) classes.push('is-training');
        if (!model.iterations_logged) classes.push('is-untrained');

        const badges = [];
        if (model.is_active) badges.push('<span class="row-badge is-active" title="The trainer is bound to this model">active</span>');
        if (model.is_training) badges.push('<span class="row-badge is-live">training</span>');

        const forkNote = (depth > 0 && node.fork_iteration)
            ? `<span class="row-fork" title="Forked from ${escapeAttr((state.byId[node.parent] || {}).name || node.parent)} at iteration ${node.fork_iteration}">@${node.fork_iteration}</span>`
            : '';

        const delta = model.elo_delta_10;
        const deltaClass = delta > 0 ? 'up' : (delta < 0 ? 'down' : 'flat');

        return `
            <div class="${classes.join(' ')}" data-model-id="${escapeAttr(model.id)}"
                 style="--depth:${depth}" role="button" tabindex="0">
                ${healthDot(model)}
                <div class="row-main">
                    <div class="row-name">
                        ${depth > 0 ? '<span class="row-branch" aria-hidden="true">└</span>' : ''}
                        <span class="row-title">${escapeHtml(model.name)}</span>
                        ${forkNote}${badges.join('')}
                    </div>
                    <div class="row-sub">
                        ${model.iterations_logged
                            ? `${model.iterations_logged} iter · ${escapeHtml(fmtAgo(model.last_trained))}`
                            : 'never trained'}
                        · ${model.board_size}×${model.board_size}
                    </div>
                </div>
                <div class="row-spark ${deltaClass}">${sparkline(model.elo_series, deltaClass)}</div>
                <div class="row-elo">
                    <span class="row-elo-value">${model.iterations_logged ? Math.round(model.elo) : '—'}</span>
                    ${delta ? `<span class="row-elo-delta ${deltaClass}">${escapeHtml(fmtSigned(delta))}</span>` : ''}
                </div>
            </div>`;
    }

    function renderRows() {
        const list = el('fleet-rows');
        const rows = orderedRows();

        if (!state.models.length) {
            list.innerHTML = '<p class="empty-hint">No models yet. Create one to get started.</p>';
            el('fleet-archived').innerHTML = '';
            return;
        }

        list.innerHTML = rows.map(rowHTML).join('')
            || '<p class="empty-hint">Every model is archived.</p>';

        document.querySelectorAll('.sort-btn').forEach((btn) => {
            btn.classList.toggle('active', btn.dataset.sort === state.sort);
        });

        const archived = state.models.filter((m) => m.archived);
        el('fleet-archived').innerHTML = archived.length ? `
            <details class="archived-group">
                <summary>Archived <span class="group-note">${archived.length}</span></summary>
                ${archived.map((m) => `
                    <div class="fleet-row is-archived" data-model-id="${escapeAttr(m.id)}"
                         style="--depth:0" role="button" tabindex="0">
                        ${healthDot(m)}
                        <div class="row-main">
                            <div class="row-name"><span class="row-title">${escapeHtml(m.name)}</span></div>
                            <div class="row-sub">${m.iterations_logged} iter · ${escapeHtml(fmtBytes(m.bytes_on_disk))}</div>
                        </div>
                        <button class="btn-small" data-act="unarchive" data-id="${escapeAttr(m.id)}">Restore</button>
                    </div>`).join('')}
            </details>` : '';
    }

    // ---- detail panel ----------------------------------------------------

    function destroyCharts() {
        state.charts.forEach((c) => c.destroy());
        state.charts = [];
    }

    function netLabel(model) {
        const net = model.network || {};
        const preset = net.size_preset || 'small';
        return `${preset.charAt(0).toUpperCase()}${preset.slice(1)} (${net.num_res_blocks || 4}×${net.num_filters || 64})`;
    }

    function vitalsHTML(model) {
        const node = state.lineage[model.id] || {};
        const gate = model.gate_matches
            ? `${model.gate_promotions}/${model.gate_matches}`
            : 'none';
        const vitals = [
            ['Elo', model.iterations_logged ? Math.round(model.elo) : '—',
             model.elo_delta_10 ? `${fmtSigned(model.elo_delta_10)} over 10 iterations` : model.kyu_rank],
            ['Iterations', model.iterations_logged || 0,
             node.parent ? `${node.own_iterations} since the fork` : 'from scratch'],
            ['Gate record', gate,
             model.gate_matches ? 'candidates promoted' : 'Elo rests on the random-bot eval'],
            ['Games', (model.games_on_disk || 0).toLocaleString(), `${fmtBytes(model.bytes_on_disk)} on disk`],
            ['Trained for', fmtDuration(model.total_train_seconds), fmtAgo(model.last_trained)],
            ['Last loss', model.last_loss ?? '—', model.buffer_size ? `buffer ${model.buffer_size.toLocaleString()}` : ''],
        ];
        return `<div class="vitals">${vitals.map(([label, value, note]) => `
            <div class="vital">
                <span class="vital-label">${escapeHtml(label)}</span>
                <span class="vital-value">${escapeHtml(String(value))}</span>
                <span class="vital-note">${escapeHtml(note || '')}</span>
            </div>`).join('')}</div>`;
    }

    function healthHTML(model) {
        const h = model.health || {};
        const reasons = (h.reasons || []).map((r) => `
            <li class="health-reason is-${r.level}">${escapeHtml(r.text)}</li>`).join('');
        return `
            <div class="health-strip is-${h.level || 'idle'}">
                <div class="health-head">
                    <span class="health-dot is-${h.level || 'idle'}"></span>
                    <strong>${escapeHtml(h.headline || '')}</strong>
                </div>
                ${reasons ? `<ul class="health-reasons">${reasons}</ul>` : ''}
            </div>`;
    }

    /**
     * Where this model came from.
     *
     * Only the PARENT is named. Its children are already visible in the
     * sidebar — nested under this row with their fork iteration on each — so
     * listing them here restated the tree in prose next to the tree itself.
     */
    function lineageHTML(model) {
        const node = state.lineage[model.id] || {};
        const parent = node.parent ? state.byId[node.parent] : null;
        if (!parent) return '';

        const since = node.own_iterations
            ? `, ${node.own_iterations} iteration${node.own_iterations === 1 ? '' : 's'} since`
            : ' — never trained since';
        return `<div class="lineage-line">Forked from
            <a href="#" data-act="select" data-id="${escapeAttr(parent.id)}">${escapeHtml(parent.name)}</a>
            at iteration ${node.fork_iteration}${since}</div>`;
    }

    function h2hHTML(model) {
        const opponents = state.headToHead[model.id] || {};
        const entries = Object.entries(opponents).sort((a, b) => b[1].games - a[1].games);

        const played = entries.map(([id, rec]) => {
            const name = id === 'random' ? 'Random Bot' : (state.byId[id] || {}).name || id;
            const verdict = rec.wins > rec.losses ? 'win' : (rec.wins < rec.losses ? 'loss' : 'draw');
            const canRematch = id === 'random' || (state.byId[id] && state.byId[id].board_size === model.board_size);
            return `
                <div class="h2h-row">
                    <span class="h2h-name">${escapeHtml(name)}</span>
                    <span class="h2h-record is-${verdict}">${rec.wins}–${rec.losses}${rec.draws ? `–${rec.draws}` : ''}</span>
                    <span class="h2h-note">${escapeHtml(fmtAgo(rec.last_played))}</span>
                    ${canRematch ? `<button class="btn-small" data-act="match" data-id="${escapeAttr(id)}">Rematch</button>` : ''}
                </div>`;
        }).join('');

        return `
            <section class="detail-section">
                <h3 class="detail-section-title">Head to head
                    <span class="section-note">real games, not Elo</span></h3>
                ${played || '<p class="empty-hint">No matches played yet — Elo alone cannot rank this model against the others.</p>'}
                ${matchWatchHTML()}
            </section>`;
    }

    function matchWatchHTML() {
        const w = state.matchWatch;
        if (!w) return '';
        const done = w.status === 'finished' || w.status === 'stopped' || w.status === 'error';
        return `
            <div class="match-watch ${done ? 'is-done' : ''}">
                <span class="match-watch-title">${escapeHtml(w.name || 'Match')}</span>
                <span class="match-watch-score">${w.series ? `${w.series.a}–${w.series.b}` : '—'}</span>
                <span class="muted">${escapeHtml(done ? w.status : `game ${w.current_game || 1} of ${w.num_games || '?'}`)}</span>
                <a class="btn-small" href="/play" target="_blank" rel="noopener">Watch</a>
            </div>`;
    }

    // Training params that have no slider, so PARAM_BOUNDS carries no label
    // for them. Without these they rendered as raw `num_epochs_per_iteration`
    // among human-written labels.
    const EXTRA_PARAM_LABELS = {
        batch_size: 'Batch Size',
        num_epochs_per_iteration: 'Epochs per Iteration',
        replay_buffer_size: 'Replay Buffer',
        reflection_interval_games: 'Reflection Interval',
    };

    /** A parameter's value as the sliders would show it. */
    function fmtParam(spec, value) {
        if (value === undefined || value === null) return '—';
        return spec.type ? formatParamValue(spec.key, value, spec) : String(value);
    }

    /** One parameter, laid out label-over-value so 27 of them stay scannable. */
    function configItemHTML(spec, value) {
        const isDefault = spec.default === undefined || value === spec.default;
        const isBool = spec.type === 'bool';

        let shown = fmtParam(spec, value);
        const shownDefault = fmtParam(spec, spec.default);
        // A setting can differ from its default by less than the slider's own
        // display precision (temperature_final 0.101 vs 0.1). Rounding both to
        // the same string would flag the row as changed and then show two
        // identical numbers, so fall back to the raw value.
        if (!isDefault && shown === shownDefault) shown = String(value);

        const title = [spec.hint, isDefault ? '' : `Default: ${shownDefault}`]
            .filter(Boolean).join(' · ');

        return `
            <div class="config-item ${isDefault ? '' : 'is-changed'}" title="${escapeAttr(title)}">
                <span class="config-key">${escapeHtml(spec.label || spec.key)}</span>
                <span class="config-val ${isBool ? `is-bool is-${shown.toLowerCase()}` : ''}">${escapeHtml(shown)}</span>
                ${isDefault ? '' : `<span class="config-was">was ${escapeHtml(shownDefault)}</span>`}
            </div>`;
    }

    /**
     * The configuration block.
     *
     * Two problems with showing 27 settings at once: the block did not look
     * expandable, and every row had the same weight, so the handful that were
     * actually changed from default did not stand out. It now opens as a
     * bordered panel with a rotating chevron, and defaults to showing ONLY the
     * changed settings — the ones that make this model different from every
     * other model in the list. "All" is one click away.
     */
    function configHTML(model) {
        const bounds = state.bounds;
        const training = model.training || {};
        if (!bounds) return '';

        // Grouped the way the edit modal groups them, so the read-only view and
        // the editor never drift apart.
        const specs = Object.values(bounds.bounds);
        const groups = bounds.categories.map((cat) => ({
            label: cat.label,
            params: specs.filter((s) => s.category === cat.key)
                .sort((a, b) => (a.order || 0) - (b.order || 0)),
        })).filter((g) => g.params.length);

        const known = new Set(specs.map((s) => s.key));
        const extras = Object.keys(training).filter((k) => !known.has(k));
        if (extras.length) {
            groups.push({
                label: 'Storage & Batching',
                params: extras.map((k) => ({ key: k, label: EXTRA_PARAM_LABELS[k] || k })),
            });
        }

        const isChanged = (spec) => spec.default !== undefined && training[spec.key] !== spec.default;
        const total = groups.reduce((n, g) => n + g.params.length, 0);
        const changed = groups.reduce((n, g) => n + g.params.filter(isChanged).length, 0);
        const showAll = state.configShowAll || changed === 0;

        const body = groups.map((group) => {
            const params = showAll ? group.params : group.params.filter(isChanged);
            if (!params.length) return '';
            return `
                <div class="config-group">
                    <span class="config-group-title">${escapeHtml(group.label)}</span>
                    <div class="config-grid">
                        ${params.map((spec) => configItemHTML(spec, training[spec.key])).join('')}
                    </div>
                </div>`;
        }).join('');

        return `
            <details class="detail-section config-section" ${state.configOpen ? 'open' : ''}>
                <summary class="config-summary">
                    <span class="config-chevron" aria-hidden="true">▸</span>
                    <span class="config-summary-title">Configuration</span>
                    <span class="section-note">${total} settings${
                        changed ? ` · <strong>${changed} changed</strong>` : ' · all at defaults'}</span>
                </summary>
                <div class="config-body">
                    ${changed ? `
                        <div class="config-filter" role="group" aria-label="Which settings to show">
                            <button class="filter-pill ${showAll ? '' : 'active'}"
                                    data-act="config-changed">Changed <span class="filter-count">${changed}</span></button>
                            <button class="filter-pill ${showAll ? 'active' : ''}"
                                    data-act="config-all">All <span class="filter-count">${total}</span></button>
                        </div>` : ''}
                    ${body}
                </div>
            </details>`;
    }

    /**
     * The action row.
     *
     * "Active" is a STATE, not an action — it used to sit here as a disabled
     * button, which made a badge and five controls share one row and one shape.
     * It now lives next to the model name (see statePillsHTML) and this row
     * holds only things you can press. Delete is separated off to the right so
     * the destructive control is never adjacent to the routine ones.
     */
    function actionsHTML(model) {
        const busy = state.training.is_running;
        const isTrainingThis = model.is_training;
        const blocked = busy && !isTrainingThis
            ? 'Another model is training — stop it first'
            : '';

        const btn = (act, label, cls, disabled, title) => `
            <button class="${cls}" data-act="${act}" ${disabled ? 'disabled' : ''}
                    title="${escapeAttr(title || '')}">${label}</button>`;

        return `<div class="detail-actions">
            ${isTrainingThis
                ? btn('stop-training', 'Stop training', 'btn-small btn-accent', false, 'Stop after the current step')
                : btn('train', 'Train', 'btn-small btn-accent', busy, blocked || 'Activate this model and start training')}
            ${model.is_active ? ''
                : btn('activate', 'Activate', 'btn-small', busy, busy ? 'Stop training before switching models' : 'Bind the trainer to this model')}
            ${btn('fork', 'Fork', 'btn-small', isTrainingThis, isTrainingThis ? 'Stop training before forking' : 'Copy this run and change settings from here')}
            ${btn('edit', 'Settings', 'btn-small', isTrainingThis, isTrainingThis ? 'Stop training before editing' : 'Edit hyperparameters')}
            ${btn('archive', 'Archive', 'btn-small', model.is_active, model.is_active ? 'Activate another model first' : 'Fold away without deleting anything')}
            <span class="action-sep" aria-hidden="true"></span>
            ${btn('delete', '🗑', 'btn-small btn-icon-danger', isTrainingThis, 'Delete this model and all its data')}
        </div>`;
    }

    /** Live state, shown with the name it belongs to rather than as a button. */
    function statePillsHTML(model) {
        const pills = [];
        if (model.is_active) {
            pills.push(`<span class="state-pill is-active" title="The trainer is bound to this model">Active</span>`);
        }
        if (model.is_training) {
            pills.push(`<span class="state-pill is-live"><span class="state-pill-dot"></span>Training</span>`);
        }
        if (model.archived) {
            pills.push(`<span class="state-pill is-archived">Archived</span>`);
        }
        return pills.join('');
    }

    function renderDetail() {
        destroyCharts();
        const panel = el('fleet-detail');
        const model = state.byId[state.selectedId];

        if (!model) {
            panel.innerHTML = `
                <div class="no-model-state">
                    <div class="no-model-icon">🧬</div>
                    <h2>No model selected</h2>
                    <p>Pick a model on the left, or create one to start training.</p>
                    <button id="btn-create-model-alt" class="btn-primary">+ Create Your First Model</button>
                </div>`;
            return;
        }

        panel.innerHTML = `
            <div class="detail-head">
                <div class="detail-identity">
                    <div class="detail-title-row">
                        <h2 class="detail-name">${escapeHtml(model.name)}</h2>
                        ${statePillsHTML(model)}
                    </div>
                    <div class="detail-badges">
                        <span class="detail-badge">${model.board_size}×${model.board_size}</span>
                        <span class="detail-badge">${escapeHtml((model.ruleset || '').replace(/^./, (c) => c.toUpperCase()))}</span>
                        <span class="detail-badge">Komi ${model.komi}</span>
                        <span class="detail-badge">${escapeHtml(netLabel(model))}</span>
                        <span class="detail-badge">${escapeHtml(model.kyu_rank || '')}</span>
                    </div>
                    ${lineageHTML(model)}
                </div>
                ${actionsHTML(model)}
            </div>

            <textarea class="detail-notes" id="detail-notes" rows="2"
                      placeholder="Why does this model exist? (saved when you click away)"
                      >${escapeHtml(model.notes || '')}</textarea>

            ${healthHTML(model)}
            ${vitalsHTML(model)}

            <section class="detail-section">
                <h3 class="detail-section-title">Progress
                    <span class="section-note">${model.iterations_logged} iterations</span></h3>
                ${model.iterations_logged >= 2 ? `
                    <div class="chart-grid">
                        <div class="chart-box"><canvas id="chart-elo"></canvas></div>
                        <div class="chart-box"><canvas id="chart-loss"></canvas></div>
                        <div class="chart-box"><canvas id="chart-gate"></canvas></div>
                    </div>`
                    : '<p class="empty-hint">Not enough iterations to chart yet.</p>'}
            </section>

            ${h2hHTML(model)}
            ${configHTML(model)}
        `;

        if (model.iterations_logged >= 2) renderCharts(model);
    }

    // ---- charts ----------------------------------------------------------

    const CHART_BASE = {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: 'index', intersect: false },
        plugins: { legend: { display: false } },
    };

    function axisStyle(title) {
        return {
            grid: { color: 'rgba(255,255,255,0.06)' },
            ticks: { color: '#9a9a9a', maxTicksLimit: 5, font: { size: 10 } },
            title: { display: !!title, text: title, color: '#9a9a9a', font: { size: 10 } },
        };
    }

    function lineChart(canvasId, labels, data, label, color, opts = {}) {
        const canvas = el(canvasId);
        if (!canvas) return;
        const chart = new Chart(canvas.getContext('2d'), {
            type: opts.type || 'line',
            data: {
                labels,
                datasets: [{
                    label,
                    data,
                    borderColor: color,
                    backgroundColor: opts.fill || color,
                    borderWidth: 2,
                    fill: opts.fill ? true : false,
                    tension: 0.3,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    spanGaps: true,
                }],
            },
            options: {
                ...CHART_BASE,
                plugins: {
                    ...CHART_BASE.plugins,
                    title: { display: true, text: label, color: '#c8c8c8', font: { size: 11 }, align: 'start' },
                    tooltip: { callbacks: { title: (i) => `Iteration ${i[0].label}` } },
                },
                scales: { x: axisStyle(''), y: { ...axisStyle(''), ...(opts.yScale || {}) } },
            },
        });
        state.charts.push(chart);
    }

    async function renderCharts(model) {
        let data;
        try {
            const res = await fetch(
                `/models/api/${encodeURIComponent(model.id)}/history?fields=elo,total_loss,gate_win_rate`);
            data = await res.json();
        } catch (e) {
            return;
        }
        // The panel may have been re-rendered while this was in flight.
        if (state.selectedId !== model.id || state.compareMode) return;

        const labels = data.iterations;
        lineChart('chart-elo', labels, data.series.elo, 'Elo', '#c8956c',
                  { fill: 'rgba(200,149,108,0.12)' });
        lineChart('chart-loss', labels, data.series.total_loss, 'Total loss', '#7aa2c8');
        lineChart('chart-gate', labels, data.series.gate_win_rate, 'Gate win rate', '#6cb98a',
                  { yScale: { min: 0, max: 1, ticks: { color: '#9a9a9a', callback: (v) => `${Math.round(v * 100)}%`, font: { size: 10 } } } });
    }

    // ---- compare mode ----------------------------------------------------

    function renderCompare() {
        destroyCharts();
        const panel = el('fleet-detail');
        const [a, b] = state.compareIds.map((id) => state.byId[id]);

        if (!a || !b) {
            panel.innerHTML = `
                <div class="no-model-state">
                    <div class="no-model-icon">⇄</div>
                    <h2>Compare two models</h2>
                    <p>Pick ${a ? 'one more model' : 'two models'} on the left to see what differs
                       between them: settings, curves, and their real record against each other.</p>
                    <button class="btn-secondary" data-act="exit-compare">Cancel</button>
                </div>`;
            return;
        }

        const rec = ((state.headToHead[a.id] || {})[b.id]) || null;
        const sameBoard = a.board_size === b.board_size;

        // Config diff — for a fork, this is the whole reason it exists.
        const specs = state.bounds ? Object.values(state.bounds.bounds) : [];
        const keys = new Set([...Object.keys(a.training || {}), ...Object.keys(b.training || {})]);
        const diffs = [...keys].map((key) => {
            const spec = specs.find((s) => s.key === key) || { key, label: key };
            const va = (a.training || {})[key];
            const vb = (b.training || {})[key];
            return { spec, va, vb, same: va === vb };
        }).filter((d) => !d.same)
          .sort((x, y) => (x.spec.order || 999) - (y.spec.order || 999));

        const show = (spec, v) => (spec.type ? formatParamValue(spec.key, v, spec)
            : (v === undefined ? '—' : String(v)));

        const stat = (label, va, vb) => `
            <tr><td>${escapeHtml(label)}</td>
                <td class="num">${escapeHtml(String(va))}</td>
                <td class="num">${escapeHtml(String(vb))}</td></tr>`;

        panel.innerHTML = `
            <div class="detail-head">
                <div class="detail-identity">
                    <h2 class="detail-name">${escapeHtml(a.name)} <span class="muted">vs</span> ${escapeHtml(b.name)}</h2>
                    <div class="lineage-line">${escapeHtml(compareRelation(a, b))}</div>
                </div>
                <div class="detail-actions">
                    ${sameBoard ? `<button class="btn-small btn-accent" data-act="match-compare">⚔ Run match</button>` : ''}
                    <button class="btn-small" data-act="exit-compare">Done</button>
                </div>
            </div>

            <section class="detail-section">
                <h3 class="detail-section-title">Record against each other</h3>
                ${rec ? `
                    <div class="h2h-big">
                        <span>${escapeHtml(a.name)}</span>
                        <strong>${rec.wins}–${rec.losses}${rec.draws ? `–${rec.draws}` : ''}</strong>
                        <span>${escapeHtml(b.name)}</span>
                    </div>
                    <p class="muted">${rec.games} game${rec.games === 1 ? '' : 's'}, last ${escapeHtml(fmtAgo(rec.last_played))}.</p>`
                : `<p class="empty-hint">They have never played. ${sameBoard
                    ? 'Elo alone cannot rank them — run a match.'
                    : 'Different board sizes, so they cannot meet.'}</p>`}
                ${matchWatchHTML()}
            </section>

            <section class="detail-section">
                <h3 class="detail-section-title">Side by side</h3>
                <div class="scroll-x">
                    <table class="compare-table">
                        <thead><tr><th></th><th class="num">${escapeHtml(a.name)}</th><th class="num">${escapeHtml(b.name)}</th></tr></thead>
                        <tbody>
                            ${stat('Elo', a.iterations_logged ? Math.round(a.elo) : '—', b.iterations_logged ? Math.round(b.elo) : '—')}
                            ${stat('Iterations', a.iterations_logged, b.iterations_logged)}
                            ${stat('Gate promotions', `${a.gate_promotions}/${a.gate_matches}`, `${b.gate_promotions}/${b.gate_matches}`)}
                            ${stat('Trained for', fmtDuration(a.total_train_seconds), fmtDuration(b.total_train_seconds))}
                            ${stat('Last loss', a.last_loss ?? '—', b.last_loss ?? '—')}
                            ${stat('Network', netLabel(a), netLabel(b))}
                            ${stat('Health', (a.health || {}).headline || '—', (b.health || {}).headline || '—')}
                        </tbody>
                    </table>
                </div>
            </section>

            <section class="detail-section">
                <h3 class="detail-section-title">Settings that differ
                    <span class="section-note">${diffs.length || 'none'}</span></h3>
                ${diffs.length ? `
                    <div class="scroll-x">
                        <table class="compare-table">
                            <tbody>${diffs.map((d) => `
                                <tr><td>${escapeHtml(d.spec.label || d.spec.key)}</td>
                                    <td class="num">${escapeHtml(show(d.spec, d.va))}</td>
                                    <td class="num">${escapeHtml(show(d.spec, d.vb))}</td></tr>`).join('')}
                            </tbody>
                        </table>
                    </div>`
                : '<p class="empty-hint">Identical training settings — any difference between them came from training, not configuration.</p>'}
            </section>

            <section class="detail-section">
                <h3 class="detail-section-title">Curves</h3>
                <div class="chart-grid">
                    <div class="chart-box chart-box-wide"><canvas id="chart-cmp-elo"></canvas></div>
                    <div class="chart-box chart-box-wide"><canvas id="chart-cmp-loss"></canvas></div>
                </div>
            </section>`;

        renderCompareCharts(a, b);
    }

    /** One line about how the two are related — usually they are one lineage. */
    function compareRelation(a, b) {
        const na = state.lineage[a.id] || {}, nb = state.lineage[b.id] || {};
        if (nb.parent === a.id) return `${b.name} was forked from ${a.name} at iteration ${nb.fork_iteration}.`;
        if (na.parent === b.id) return `${a.name} was forked from ${b.name} at iteration ${na.fork_iteration}.`;
        if (na.root === nb.root && na.root) {
            return `Both descend from ${(state.byId[na.root] || {}).name || na.root}.`;
        }
        return 'Independent runs.';
    }

    async function renderCompareCharts(a, b) {
        const fetchOne = async (m) => {
            try {
                const res = await fetch(`/models/api/${encodeURIComponent(m.id)}/history?fields=elo,total_loss`);
                return await res.json();
            } catch (e) { return { iterations: [], series: {} }; }
        };
        const [da, db] = await Promise.all([fetchOne(a), fetchOne(b)]);
        if (!state.compareMode) return;

        const pairs = (data, field) => (data.iterations || [])
            .map((it, i) => ({ x: it, y: (data.series[field] || [])[i] }))
            .filter((p) => p.y !== null && p.y !== undefined);

        const overlay = (canvasId, field, title, yScale) => {
            const canvas = el(canvasId);
            if (!canvas) return;
            const chart = new Chart(canvas.getContext('2d'), {
                type: 'line',
                data: {
                    datasets: [
                        { label: a.name, data: pairs(da, field), borderColor: '#c8956c', borderWidth: 2, pointRadius: 0, tension: 0.3 },
                        { label: b.name, data: pairs(db, field), borderColor: '#7aa2c8', borderWidth: 2, pointRadius: 0, tension: 0.3 },
                    ],
                },
                options: {
                    ...CHART_BASE,
                    plugins: {
                        legend: { display: true, labels: { color: '#c8c8c8', boxWidth: 10, font: { size: 10 } } },
                        title: { display: true, text: title, color: '#c8c8c8', font: { size: 11 }, align: 'start' },
                    },
                    scales: {
                        x: { type: 'linear', ...axisStyle('Iteration') },
                        y: { ...axisStyle(''), ...(yScale || {}) },
                    },
                },
            });
            state.charts.push(chart);
        };

        overlay('chart-cmp-elo', 'elo', 'Elo');
        overlay('chart-cmp-loss', 'total_loss', 'Total loss');
    }

    // ---- actions ---------------------------------------------------------

    async function post(url, body, method = 'POST') {
        const res = await fetch(url, {
            method,
            headers: body ? { 'Content-Type': 'application/json' } : {},
            body: body ? JSON.stringify(body) : undefined,
        });
        let data = {};
        try { data = await res.json(); } catch (e) { /* empty body is fine */ }
        if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
        return data;
    }

    function flash(message, kind = 'info') {
        const box = el('fleet-live');
        const note = document.createElement('div');
        note.className = `fleet-flash is-${kind}`;
        note.textContent = message;
        box.parentNode.insertBefore(note, box.nextSibling);
        setTimeout(() => note.remove(), 5000);
    }

    function selectModel(id) {
        if (state.compareMode) {
            const at = state.compareIds.indexOf(id);
            if (at >= 0) state.compareIds.splice(at, 1);
            else state.compareIds = [...state.compareIds, id].slice(-2);
        } else {
            state.selectedId = id;
            localStorage.setItem(SELECT_KEY, id);
        }
        renderRows();
        renderRight();
    }

    async function doActivate(id) {
        try {
            await post(`/models/api/${encodeURIComponent(id)}/select`);
            await loadFleet();
        } catch (e) { flash(e.message, 'error'); }
    }

    async function doTrain(id) {
        try {
            if (!state.byId[id].is_active) {
                await post(`/models/api/${encodeURIComponent(id)}/select`);
            }
            await post('/training/api/start', {});
            flash('Training started.', 'ok');
            scheduleRefresh(800);
        } catch (e) { flash(e.message, 'error'); }
    }

    async function doStop() {
        try {
            await post('/training/api/stop');
            flash('Stop requested — training ends after the current step.', 'ok');
            scheduleRefresh(800);
        } catch (e) { flash(e.message, 'error'); }
    }

    async function doArchive(id, archived) {
        try {
            await post(`/models/api/${encodeURIComponent(id)}/meta`, { archived });
            await loadFleet();
        } catch (e) { flash(e.message, 'error'); }
    }

    async function saveNotes(id, notes) {
        const model = state.byId[id];
        if (!model || (model.notes || '') === notes) return;
        try {
            await post(`/models/api/${encodeURIComponent(id)}/meta`, { notes });
            model.notes = notes;
            flash('Notes saved.', 'ok');
        } catch (e) { flash(e.message, 'error'); }
    }

    // ---- delete (with the measured cost) ---------------------------------

    let deleteTarget = null;

    function openDeleteModal(model) {
        deleteTarget = model;
        el('delete-impact').innerHTML = `
            Deleting <strong>${escapeHtml(model.name)}</strong> removes
            <strong>${model.iterations_logged} iteration${model.iterations_logged === 1 ? '' : 's'}</strong>,
            <strong>${(model.games_on_disk || 0).toLocaleString()} stored games</strong>,
            its weights and its metrics log —
            <strong>${escapeHtml(fmtBytes(model.bytes_on_disk))}</strong> in total. This cannot be undone.
            ${(state.lineage[model.id] || {}).children?.length
                ? ` ${((state.lineage[model.id] || {}).children || []).length} fork(s) of this run will keep their own copies.`
                : ''}`;
        el('delete-confirm-name').value = '';
        el('btn-confirm-delete').disabled = true;
        el('delete-model-modal').style.display = 'flex';
    }

    // ---- match -----------------------------------------------------------

    let matchTarget = null;   // {a, b}

    function openMatchModal(aId, bId) {
        const a = state.byId[aId];
        const b = bId === 'random' ? { id: 'random', name: 'Random Bot' } : state.byId[bId];
        if (!a || !b) return;
        matchTarget = { a, b };

        el('match-modal-intro').innerHTML =
            `<strong>${escapeHtml(a.name)}</strong> vs <strong>${escapeHtml(b.name)}</strong> — ` +
            `played on ${a.board_size}×${a.board_size}, colours alternating. ` +
            `Results update both models' Elo and are saved to Review Games.`;
        el('match-modal-warning').style.display = 'none';
        el('match-modal').style.display = 'flex';
    }

    async function startMatch() {
        if (!matchTarget) return;
        const { a, b } = matchTarget;
        const sims = parseInt(el('match-sims').value, 10) || 0;
        const spec = (m) => (m.id === 'random'
            ? { type: 'random' }
            : { type: 'model', model_id: m.id, ...(sims ? { num_simulations: sims } : {}) });

        try {
            const snap = await post('/api/match/new', {
                player_a: spec(a),
                player_b: spec(b),
                num_games: parseInt(el('match-games').value, 10) || 4,
                name: `${a.name} vs ${b.name}`,
            });
            el('match-modal').style.display = 'none';
            state.matchWatch = {
                match_id: snap.match_id, name: snap.name, status: snap.status,
                series: snap.series, current_game: snap.current_game, num_games: snap.num_games,
            };
            renderRight();
            watchMatch(snap.match_id);
        } catch (e) {
            el('match-modal-warning').textContent = e.message;
            el('match-modal-warning').style.display = '';
        }
    }

    /** Poll a running match so the record updates without leaving the page. */
    function watchMatch(matchId) {
        const tick = async () => {
            let snap;
            try {
                const res = await fetch(`/api/match/${encodeURIComponent(matchId)}`);
                snap = await res.json();
            } catch (e) { return; }
            if (!state.matchWatch || state.matchWatch.match_id !== matchId) return;

            state.matchWatch = {
                match_id: matchId, name: snap.name, status: snap.status,
                series: snap.series, current_game: snap.current_game, num_games: snap.num_games,
            };

            if (['finished', 'stopped', 'error'].includes(snap.status)) {
                await loadFleet();     // the win matrix and both Elos have moved
                return;
            }
            renderRight();
            setTimeout(tick, 3000);
        };
        setTimeout(tick, 2000);
    }

    // ---- create / edit / fork modal --------------------------------------

    const modal = {
        mode: 'create',   // create | edit | fork
        model: null,
        netPresets: [],
        netDefaultKey: 'small',
    };

    const setField = (id, value) => {
        const node = el(id);
        if (node != null && value !== undefined && value !== null) node.value = value;
    };

    function fmtParams(n) {
        if (n >= 1e6) return (n / 1e6).toFixed(2).replace(/\.?0+$/, '') + 'M params';
        if (n >= 1e3) return Math.round(n / 1e3) + 'K params';
        return n + ' params';
    }

    function renderNetSize() {
        const slider = el('new-model-net-size');
        if (!slider || !modal.netPresets.length) return;
        const idx = Math.max(0, Math.min(modal.netPresets.length - 1, parseInt(slider.value, 10) || 0));
        const preset = modal.netPresets[idx];
        el('net-size-label').textContent = preset.label;
        el('net-size-note').textContent = preset.note || '';
        el('net-size-params').textContent = fmtParams(preset.params);
    }

    async function loadNetworkPresets(boardSize, keepKey) {
        const slider = el('new-model-net-size');
        if (!slider) return;
        try {
            const res = await fetch(`/models/api/network_presets?board_size=${boardSize}`);
            const data = await res.json();
            modal.netPresets = data.presets || [];
            modal.netDefaultKey = data.default || 'small';
        } catch (e) { return; }

        slider.max = String(Math.max(0, modal.netPresets.length - 1));
        const target = keepKey || modal.netDefaultKey;
        let idx = modal.netPresets.findIndex((p) => p.key === target);
        if (idx < 0) idx = modal.netPresets.findIndex((p) => p.key === modal.netDefaultKey);
        slider.value = String(Math.max(0, idx));
        renderNetSize();
    }

    async function initModalParams(values = {}) {
        const container = el('modal-param-categories');
        if (!container) return;
        if (!state.bounds) state.bounds = await getParamBounds();
        if (!state.bounds) return;
        container.innerHTML = buildParamSlidersHTML('modal-param', state.bounds, values);
        bindParamSliders('modal-param', state.bounds);
        setParamSliderValues('modal-param', state.bounds, values);
    }

    function setNetLocked(locked) {
        el('new-model-net-size').disabled = locked;
        el('net-size-locked').style.display = locked ? '' : 'none';
    }

    function setFieldsDisabled(disabled) {
        ['new-model-board-size', 'new-model-komi', 'new-model-ruleset'].forEach((id) => {
            el(id).disabled = disabled;
        });
    }

    async function openCreateModal() {
        modal.mode = 'create';
        modal.model = null;
        el('model-form-title').textContent = 'Create New Model';
        el('model-form-subtitle').style.display = 'none';
        el('btn-confirm-create').textContent = 'Create Model';
        el('btn-confirm-create').disabled = false;
        el('model-form-warning').style.display = 'none';

        setField('new-model-name', '');
        setField('new-model-notes', '');
        setField('new-model-board-size', '9');
        setField('new-model-komi', '6.5');
        setField('new-model-ruleset', 'chinese');
        setFieldsDisabled(false);

        await initModalParams();
        setNetLocked(false);
        await loadNetworkPresets(9);
        el('create-model-modal').style.display = 'flex';
    }

    async function openEditModal(model) {
        modal.mode = 'edit';
        modal.model = model;
        el('model-form-title').textContent = `Settings — ${model.name}`;
        el('model-form-subtitle').style.display = 'none';
        el('btn-confirm-create').textContent = 'Save Changes';
        el('btn-confirm-create').disabled = false;
        el('model-form-warning').style.display = 'none';

        setField('new-model-name', model.name);
        setField('new-model-notes', model.notes || '');
        setField('new-model-board-size', String(model.board_size));
        setField('new-model-komi', model.komi);
        setField('new-model-ruleset', model.ruleset);
        setFieldsDisabled(false);

        await initModalParams(model.training || {});
        await loadNetworkPresets(model.board_size, (model.network || {}).size_preset);
        setNetLocked(true);
        el('create-model-modal').style.display = 'flex';
    }

    /**
     * Fork: copy the run, then change what you wanted to change — in one step.
     * Copy-then-edit was the old two-step version, and it left the fork briefly
     * existing with settings nobody wanted.
     */
    async function openForkModal(model) {
        modal.mode = 'fork';
        modal.model = model;
        el('model-form-title').textContent = `Fork — ${model.name}`;
        el('model-form-subtitle').style.display = '';
        el('model-form-subtitle').innerHTML =
            `Copies the weights, games and history at <strong>iteration ${model.iterations_logged}</strong>, ` +
            `then applies whatever you change below. The original keeps training from where it is.`;
        el('btn-confirm-create').textContent = 'Create Fork';
        el('btn-confirm-create').disabled = false;
        el('model-form-warning').style.display = 'none';

        setField('new-model-name', `${model.name} @${model.iterations_logged}`);
        setField('new-model-notes', '');
        setField('new-model-board-size', String(model.board_size));
        setField('new-model-komi', model.komi);
        setField('new-model-ruleset', model.ruleset);
        // A fork inherits its parent's weights, so board size and network are
        // fixed — changing them would make those weights meaningless.
        setFieldsDisabled(true);

        await initModalParams(model.training || {});
        await loadNetworkPresets(model.board_size, (model.network || {}).size_preset);
        setNetLocked(true);
        el('create-model-modal').style.display = 'flex';
    }

    async function submitModal() {
        const btn = el('btn-confirm-create');
        const name = el('new-model-name').value.trim();
        if (!name) { flash('Give the model a name.', 'error'); return; }

        const params = extractParamSliderValues('modal-param', state.bounds);
        const notes = el('new-model-notes').value.trim();
        const payload = {
            name,
            notes,
            board_size: parseInt(el('new-model-board-size').value, 10),
            komi: parseFloat(el('new-model-komi').value),
            ruleset: el('new-model-ruleset').value,
            ...params,
        };

        let url = '/models/api/create';
        if (modal.mode === 'edit') url = `/models/api/${encodeURIComponent(modal.model.id)}/update`;
        if (modal.mode === 'fork') url = `/models/api/${encodeURIComponent(modal.model.id)}/fork`;

        if (modal.mode === 'create' && modal.netPresets.length) {
            const idx = parseInt(el('new-model-net-size').value, 10) || 0;
            payload.network_size = (modal.netPresets[idx] || {}).key;
        }

        const original = btn.textContent;
        btn.disabled = true;
        btn.textContent = 'Working…';
        try {
            const data = await post(url, payload);
            if (data.warning) flash(data.warning, 'warn');

            // Notes live outside /update and /create, so they are set separately.
            const newId = (data.model && data.model.id) || data.id || (modal.model || {}).id;
            if (newId && modal.mode !== 'fork') await post(`/models/api/${encodeURIComponent(newId)}/meta`, { notes });
            if (newId) { state.selectedId = newId; localStorage.setItem(SELECT_KEY, newId); }

            el('create-model-modal').style.display = 'none';
            await loadFleet();
            if (modal.mode === 'fork') flash(`Forked into “${name}”.`, 'ok');
        } catch (e) {
            flash(e.message, 'error');
        } finally {
            btn.disabled = false;
            btn.textContent = original;
        }
    }

    // ---- wiring ----------------------------------------------------------

    function onRowClick(event) {
        const restore = event.target.closest('[data-act="unarchive"]');
        if (restore) {
            event.stopPropagation();
            doArchive(restore.dataset.id, false);
            return;
        }
        const row = event.target.closest('.fleet-row');
        if (row) selectModel(row.dataset.modelId);
    }

    /** Swap the config block in place — re-rendering the panel would rebuild
     *  its charts for what is only a change of which rows are listed. */
    function refreshConfigSection() {
        const model = state.byId[state.selectedId];
        const existing = document.querySelector('#fleet-detail .config-section');
        if (!model || !existing) return;
        const holder = document.createElement('div');
        holder.innerHTML = configHTML(model);
        existing.replaceWith(holder.firstElementChild);
    }

    function onPanelClick(event) {
        // Record the expand state so it survives the next re-render. The
        // `toggle` event does not bubble, so the click is what we can see.
        const summary = event.target.closest('.config-summary');
        if (summary) {
            state.configOpen = !summary.parentElement.open;
            return;
        }

        const trigger = event.target.closest('[data-act]');
        if (!trigger) return;
        const act = trigger.dataset.act;
        const model = state.byId[state.selectedId];

        if (act === 'select') { event.preventDefault(); selectModel(trigger.dataset.id); return; }
        if (act === 'config-all' || act === 'config-changed') {
            state.configShowAll = (act === 'config-all');
            refreshConfigSection();
            return;
        }
        if (act === 'exit-compare') { setCompareMode(false); return; }
        if (act === 'stop-training') { doStop(); return; }
        if (!model && act !== 'match-compare') return;

        switch (act) {
            case 'activate': doActivate(model.id); break;
            case 'train': doTrain(model.id); break;
            case 'fork': openForkModal(model); break;
            case 'edit': openEditModal(model); break;
            case 'archive': doArchive(model.id, true); break;
            case 'delete': openDeleteModal(model); break;
            case 'match': openMatchModal(model.id, trigger.dataset.id); break;
            case 'match-compare': openMatchModal(state.compareIds[0], state.compareIds[1]); break;
            default: break;
        }
    }

    function setCompareMode(on) {
        state.compareMode = on;
        state.compareIds = on && state.selectedId ? [state.selectedId] : [];
        el('btn-compare-mode').classList.toggle('active', on);
        document.querySelector('.fleet').classList.toggle('is-comparing', on);
        renderRows();
        renderRight();
    }

    document.addEventListener('DOMContentLoaded', async () => {
        if (!el('fleet-rows')) return;   // not this page

        state.bounds = await getParamBounds();
        await loadFleet();

        el('fleet-rows').addEventListener('click', onRowClick);
        el('fleet-archived').addEventListener('click', onRowClick);
        el('fleet-rows').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onRowClick(e); }
        });

        el('fleet-detail').addEventListener('click', onPanelClick);
        el('fleet-live').addEventListener('click', onPanelClick);

        el('fleet-detail').addEventListener('focusout', (e) => {
            if (e.target.id === 'detail-notes') saveNotes(state.selectedId, e.target.value.trim());
        });

        el('fleet-sort').addEventListener('click', (e) => {
            const btn = e.target.closest('.sort-btn');
            if (!btn) return;
            state.sort = btn.dataset.sort;
            localStorage.setItem(SORT_KEY, state.sort);
            renderRows();
        });

        el('btn-compare-mode').addEventListener('click', () => setCompareMode(!state.compareMode));
        el('btn-create-model').addEventListener('click', openCreateModal);
        document.body.addEventListener('click', (e) => {
            if (e.target.id === 'btn-create-model-alt') openCreateModal();
        });

        el('btn-cancel-create').addEventListener('click', () => {
            el('create-model-modal').style.display = 'none';
        });
        el('btn-confirm-create').addEventListener('click', submitModal);
        el('new-model-net-size').addEventListener('input', renderNetSize);
        el('new-model-board-size').addEventListener('change', (e) => {
            loadNetworkPresets(parseInt(e.target.value, 10) || 9);
            const warn = el('model-form-warning');
            if (modal.mode === 'edit' && modal.model && modal.model.iterations_logged > 0
                && parseInt(e.target.value, 10) !== modal.model.board_size) {
                warn.textContent = '⚠ Changing board size on a trained model discards its weights — training restarts from scratch.';
                warn.style.display = '';
            } else {
                warn.style.display = 'none';
            }
        });

        // Delete needs the name typed: the impact line says what is at stake,
        // and a one-click confirm is not proportional to 29 MB of history.
        el('delete-confirm-name').addEventListener('input', (e) => {
            el('btn-confirm-delete').disabled =
                !deleteTarget || e.target.value.trim() !== deleteTarget.name;
        });
        el('btn-cancel-delete').addEventListener('click', () => {
            el('delete-model-modal').style.display = 'none';
        });
        el('btn-confirm-delete').addEventListener('click', async () => {
            if (!deleteTarget) return;
            try {
                await post(`/models/api/${encodeURIComponent(deleteTarget.id)}/delete`, null, 'DELETE');
                el('delete-model-modal').style.display = 'none';
                if (state.selectedId === deleteTarget.id) state.selectedId = null;
                flash(`Deleted “${deleteTarget.name}”.`, 'ok');
                deleteTarget = null;
                await loadFleet();
            } catch (e) { flash(e.message, 'error'); }
        });

        el('match-games').addEventListener('input', (e) => {
            el('match-games-value').textContent = e.target.value;
        });
        el('match-sims').addEventListener('input', (e) => {
            const v = parseInt(e.target.value, 10);
            el('match-sims-value').textContent = v ? v : 'model default';
        });
        el('btn-cancel-match').addEventListener('click', () => {
            el('match-modal').style.display = 'none';
        });
        el('btn-confirm-match').addEventListener('click', startMatch);

        // Click the backdrop to dismiss any modal.
        document.querySelectorAll('.modal').forEach((m) => {
            m.addEventListener('click', (e) => { if (e.target === m) m.style.display = 'none'; });
        });

        // ---- live training ----
        // The trainer broadcasts per game; the banner follows every event, but
        // the fleet summary only refetches on the ones that change a stat.
        if (typeof socket !== 'undefined') {
            socket.on('training_update', (event) => {
                if (!event) return;
                const data = event.data || event;

                if (event.type === 'status' && data.current_stage) {
                    state.training = {
                        is_running: !!data.is_running,
                        model_id: data.is_running ? (data.active_model && data.active_model.id) : null,
                        stage_name: (data.current_stage || {}).stage_name,
                        percent: (data.current_stage || {}).percent,
                        detail: (data.current_stage || {}).detail,
                        iteration: data.iteration,
                    };
                    renderLive();
                    return;
                }

                if (data.model_id) {
                    const stage = data.current_stage || {};
                    const running = !['training_stopped', 'training_done'].includes(event.type);
                    // Whether the detail panel's buttons are still right. Only
                    // a transition matters — re-rendering the panel on every
                    // game event would rebuild its charts a few times a minute.
                    const was = `${state.training.is_running}:${state.training.model_id}`;
                    state.training = {
                        is_running: running,
                        model_id: running ? data.model_id : null,
                        stage_name: stage.stage_name,
                        percent: stage.percent,
                        detail: stage.detail,
                        iteration: data.iteration,
                    };
                    // Keep the row's headline numbers moving between refetches.
                    const model = state.byId[data.model_id];
                    if (model) {
                        if (data.elo !== undefined) model.elo = data.elo;
                        if (data.iteration !== undefined) model.iteration = data.iteration;
                        model.is_training = running;
                    }
                    renderLive();
                    renderRows();
                    if (was !== `${state.training.is_running}:${state.training.model_id}`) {
                        renderRight();
                    }
                }

                if (['iteration_done', 'training_stopped', 'training_done', 'gate_promoted'].includes(event.type)) {
                    scheduleRefresh();
                }
            });
            socket.emit('request_status');
        }
    });
})();
