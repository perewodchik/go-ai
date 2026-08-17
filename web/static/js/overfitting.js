/*
 * overfitting.js — renders /overfitting from /models/api/<id>/overfitting.
 *
 * Every chart here draws nulls as GAPS, never as zeros. That matters more than
 * usual on this page: a model with 185 iterations of history and 3 iterations of
 * entropy data must look like exactly that, or the picture invents a collapse
 * from nothing (or hides one).
 */

const OF = {
    charts: {},
    // Palette borrowed from the CSS custom properties so the page stays themed.
    colors: {
        internal: '#ce93d8',
        external: '#4fc3f7',
        label: '#ffa726',
        network: '#66bb6a',
        gate: '#dbb08a',
        danger: '#ef5350',
        muted: 'rgba(255,255,255,0.25)',
    },
};

const pct = (v) => (v === null || v === undefined ? '—' : `${(v * 100).toFixed(0)}%`);
const nats = (v) => (v === null || v === undefined ? '—' : v.toFixed(2));

function destroyChart(key) {
    if (OF.charts[key]) {
        OF.charts[key].destroy();
        delete OF.charts[key];
    }
}

const BASE_OPTS = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
        legend: { labels: { color: '#9a9a9a', boxWidth: 12, font: { size: 11 } } },
    },
    scales: {
        x: { ticks: { color: '#5a5a5a', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
        y: { ticks: { color: '#5a5a5a', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
    },
};

function opts(extra) {
    return JSON.parse(JSON.stringify(BASE_OPTS)) && Object.assign(
        JSON.parse(JSON.stringify(BASE_OPTS)), extra || {});
}

// --- Findings ---------------------------------------------------------------

const ICONS = { critical: '🔴', warning: '🟠', ok: '🟢' };

function renderFindings(findings) {
    const host = document.getElementById('of-findings');
    if (!findings || !findings.length) {
        host.innerHTML = '<div class="of-empty">Not enough recorded data to judge this model yet. '
            + 'Play some matches against external opponents, or run a few training iterations, '
            + 'and the probes will have something to read.</div>';
        return;
    }
    host.innerHTML = findings.map((f) => `
        <div class="of-finding ${f.severity}">
            <div class="of-finding-icon">${ICONS[f.severity] || '•'}</div>
            <div>
                <div class="of-finding-title">${escapeHtml(f.title)}</div>
                <div class="of-finding-detail">${escapeHtml(f.detail)}</div>
            </div>
        </div>`).join('');
}

function escapeHtml(s) {
    const d = document.createElement('div');
    d.textContent = s === null || s === undefined ? '' : String(s);
    return d.innerHTML;
}

// --- Headline metrics -------------------------------------------------------

function metricCard(label, value, cls, hint) {
    const shown = value === null || value === undefined ? '—' : value;
    const klass = value === null || value === undefined ? 'na' : (cls || '');
    return `<div class="of-metric">
        <div class="of-metric-label">${escapeHtml(label)}</div>
        <div class="of-metric-value ${klass}">${escapeHtml(shown)}</div>
        <div class="of-metric-hint">${escapeHtml(hint)}</div>
    </div>`;
}

function renderMetrics(data) {
    const labels = data.labels || {};
    const net = data.network || {};
    const totals = (data.generalization || {}).totals || {};
    const cards = [];

    // Thresholds mirror overfit_stats._findings so the colour and the prose
    // never disagree with each other.
    if (labels.available) {
        cards.push(metricCard('One-hot labels', pct(labels.one_hot_frac),
            labels.one_hot_frac >= 0.25 ? 'bad' : 'good',
            'Share of training labels naming a single move. These teach the policy head only its own argmax.'));
        cards.push(metricCard('Label entropy', nats(labels.entropy),
            labels.entropy < 0.55 ? 'bad' : 'good',
            `nats, over a support of ${labels.support ?? '—'} moves. ~0.95 is healthy on 9x9 here.`));
    } else {
        cards.push(metricCard('Label entropy', null, null,
            labels.reason || 'No replay buffer to read.'));
    }

    if (net.available) {
        cards.push(metricCard('Moves considered', net.effective_moves,
            net.effective_moves < 4 ? 'bad' : (net.effective_moves < 6 ? 'warn' : 'good'),
            'Perplexity of the prior over legal moves — how many moves the network effectively entertains.'));
        cards.push(metricCard('Prior on top 5', pct(net.top5_mass),
            net.top5_mass >= 0.93 ? 'bad' : 'good',
            'Everything outside this set starts at a prior near zero, where search cannot recover it.'));
    } else {
        cards.push(metricCard('Moves considered', null, null,
            net.reason || 'Could not probe the network.'));
    }

    const gap = (totals.internal_rate !== null && totals.internal_rate !== undefined
                 && totals.external_rate !== null && totals.external_rate !== undefined)
        ? totals.internal_rate - totals.external_rate : null;
    cards.push(metricCard('Internal win rate', pct(totals.internal_rate), 'good',
        `${totals.internal_games || 0} games against its own lineage and the random anchor.`));
    cards.push(metricCard('External win rate', pct(totals.external_rate),
        gap !== null && gap >= 0.25 ? 'bad' : 'good',
        `${totals.external_games || 0} games against opponents from outside the loop.`));
    cards.push(metricCard('Generalization gap', gap === null ? null : pct(gap),
        gap !== null && gap >= 0.25 ? 'bad' : 'good',
        'Internal minus external. A large positive gap is overfitting to self-play, stated in wins.'));

    document.getElementById('of-metrics').innerHTML = cards.join('');
}

// --- Charts -----------------------------------------------------------------

function renderGapChart(gen) {
    destroyChart('gap');
    const series = (gen && gen.series) || [];
    const el = document.getElementById('of-gap-chart');
    if (!series.length) return;

    OF.charts.gap = new Chart(el, {
        type: 'line',
        data: {
            labels: series.map((s) => `iter ${s.iteration}+`),
            datasets: [
                {
                    label: 'vs own lineage (internal)',
                    data: series.map((s) => s.internal_rate),
                    borderColor: OF.colors.internal,
                    backgroundColor: 'rgba(206,147,216,0.12)',
                    spanGaps: false, tension: 0.25, pointRadius: 4, fill: false,
                },
                {
                    label: 'vs external opponents',
                    data: series.map((s) => s.external_rate),
                    borderColor: OF.colors.external,
                    backgroundColor: 'rgba(79,195,247,0.14)',
                    spanGaps: false, tension: 0.25, pointRadius: 4, fill: true,
                },
                {
                    label: 'even (50%)',
                    data: series.map(() => 0.5),
                    borderColor: OF.colors.muted, borderDash: [4, 4],
                    borderWidth: 1, pointRadius: 0, fill: false,
                },
            ],
        },
        options: opts({
            scales: Object.assign(JSON.parse(JSON.stringify(BASE_OPTS.scales)), {
                y: {
                    min: 0, max: 1,
                    ticks: {
                        color: '#5a5a5a', font: { size: 10 },
                        callback: (v) => `${Math.round(v * 100)}%`,
                    },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                },
            }),
            plugins: {
                legend: { labels: { color: '#9a9a9a', boxWidth: 12, font: { size: 11 } } },
                tooltip: {
                    callbacks: {
                        afterBody: (items) => {
                            const s = series[items[0].dataIndex];
                            return `internal: ${s.internal_games} games · external: ${s.external_games} games`;
                        },
                    },
                },
            },
        }),
    });
}

function renderEntropyChart(training) {
    destroyChart('entropy');
    const el = document.getElementById('of-entropy-chart');
    const s = (training && training.series) || {};
    const iters = (training && training.iterations) || [];

    if (!training || !training.has_entropy_data) {
        // Say so rather than drawing an empty axis: these metrics only start
        // being logged once the model trains an iteration with the new build.
        el.parentElement.innerHTML = '<div class="of-empty">No per-iteration entropy '
            + 'history yet — these metrics are recorded from the first training iteration '
            + 'run after the policy-target fix. The headline cards above are measured live '
            + 'from the current weights and replay buffer, so they work immediately.</div>';
        return;
    }

    OF.charts.entropy = new Chart(el, {
        type: 'line',
        data: {
            labels: iters,
            datasets: [
                {
                    label: 'label entropy (buffer)',
                    data: s.target_entropy || [],
                    borderColor: OF.colors.label, spanGaps: false,
                    tension: 0.2, pointRadius: 0, borderWidth: 2, fill: false,
                },
                {
                    label: 'label entropy (this iteration)',
                    data: s.iter_target_entropy || [],
                    borderColor: OF.colors.label, borderDash: [3, 3], spanGaps: false,
                    tension: 0.2, pointRadius: 0, borderWidth: 1.5, fill: false,
                },
                {
                    label: 'network policy entropy',
                    data: s.policy_entropy || [],
                    borderColor: OF.colors.network, spanGaps: false,
                    tension: 0.2, pointRadius: 0, borderWidth: 2, fill: false,
                },
            ],
        },
        options: opts({
            plugins: {
                legend: { labels: { color: '#9a9a9a', boxWidth: 12, font: { size: 11 } } },
                title: { display: true, text: 'Entropy (nats) — higher is wider', color: '#9a9a9a', font: { size: 11 } },
            },
        }),
    });
}

function renderGateChart(training) {
    destroyChart('gate');
    const el = document.getElementById('of-gate-chart');
    const s = (training && training.series) || {};
    const gate = s.gate_win_rate || [];
    if (!gate.some((v) => v !== null && v !== undefined)) {
        el.parentElement.innerHTML = '<div class="of-empty">No gate history recorded.</div>';
        return;
    }

    OF.charts.gate = new Chart(el, {
        type: 'line',
        data: {
            labels: (training && training.iterations) || [],
            datasets: [
                {
                    label: 'gate win rate',
                    data: gate,
                    borderColor: OF.colors.gate, spanGaps: false,
                    tension: 0.15, pointRadius: 0, borderWidth: 1.5, fill: false,
                },
                {
                    label: 'no signal (50%)',
                    data: gate.map(() => 0.5),
                    borderColor: OF.colors.danger, borderDash: [4, 4],
                    borderWidth: 1, pointRadius: 0, fill: false,
                },
            ],
        },
        options: opts({
            plugins: {
                legend: { labels: { color: '#9a9a9a', boxWidth: 12, font: { size: 11 } } },
                title: {
                    display: true,
                    text: 'Promotion gate — hugging 50% means candidates are indistinguishable',
                    color: '#9a9a9a', font: { size: 11 },
                },
            },
        }),
    });
}

function renderDecileChart(labels) {
    destroyChart('decile');
    const el = document.getElementById('of-decile-chart');
    const dec = (labels && labels.entropy_deciles) || [];
    if (!dec.length) {
        el.parentElement.innerHTML = `<div class="of-empty">${escapeHtml(
            (labels && labels.reason) || 'No replay buffer to profile.')}</div>`;
        return;
    }
    OF.charts.decile = new Chart(el, {
        type: 'bar',
        data: {
            labels: dec.map((_, i) => (i === 0 ? 'oldest' : (i === dec.length - 1 ? 'newest' : `${i * 10}%`))),
            datasets: [{
                label: 'label entropy (nats)',
                data: dec,
                backgroundColor: dec.map((v) => (v < 0.55 ? 'rgba(239,83,80,0.55)' : 'rgba(102,187,106,0.55)')),
                borderColor: dec.map((v) => (v < 0.55 ? '#ef5350' : '#66bb6a')),
                borderWidth: 1,
            }],
        },
        options: opts({
            plugins: {
                legend: { display: false },
                title: {
                    display: true,
                    text: 'Buffer sliced oldest → newest. Below 0.55 nats (red) the labels are near-deterministic.',
                    color: '#9a9a9a', font: { size: 11 },
                },
            },
        }),
    });
}

function renderOpponents(gen) {
    const host = document.getElementById('of-opponents');
    const rows = (gen && gen.by_opponent) || [];
    if (!rows.length) {
        host.innerHTML = '<tr><td colspan="6" class="of-muted">No games against external opponents recorded.</td></tr>';
        return;
    }
    host.innerHTML = rows.map((c) => {
        let delta = '<span class="of-muted">—</span>';
        if (c.delta !== null && c.delta !== undefined) {
            const cls = c.delta < 0 ? 'of-delta-down' : 'of-delta-up';
            delta = `<span class="${cls}">${c.delta > 0 ? '+' : ''}${(c.delta * 100).toFixed(0)}%</span>`;
        }
        const cell = (rate, n) => (rate === null || rate === undefined
            ? '<span class="of-muted">—</span>'
            : `${(rate * 100).toFixed(0)}% <span class="of-muted">(${n})</span>`);
        return `<tr>
            <td>${escapeHtml(c.opponent)}</td>
            <td class="num">${cell(c.early_rate, c.early_n)}</td>
            <td class="num">${cell(c.late_rate, c.late_n)}</td>
            <td class="num">${delta}</td>
            <td class="num">${c.games}</td>
            <td class="num of-muted">${c.first_iteration}–${c.last_iteration}</td>
        </tr>`;
    }).join('');
}

// --- Load -------------------------------------------------------------------

async function load(modelId) {
    const loading = document.getElementById('of-loading');
    const body = document.getElementById('of-body');
    const error = document.getElementById('of-error');
    loading.style.display = 'block';
    body.style.display = 'none';
    error.style.display = 'none';

    try {
        const res = await fetch(`/models/api/${encodeURIComponent(modelId)}/overfitting`);
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);

        renderFindings(data.findings);
        renderMetrics(data);
        renderGapChart(data.generalization);
        renderOpponents(data.generalization);
        renderEntropyChart(data.training);
        renderGateChart(data.training);
        renderDecileChart(data.labels);

        loading.style.display = 'none';
        body.style.display = 'block';
    } catch (err) {
        loading.style.display = 'none';
        error.textContent = `Could not load diagnostics: ${err.message}`;
        error.style.display = 'block';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const select = document.getElementById('of-model');
    const refresh = document.getElementById('of-refresh');
    if (!select) return;

    // Re-rendering into a canvas that already holds a chart leaves the old one
    // alive and leaking, and each reload replaces the wrapper's innerHTML in the
    // no-data paths — so a full page navigation is the honest way to switch.
    select.addEventListener('change', () => {
        window.location.search = `?model=${encodeURIComponent(select.value)}`;
    });
    if (refresh) refresh.addEventListener('click', () => window.location.reload());

    if (select.value) load(select.value);
});
