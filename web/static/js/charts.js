/**
 * charts.js — Chart.js graph initialization for the training dashboard.
 *
 * Creates the line charts:
 * 1. Elo rating over iterations
 * 2. Policy loss over iterations
 * 3. Value loss over iterations
 * (plus the diverging self-play / vs-random margin bar charts)
 *
 * Charts are updated in real-time via training.js pushing new data points.
 */

// Shared chart styling for dark theme
const chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: { labels: { color: '#9a9a9a', font: { family: 'Inter', size: 11 } } },
    },
    scales: {
        x: {
            ticks: { color: '#5a5a5a', font: { size: 10 } },
            grid: { color: 'rgba(255,255,255,0.04)' },
        },
        y: {
            ticks: { color: '#5a5a5a', font: { size: 10 } },
            grid: { color: 'rgba(255,255,255,0.04)' },
        },
    },
};

// ---- Elo Chart ----
const eloChart = new Chart(document.getElementById('elo-chart'), {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Elo Rating',
            data: [],
            borderColor: '#c8956c',
            backgroundColor: 'rgba(200, 149, 108, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.3,
            pointRadius: 2,
        }],
    },
    options: {
        ...chartDefaults,
        scales: {
            ...chartDefaults.scales,
            y: { ...chartDefaults.scales.y, suggestedMin: 400 },
        },
    },
});

// ---- Policy Loss Chart ----
// Policy and value losses live on very different scales, so each gets its own
// auto-scaled chart instead of being overlaid on one axis.
const policyLossChart = new Chart(document.getElementById('policy-loss-chart'), {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Policy Loss',
            data: [],
            borderColor: '#ef5350',
            backgroundColor: 'rgba(239, 83, 80, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.3,
            pointRadius: 1,
        }],
    },
    options: chartDefaults,
});

// ---- Value Loss Chart ----
const valueLossChart = new Chart(document.getElementById('value-loss-chart'), {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Value Loss',
            data: [],
            borderColor: '#42a5f5',
            backgroundColor: 'rgba(66, 165, 245, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.3,
            pointRadius: 1,
        }],
    },
    options: chartDefaults,
});

// ---- Diverging margin charts (0-centered bar charts) ----
// Positive bars = first color/side winning, negative = second side winning.
function makeMarginChart(canvasId, posColor, negColor) {
    return new Chart(document.getElementById(canvasId), {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                data: [],
                backgroundColor: [],
                borderWidth: 0,
                categoryPercentage: 0.9,
                barPercentage: 0.95,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title: (items) => `Game ${items[0].dataIndex + 1}`,
                        label: (item) => {
                            const v = item.raw;
                            if (v === 0) return 'Draw';
                            return `Margin: ${Math.abs(v)} pts`;
                        },
                    },
                },
            },
            scales: {
                x: { display: false, grid: { display: false } },
                y: {
                    ticks: {
                        color: '#5a5a5a',
                        font: { size: 10 },
                        callback: (v) => Math.abs(v),
                    },
                    grid: {
                        color: (ctx) => ctx.tick.value === 0 ? 'rgba(255,255,255,0.25)' : 'rgba(255,255,255,0.04)',
                        lineWidth: (ctx) => ctx.tick.value === 0 ? 1.5 : 1,
                    },
                },
            },
            _posColor: posColor,
            _negColor: negColor,
        },
    });
}

const selfplayMarginChart = makeMarginChart('selfplay-margin-chart', '#3a3a3e', '#e8e8e2');
const randomMarginChart = makeMarginChart('random-margin-chart', '#4caf50', '#ef5350');

/** Replace a margin chart's series with a fresh signed-margin array. */
function setMarginSeries(chart, series, posColor, negColor) {
    chart.data.labels = series.map((_, i) => i + 1);
    chart.data.datasets[0].data = series;
    chart.data.datasets[0].backgroundColor = series.map(v => v >= 0 ? posColor : negColor);
    chart.update('none');
}

function updateSelfPlayMargins(series) {
    // On the dark theme a literal-black bar is invisible, so "black winning"
    // (up) uses a visible cool slate and "white winning" (down) an off-white.
    // Direction + the axis labels carry the black/white meaning.
    setMarginSeries(selfplayMarginChart, series, '#8f9bb3', '#e8e8e2');
}

function updateRandomMargins(series) {
    setMarginSeries(randomMarginChart, series, '#4caf50', '#ef5350');
}

/**
 * Champion lineage chart — bars are each iteration's gate score against the
 * reigning champion, the line is the champion's cumulative self-referential
 * Elo. This is the progress metric that matters: win-rate-vs-random saturates
 * at 100% and stays there even while the model degrades, whereas every gate
 * match is a fresh head-to-head against the previous best.
 */
const gateChart = (() => {
    const el = document.getElementById('gate-chart');
    if (!el) return null;
    return new Chart(el, {
        data: {
            labels: [],
            datasets: [
                {
                    type: 'bar',
                    label: 'Gate score',
                    yAxisID: 'y',
                    data: [],
                    backgroundColor: [],
                    borderWidth: 0,
                    order: 2,
                },
                {
                    type: 'line',
                    label: 'Champion strength',
                    yAxisID: 'y1',
                    data: [],
                    borderColor: '#c8956c',
                    backgroundColor: 'rgba(200, 149, 108, 0.10)',
                    borderWidth: 2,
                    fill: true,
                    stepped: true,      // strength is constant until a promotion
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    order: 1,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title: (items) => `Iteration ${items[0].label}`,
                        label: (item) => {
                            if (item.datasetIndex === 0) {
                                const p = gateChart._promoted?.[item.dataIndex];
                                return `Gate: ${(item.raw * 100).toFixed(0)}%  ${p ? 'promoted' : 'rejected'}`;
                            }
                            const d = gateChart._delta?.[item.dataIndex] || 0;
                            return `Strength: ${Number(item.raw).toFixed(0)} Elo` + (d ? ` (+${d.toFixed(0)})` : '');
                        },
                    },
                },
                annotation: undefined,
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.06)' },
                    ticks: { color: '#9a9a9a', maxTicksLimit: 14 },
                },
                y: {
                    position: 'left',
                    min: 0,
                    max: 1,
                    title: { display: true, text: 'Gate score', color: '#9a9a9a' },
                    grid: { color: 'rgba(255,255,255,0.06)' },
                    ticks: { color: '#9a9a9a', stepSize: 0.25, callback: v => `${v * 100}%` },
                },
                y1: {
                    position: 'right',
                    title: { display: true, text: 'Champion Elo', color: '#c8956c' },
                    grid: { drawOnChartArea: false },
                    ticks: { color: '#c8956c' },
                },
            },
        },
        plugins: [{
            // Dashed promotion-threshold rule drawn straight onto the canvas,
            // so no external annotation plugin is needed.
            id: 'gateThreshold',
            afterDatasetsDraw(chart) {
                const t = chart.$gateThreshold;
                if (t == null) return;
                const { ctx, chartArea, scales } = chart;
                const y = scales.y.getPixelForValue(t);
                ctx.save();
                ctx.setLineDash([5, 4]);
                ctx.strokeStyle = 'rgba(255,255,255,0.45)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(chartArea.left, y);
                ctx.lineTo(chartArea.right, y);
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.fillStyle = 'rgba(255,255,255,0.6)';
                ctx.font = '10px Inter, sans-serif';
                ctx.textAlign = 'right';
                ctx.fillText(`promote ≥ ${Math.round(t * 100)}%`, chartArea.right - 4, y - 4);
                ctx.restore();
            },
        }],
    });
})();

/** Replace the champion-lineage chart's data from /api/gate_history points. */
function updateGateChart(points, threshold) {
    if (!gateChart) return;
    gateChart.data.labels = points.map(p => p.iteration);
    gateChart.data.datasets[0].data = points.map(p => p.gate_win_rate);
    gateChart.data.datasets[0].backgroundColor = points.map(
        p => (p.promoted ? 'rgba(76, 175, 80, 0.75)' : 'rgba(239, 83, 80, 0.55)')
    );
    gateChart.data.datasets[1].data = points.map(p => p.gate_elo);
    gateChart._promoted = points.map(p => p.promoted);
    gateChart._delta = points.map(p => p.elo_delta);
    gateChart.$gateThreshold = threshold ?? 0.55;
    gateChart.update('none');
}

const MAX_CHART_POINTS = 100;

/**
 * Insert-or-replace one point on an iteration-indexed chart.
 *
 * Charts used to blindly `push()`, which meant the series had no identity: the
 * same iteration arriving twice (socket redelivery), a model switch, or a
 * server restart with the page left open would concatenate a SECOND run onto
 * the first — the data appeared doubled with a hard jump at the seam. Keying
 * every point by iteration number makes the operation idempotent, so replaying
 * the same metrics is now a no-op instead of a duplicate.
 */
function upsertPoint(chart, iteration, value) {
    if (value === undefined || value === null) return;

    const label = `Iter ${iteration}`;
    const labels = chart.data.labels;
    const data = chart.data.datasets[0].data;

    const existing = labels.indexOf(label);
    if (existing !== -1) {
        data[existing] = value;   // same iteration re-reported — replace in place
        return;
    }

    // Keep the series ordered by iteration even if events arrive out of order.
    let at = labels.length;
    while (at > 0 && chartIterationOf(labels[at - 1]) > iteration) at--;

    labels.splice(at, 0, label);
    data.splice(at, 0, value);

    while (labels.length > MAX_CHART_POINTS) {
        labels.shift();
        data.shift();
    }
}

function chartIterationOf(label) {
    const n = parseInt(String(label).replace(/^Iter\s*/, ''), 10);
    return Number.isNaN(n) ? -1 : n;
}

/** Drop every point from the iteration-indexed charts (e.g. on model switch). */
function resetCharts() {
    [eloChart, policyLossChart, valueLossChart].forEach(c => {
        c.data.labels.length = 0;
        c.data.datasets[0].data.length = 0;
        c.update('none');
    });
}

/** Add (or refresh) a data point on all charts from a metrics object. */
function updateCharts(metrics) {
    const iteration = metrics.iteration;
    if (iteration === undefined || iteration === null) return;

    upsertPoint(eloChart, iteration, metrics.elo);
    eloChart.update('none');

    // Policy & value loss (separate charts, each auto-scaled)
    if (metrics.policy_loss !== undefined) {
        upsertPoint(policyLossChart, iteration, metrics.policy_loss);
        upsertPoint(valueLossChart, iteration, metrics.value_loss);
        policyLossChart.update('none');
        valueLossChart.update('none');
    }
}

// ---- Time Breakdown Chart ----
const timeBreakdownChart = (() => {
    const el = document.getElementById('time-breakdown-chart');
    if (!el) return null;

    return new Chart(el, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Self',
                    key: 'self_play',
                    data: [],
                    borderColor: '#4fc3f7',
                    backgroundColor: 'rgba(79, 195, 247, 0.15)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 3,
                },
                {
                    label: 'Training',
                    key: 'nn_train',
                    data: [],
                    borderColor: '#ab47bc',
                    backgroundColor: 'rgba(171, 71, 188, 0.15)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 3,
                },
                {
                    label: 'Random',
                    key: 'random_eval',
                    data: [],
                    borderColor: '#66bb6a',
                    backgroundColor: 'rgba(102, 187, 106, 0.15)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 3,
                },
                {
                    label: 'Champion',
                    key: 'champion_gate',
                    data: [],
                    borderColor: '#ffa726',
                    backgroundColor: 'rgba(255, 167, 38, 0.15)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 3,
                },
                {
                    label: 'Total',
                    key: 'total',
                    data: [],
                    borderColor: '#f5f5f5',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.3,
                    pointRadius: 3,
                },
            ],
        },
        options: {
            ...chartDefaults,
            plugins: {
                ...chartDefaults.plugins,
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title: (items) => `Iteration ${items[0].label}`,
                        label: (item) => {
                            const val = item.raw;
                            if (val === null || val === undefined) return `${item.dataset.label}: —`;
                            let timeStr = `${val.toFixed(1)}s`;
                            if (val >= 60) {
                                const m = Math.floor(val / 60);
                                const s = (val % 60).toFixed(0);
                                timeStr += ` (${m}m ${s}s)`;
                            }
                            return `${item.dataset.label}: ${timeStr}`;
                        },
                    },
                },
            },
            scales: {
                ...chartDefaults.scales,
                y: {
                    ...chartDefaults.scales.y,
                    title: { display: true, text: 'Seconds', color: '#9a9a9a', font: { size: 11 } },
                    ticks: {
                        color: '#9a9a9a',
                        callback: (v) => v >= 60 ? `${(v/60).toFixed(1)}m` : `${v}s`,
                    },
                },
            },
        },
    });
})();

/** Update Time Breakdown Chart data & filter visibility. */
function updateTimeBreakdownChart(history, activeFilters = {}) {
    if (!timeBreakdownChart || !Array.isArray(history)) return;

    const labels = history.map(h => h.iteration);

    const dsData = {
        self_play: history.map(h => h.self_play_time ?? 0),
        nn_train: history.map(h => h.nn_train_time ?? 0),
        random_eval: history.map(h => h.random_eval_time ?? 0),
        champion_gate: history.map(h => h.champion_gate_time ?? 0),
        total: history.map(h => h.total_time ?? 0),
    };

    timeBreakdownChart.data.labels = labels;

    timeBreakdownChart.data.datasets.forEach(ds => {
        const key = ds.key;
        ds.data = dsData[key] || [];

        // If key is present in activeFilters dictionary and is false, hide dataset
        if (activeFilters && typeof activeFilters === 'object') {
            if (activeFilters[key] === false) {
                ds.hidden = true;
            } else {
                ds.hidden = false;
            }
        } else {
            ds.hidden = false;
        }
    });

    timeBreakdownChart.update('none');
}
