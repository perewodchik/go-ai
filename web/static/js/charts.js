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
const eloChart = (() => {
    const el = document.getElementById('elo-chart');
    if (!el) return null;
    return new Chart(el, {
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
})();

// ---- Policy Loss Chart ----
// Policy and value losses live on very different scales, so each gets its own
// auto-scaled chart instead of being overlaid on one axis.
const policyLossChart = (() => {
    const el = document.getElementById('policy-loss-chart');
    if (!el) return null;
    return new Chart(el, {
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
})();

// ---- Value Loss Chart ----
const valueLossChart = (() => {
    const el = document.getElementById('value-loss-chart');
    if (!el) return null;
    return new Chart(el, {
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
})();

// ---- Diverging margin charts (0-centered bar charts) ----
// Positive bars = first color/side winning, negative = second side winning.
function makeMarginChart(canvasId, posColor, negColor) {
    const el = document.getElementById(canvasId);
    if (!el) return null;
    return new Chart(el, {
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

// ---- Margin Dispersion & Spread Chart ----
// Visualizes rolling win margin dispersion, moving standard deviation (±1σ band),
// and victory margin center (μ).
const marginDispersionChart = (() => {
    const el = document.getElementById('margin-dispersion-chart');
    if (!el) return null;
    return new Chart(el, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Upper (+1σ)',
                    data: [],
                    borderColor: 'rgba(200, 149, 108, 0.3)',
                    borderWidth: 1,
                    pointRadius: 0,
                    pointHoverRadius: 0,
                    fill: '+1',
                    backgroundColor: 'rgba(200, 149, 108, 0.12)',
                    tension: 0.35,
                    order: 3,
                },
                {
                    label: 'Lower (-1σ)',
                    data: [],
                    borderColor: 'rgba(200, 149, 108, 0.3)',
                    borderWidth: 1,
                    pointRadius: 0,
                    pointHoverRadius: 0,
                    fill: false,
                    tension: 0.35,
                    order: 3,
                },
                {
                    label: 'Avg Margin (μ)',
                    data: [],
                    borderColor: '#c8956c',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    fill: false,
                    tension: 0.35,
                    order: 1,
                },
                {
                    label: 'Std Dev (σ)',
                    data: [],
                    borderColor: '#ef5350',
                    backgroundColor: 'transparent',
                    borderWidth: 1.5,
                    borderDash: [4, 3],
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    fill: false,
                    tension: 0.35,
                    order: 2,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    align: 'end',
                    labels: {
                        boxWidth: 12,
                        boxHeight: 2,
                        color: '#9a9a9a',
                        font: { family: 'Inter', size: 10 },
                        filter: (item) => item.text === 'Avg Margin (μ)' || item.text === 'Std Dev (σ)',
                    },
                },
                tooltip: {
                    callbacks: {
                        title: (items) => `Game ${items[0].label}`,
                        label: (item) => {
                            const val = item.raw;
                            if (val === null || val === undefined || isNaN(val)) return null;
                            if (item.datasetIndex === 2) {
                                const pt = marginDispersionChart._stats?.[item.dataIndex];
                                const raw = pt ? ` (Game: ${pt.raw.toFixed(1)} pts ${pt.winner})` : '';
                                return `Avg Margin (μ): ${val.toFixed(1)} pts${raw}`;
                            }
                            if (item.datasetIndex === 3) {
                                return `Std Dev (σ): ±${val.toFixed(1)} pts`;
                            }
                            if (item.datasetIndex === 0) {
                                const pt = marginDispersionChart._stats?.[item.dataIndex];
                                if (!pt) return null;
                                return `±1σ Spread: ${pt.lower.toFixed(1)} – ${pt.upper.toFixed(1)} pts`;
                            }
                            return null;
                        },
                    },
                },
            },
            scales: {
                x: {
                    display: false,
                    grid: { display: false },
                },
                y: {
                    min: 0,
                    ticks: {
                        color: '#5a5a5a',
                        font: { size: 10 },
                        callback: (v) => `${v}p`,
                    },
                    grid: {
                        color: 'rgba(255,255,255,0.04)',
                    },
                },
            },
        },
    });
})();

/** Update the margin dispersion chart and its statistical readout chips. */
function updateMarginDispersion(series) {
    const el = (id) => document.getElementById(id);
    if (!Array.isArray(series) || series.length === 0) {
        if (el('disp-std-val')) el('disp-std-val').textContent = '—';
        if (el('disp-mean-val')) el('disp-mean-val').textContent = '—';
        if (el('disp-close-val')) el('disp-close-val').textContent = '—';
        if (el('disp-blowout-val')) el('disp-blowout-val').textContent = '—';
        if (marginDispersionChart) {
            marginDispersionChart.data.labels = [];
            marginDispersionChart.data.datasets.forEach(ds => ds.data = []);
            marginDispersionChart.update('none');
        }
        return;
    }

    const n = series.length;
    const absSeries = series.map(v => Math.abs(v));

    // Summary statistics over the full slice
    const totalSum = absSeries.reduce((a, b) => a + b, 0);
    const meanTot = totalSum / n;
    const varTot = absSeries.reduce((a, b) => a + (b - meanTot) ** 2, 0) / n;
    const stdTot = Math.sqrt(varTot);

    const closeCount = absSeries.filter(v => v <= 10).length;
    const blowoutCount = absSeries.filter(v => v >= 40).length;
    const closePct = Math.round((closeCount / n) * 100);
    const blowoutPct = Math.round((blowoutCount / n) * 100);

    if (el('disp-std-val')) el('disp-std-val').textContent = `±${stdTot.toFixed(1)} pts`;
    if (el('disp-mean-val')) el('disp-mean-val').textContent = `${meanTot.toFixed(1)} pts`;
    if (el('disp-close-val')) el('disp-close-val').textContent = `${closePct}% (${closeCount})`;
    if (el('disp-blowout-val')) el('disp-blowout-val').textContent = `${blowoutPct}% (${blowoutCount})`;

    if (!marginDispersionChart) return;

    // Adaptive moving window size: 15 for normal size, smaller if few games
    const W = Math.max(3, Math.min(15, Math.floor(n / 3) || n));

    const labels = [];
    const upperData = [];
    const lowerData = [];
    const meanData = [];
    const stdData = [];
    const statsArr = [];

    for (let i = 0; i < n; i++) {
        labels.push(i + 1);
        const winStart = Math.max(0, i - W + 1);
        const windowSlice = absSeries.slice(winStart, i + 1);
        const wLen = windowSlice.length;
        const wMean = windowSlice.reduce((a, b) => a + b, 0) / wLen;
        const wVar = windowSlice.reduce((a, b) => a + (b - wMean) ** 2, 0) / wLen;
        const wStd = Math.sqrt(wVar);
        const upper = wMean + wStd;
        const lower = Math.max(0, wMean - wStd);

        const rawVal = series[i];
        const winner = rawVal > 0 ? '⚫ Black' : (rawVal < 0 ? '⚪ White' : 'Draw');

        upperData.push(round1(upper));
        lowerData.push(round1(lower));
        meanData.push(round1(wMean));
        stdData.push(round1(wStd));

        statsArr.push({
            mean: wMean,
            std: wStd,
            upper: upper,
            lower: lower,
            raw: Math.abs(rawVal),
            winner: winner,
        });
    }

    marginDispersionChart.data.labels = labels;
    marginDispersionChart.data.datasets[0].data = upperData;
    marginDispersionChart.data.datasets[1].data = lowerData;
    marginDispersionChart.data.datasets[2].data = meanData;
    marginDispersionChart.data.datasets[3].data = stdData;
    marginDispersionChart._stats = statsArr;
    marginDispersionChart.update('none');
}

function round1(v) {
    return Math.round(v * 10) / 10;
}

/** Replace a margin chart's series with a fresh signed-margin array. */
function setMarginSeries(chart, series, posColor, negColor) {
    if (!chart) return;
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
 * reigning champion.
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
                    data: [],
                    backgroundColor: [],
                    borderWidth: 0,
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
                            const p = gateChart._promoted?.[item.dataIndex];
                            return `Gate: ${(item.raw * 100).toFixed(0)}%  ${p ? 'promoted' : 'rejected'}`;
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
    gateChart._promoted = points.map(p => p.promoted);
    gateChart.$gateThreshold = threshold ?? 0.55;
    gateChart.update('none');
}

/**
 * Mercy-rule chart — is resignation paying for itself, or throwing games away?
 *
 * Bars (left axis) are how often the rule fired that iteration: the benefit.
 * The line (right axis) is the CUMULATIVE share of checked resignations that
 * turned out to be wrong: the cost. It has to be cumulative — with a handful of
 * playout checks per iteration, a per-iteration rate is 0% or 100% and tells
 * you nothing. The shaded band is the Wilson confidence interval, which is what
 * makes "we don't know yet" visually distinct from "it's fine": early on the
 * band covers most of the axis, and it narrows as evidence accumulates.
 *
 * The dashed rule is the 5% danger line. What matters is not whether the line
 * is under it, but whether the BAND is.
 */
const resignChart = (() => {
    const el = document.getElementById('resign-chart');
    if (!el) return null;
    return new Chart(el, {
        data: {
            labels: [],
            datasets: [
                {
                    type: 'bar',
                    label: 'Games resigned',
                    yAxisID: 'y',
                    data: [],
                    backgroundColor: 'rgba(120, 144, 156, 0.55)',
                    borderWidth: 0,
                    order: 3,
                },
                {
                    type: 'line',
                    label: 'Wrong-resignation rate (cumulative)',
                    yAxisID: 'y1',
                    data: [],
                    borderColor: '#ef5350',
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    spanGaps: true,
                    order: 1,
                },
                {
                    // Upper edge of the interval, filled down to the lower edge.
                    type: 'line',
                    label: 'ci-high',
                    yAxisID: 'y1',
                    data: [],
                    borderColor: 'rgba(239, 83, 80, 0.25)',
                    borderWidth: 1,
                    pointRadius: 0,
                    fill: '+1',
                    backgroundColor: 'rgba(239, 83, 80, 0.12)',
                    spanGaps: true,
                    order: 2,
                },
                {
                    type: 'line',
                    label: 'ci-low',
                    yAxisID: 'y1',
                    data: [],
                    borderColor: 'rgba(239, 83, 80, 0.25)',
                    borderWidth: 1,
                    pointRadius: 0,
                    spanGaps: true,
                    order: 2,
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
                    filter: (item) => item.datasetIndex < 2,
                    callbacks: {
                        title: (items) => `Iteration ${items[0].label}`,
                        label: (item) => {
                            const i = item.dataIndex;
                            if (item.datasetIndex === 0) {
                                const p = resignChart._points?.[i] || {};
                                return `Resigned ${p.resigned}/${p.games} games`;
                            }
                            const p = resignChart._points?.[i] || {};
                            if (p.cum_false_rate == null) return 'No checks yet';
                            return `Wrong: ${(p.cum_false_rate * 100).toFixed(0)}%`
                                + ` (${p.cum_checked} check${p.cum_checked === 1 ? '' : 's'},`
                                + ` up to ${(p.cum_ci_high * 100).toFixed(0)}%)`;
                        },
                    },
                },
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
                    title: { display: true, text: 'Games resigned', color: '#9a9a9a' },
                    grid: { color: 'rgba(255,255,255,0.06)' },
                    ticks: { color: '#9a9a9a', stepSize: 0.25, callback: v => `${v * 100}%` },
                },
                y1: {
                    position: 'right',
                    min: 0,
                    max: 1,
                    title: { display: true, text: 'Wrong resignations', color: '#ef5350' },
                    grid: { drawOnChartArea: false },
                    ticks: { color: '#ef5350', stepSize: 0.25, callback: v => `${v * 100}%` },
                },
            },
        },
        plugins: [{
            id: 'resignDanger',
            afterDatasetsDraw(chart) {
                const t = chart.$dangerRate;
                if (t == null) return;
                const { ctx, chartArea, scales } = chart;
                const y = scales.y1.getPixelForValue(t);
                ctx.save();
                ctx.setLineDash([5, 4]);
                ctx.strokeStyle = 'rgba(239, 83, 80, 0.55)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(chartArea.left, y);
                ctx.lineTo(chartArea.right, y);
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.fillStyle = 'rgba(239, 83, 80, 0.75)';
                ctx.font = '10px Inter, sans-serif';
                ctx.textAlign = 'right';
                ctx.fillText(`danger ≥ ${Math.round(t * 100)}%`, chartArea.right - 4, y - 4);
                ctx.restore();
            },
        }],
    });
})();

/** Replace the mercy-rule chart's data from /api/resign_stats. */
function updateResignChart(points, dangerRate) {
    if (!resignChart) return;
    resignChart.data.labels = points.map(p => p.iteration);
    resignChart.data.datasets[0].data = points.map(p => p.resign_rate);
    resignChart.data.datasets[1].data = points.map(p => p.cum_false_rate);
    resignChart.data.datasets[2].data = points.map(p => p.cum_ci_high);
    resignChart.data.datasets[3].data = points.map(p => p.cum_ci_low);
    resignChart._points = points;
    resignChart.$dangerRate = dangerRate ?? 0.05;
    // The card starts hidden — a model that has never used the mercy rule
    // should carry no dead UI — so Chart.js sized this canvas 0x0 at
    // construction and will happily keep drawing into nothing after the card
    // is revealed. Resize first, exactly as the collapsible blocks do.
    resignChart.resize();
    resignChart.update('none');
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
        if (c) {
            c.data.labels.length = 0;
            c.data.datasets[0].data.length = 0;
            c.update('none');
        }
    });
}

/** Add (or refresh) a data point on all charts from a metrics object. */
function updateCharts(metrics) {
    const iteration = metrics.iteration;
    if (iteration === undefined || iteration === null) return;

    if (eloChart) {
        upsertPoint(eloChart, iteration, metrics.elo);
        eloChart.update('none');
    }

    // Policy & value loss (separate charts, each auto-scaled)
    if (metrics.policy_loss !== undefined) {
        if (policyLossChart) {
            upsertPoint(policyLossChart, iteration, metrics.policy_loss);
            policyLossChart.update('none');
        }
        if (valueLossChart) {
            upsertPoint(valueLossChart, iteration, metrics.value_loss);
            valueLossChart.update('none');
        }
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
