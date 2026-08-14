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
