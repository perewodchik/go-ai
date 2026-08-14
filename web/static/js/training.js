/**
 * training.js — Training dashboard real-time updates.
 *
 * Connects via SocketIO for live training events.
 * Manages: start/stop buttons, log feed, game browser,
 * learning stats, milestone toasts.
 */

let replayBoard = null;
let replayData = null;
let replayMoveIndex = 0;

// ---- Training Controls ----
const btnStart = document.getElementById('btn-start-training');
if (btnStart) {
    btnStart.addEventListener('click', async () => {
        btnStart.disabled = true;
        btnStart.textContent = '⏳ Starting...';
        try {
            if (typeof socket !== 'undefined' && socket && socket.connected) {
                socket.emit('start_training');
            } else {
                const res = await fetch('/training/api/start', { method: 'POST' });
                const data = await res.json();
                if (!res.ok || data.error) {
                    showToast('⚠ ' + (data.error || 'Failed to start training'));
                    btnStart.disabled = false;
                    btnStart.textContent = '▶ Start Training';
                }
            }
        } catch (e) {
            showToast('⚠ Error starting training');
            btnStart.disabled = false;
            btnStart.textContent = '▶ Start Training';
        }
    });
}

const btnStop = document.getElementById('btn-stop-training');
if (btnStop) {
    btnStop.addEventListener('click', async () => {
        btnStop.disabled = true;
        btnStop.textContent = '⏳ Stopping...';
        if (typeof socket !== 'undefined' && socket && socket.connected) {
            socket.emit('stop_training');
        } else {
            await fetch('/training/api/stop', { method: 'POST' });
        }
    });
}

const btnForceStop = document.getElementById('btn-force-stop');
if (btnForceStop) {
    btnForceStop.addEventListener('click', async () => {
        if (confirm('Are you sure you want to force stop training? Immediate stop will discard current iteration progress and restore weights from the last completed iteration.')) {
            btnForceStop.disabled = true;
            btnForceStop.textContent = '⚡ Halting...';
            if (typeof socket !== 'undefined' && socket && socket.connected) {
                socket.emit('force_stop_training');
            } else {
                await fetch('/training/api/force_stop', { method: 'POST' });
            }
        }
    });
}

// Track the previous Elo so we can show a trend arrow on the hero card.
let prevEloForTrend = null;

function updateEloTrend(delta) {
    const el = document.getElementById('t-elo-trend');
    if (!el || delta === null || delta === undefined || Math.abs(delta) < 0.5) {
        if (el) el.textContent = '';
        return;
    }
    const up = delta > 0;
    el.textContent = `${up ? '▲' : '▼'}${up ? '+' : ''}${Math.round(delta)}`;
    el.style.color = up ? 'var(--success)' : 'var(--danger)';
}

function updateHeroStatus(isRunning, subStatusText = null) {
    const dot = document.getElementById('t-status-dot');
    const label = document.getElementById('t-status-label');
    if (!dot || !label) return;
    if (isRunning) {
        dot.classList.add('active');
        label.textContent = subStatusText || 'Training';
    } else {
        dot.classList.remove('active');
        label.textContent = 'Idle';
    }
}

// Debounce helper for live block updates
let refreshStatsTimeout = null;
function debouncedRefreshStats() {
    clearTimeout(refreshStatsTimeout);
    refreshStatsTimeout = setTimeout(() => {
        loadLearningStats();
        loadGamesList();
    }, 400);
}

// ---- SocketIO Events ----
socket.on('training_update', (data) => {
    // Unwrap data object if type is status
    const d = (data.type === 'status' && data.data) ? data.data : data;

    // If the active model changed under us (model switch, or a server restart
    // that came back on a different model), the charts are showing another
    // model's run. Clear and reload rather than appending a second series onto
    // the first — that concatenation is what made the data look doubled.
    const incomingModel = data.model_id
        || (d && d.active_model && d.active_model.id)
        || null;
    if (incomingModel && currentModelId && incomingModel !== currentModelId) {
        currentModelId = incomingModel;
        resetCharts();
        allMetrics = [];
        loadHistoricalMetrics();
    } else if (incomingModel && !currentModelId) {
        currentModelId = incomingModel;
    }

    // Update stat cards
    if (d.elo != null) document.getElementById('t-elo').textContent = Math.round(d.elo);
    if (d.kyu_rank != null) document.getElementById('t-rank').textContent = d.kyu_rank;
    if (d.iteration != null) document.getElementById('t-iter').textContent = d.iteration;
    if (d.total_games != null) document.getElementById('t-games').textContent = d.total_games;

    // Update button states
    const startBtn = document.getElementById('btn-start-training');
    const stopBtn = document.getElementById('btn-stop-training');
    const forceBtn = document.getElementById('btn-force-stop');
    
    if (startBtn && stopBtn) {
        if (data.type === 'error') {
            showToast('⚠ ' + (data.message || 'Error occurred'));
            startBtn.style.display = '';
            startBtn.disabled = false;
            startBtn.textContent = '▶ Start Training';
            stopBtn.style.display = 'none';
            if (forceBtn) forceBtn.style.display = 'none';
            updateHeroStatus(false);
        } else if (data.type === 'training_started' || (data.type === 'status' && data.data && data.data.is_running && !data.data.stop_requested)) {
            startBtn.style.display = 'none';
            startBtn.disabled = false;
            startBtn.textContent = '▶ Start Training';
            stopBtn.style.display = '';
            stopBtn.disabled = false;
            stopBtn.textContent = '⏹ Stop Training';
            if (forceBtn) {
                forceBtn.style.display = '';
                forceBtn.disabled = false;
                forceBtn.textContent = '⚡ Force Stop';
            }
            updateHeroStatus(true, 'Training');
        } else if (data.type === 'status' && data.data && data.data.is_running && data.data.stop_requested) {
            startBtn.style.display = 'none';
            startBtn.disabled = false;
            startBtn.textContent = '▶ Start Training';
            stopBtn.style.display = '';
            stopBtn.disabled = true;
            stopBtn.textContent = '⏳ Stopping...';
            if (forceBtn) {
                forceBtn.style.display = '';
                forceBtn.disabled = false;
            }
            updateHeroStatus(true, 'Stopping...');
        } else if (data.type === 'training_stopped' || (data.type === 'status' && data.data && !data.data.is_running)) {
            startBtn.style.display = '';
            startBtn.disabled = false;
            startBtn.textContent = '▶ Start Training';
            stopBtn.style.display = 'none';
            if (forceBtn) forceBtn.style.display = 'none';
            updateHeroStatus(false);
        }
    }

    // Dynamic phase status label updates
    if (data.type === 'self_play_start') {
        updateHeroStatus(true, 'Self-Play');
    } else if (data.type === 'game_complete') {
        updateHeroStatus(true, `Self-Play (${data.game_num}/${data.total})`);
        debouncedRefreshStats();
    } else if (data.type === 'self_play_done') {
        updateHeroStatus(true, 'Self-Play Done');
        debouncedRefreshStats();
    } else if (data.type === 'training_start') {
        updateHeroStatus(true, 'Training NN');
    } else if (data.type === 'training_done') {
        updateHeroStatus(true, 'Training Done');
        debouncedRefreshStats();
    } else if (data.type === 'eval_start') {
        updateHeroStatus(true, 'Evaluating');
    } else if (data.type === 'eval_done') {
        updateHeroStatus(true, 'Eval Done');
        debouncedRefreshStats();
    }

    // Populate recent logs & trigger full load if status event
    if (data.type === 'status' && data.data) {
        if (data.data.recent_logs) {
            document.getElementById('training-log').innerHTML = ''; // Clear existing
            data.data.recent_logs.forEach(log => {
                if (log.message) addLogEntry(log); // Always prepend to maintain newest-first order
            });
        }
        loadLearningStats();
        loadGamesList();
    }

    // Update charts on iteration_done
    if (data.type === 'iteration_done') {
        updateHeroStatus(true, 'Iteration Done');
        if (prevEloForTrend !== null && data.elo !== undefined) {
            updateEloTrend(data.elo - prevEloForTrend);
        }
        if (data.elo !== undefined) prevEloForTrend = data.elo;
        updateCharts(data);
        pushMetric({
            iteration: data.iteration,
            elo: data.elo,
            kyu_rank: data.kyu_rank,
            policy_loss: data.policy_loss,
            value_loss: data.value_loss,
            total_loss: data.total_loss,
            win_rate_vs_random: data.win_rate_vs_random,
        });
        if (metricsTableVisible()) renderMetricsTable();
        loadGamesList();
        loadLearningStats();
    }

    // Log entry
    if (data.message) {
        addLogEntry(data);
    }

    // Milestone toast
    if (data.type === 'reflection' && data.milestone) {
        showMilestone(data.milestone);
    }
});

// ---- Log ----
function addLogEntry(data, append = false) {
    const log = document.getElementById('training-log');
    const entry = document.createElement('div');
    entry.className = 'log-entry';

    let timeStr = '';
    if (data.timestamp) {
        timeStr = new Date(data.timestamp).toLocaleTimeString();
    } else {
        timeStr = new Date().toLocaleTimeString();
    }

    const isMilestone = data.type === 'reflection';

    entry.innerHTML = `<span class="log-time">${timeStr}</span>` +
        `<span class="${isMilestone ? 'log-milestone' : ''}">${data.message}</span>`;

    if (append) {
        log.appendChild(entry);
    } else {
        log.prepend(entry);
    }

    // Keep log manageable
    while (log.children.length > 200) {
        log.removeChild(log.lastChild);
    }
}

// ---- Games List ----
async function loadGamesList() {
    try {
        const res = await fetch('/training/api/games');
        const groupedGames = await res.json();
        const list = document.getElementById('games-list');
        list.innerHTML = '';

        groupedGames.forEach((group, groupIdx) => {
            const details = document.createElement('details');
            details.className = 'iteration-group';
            if (groupIdx === 0) details.open = true; // Open the most recent by default
            
            let evalCount = 0;
            let aiWins = 0;
            group.games.forEach(game => {
                if (game.is_eval && game.network_color !== undefined) {
                    evalCount++;
                    if (game.winner === game.network_color) {
                        aiWins++;
                    }
                }
            });
            
            let winrateHtml = '';
            if (evalCount > 0) {
                const wr = Math.round((aiWins / evalCount) * 100);
                let wrColor = 'var(--text-muted)';
                if (wr > 50) wrColor = 'var(--success)';
                else if (wr < 50) wrColor = 'var(--danger)';
                else wrColor = 'var(--warning)';
                
                winrateHtml = `<span style="font-size: 0.85em; font-weight: normal; color: ${wrColor};">WR: ${wr}% (${aiWins}/${evalCount})</span>`;
            }
            
            const eloText = group.elo ? `<span style="color: var(--text-muted); font-size: 0.85em; font-weight: normal; margin-left: 0.5rem;">(~${group.elo} Elo)</span>` : '';
            
            details.innerHTML = `<summary style="display: flex; justify-content: space-between; align-items: center;">
                <span>Iteration ${group.iteration}${eloText}</span>
                ${winrateHtml}
            </summary>`;
            
            // Sort games: eval games first, then by game index
            group.games.sort((a, b) => {
                if (a.is_eval && !b.is_eval) return -1;
                if (!a.is_eval && b.is_eval) return 1;
                return a.game_index - b.game_index;
            });
            
            group.games.forEach(game => {
                const item = document.createElement('div');
                item.className = 'game-item';
                
                const winnerIcon = game.winner === 1 ? '⚫' : (game.winner === 2 ? '⚪' : '🤝');
                let label = game.is_eval ? `Eval ${game.game_index}` : `Game ${game.game_index}`;
                
                if (game.is_eval && game.network_color !== undefined) {
                    const aiIcon = game.network_color === 1 ? '⚫' : '⚪';
                    const randIcon = game.network_color === 1 ? '⚪' : '⚫';
                    label = `Eval ${game.game_index} (${aiIcon} AI vs ${randIcon} Rand)`;
                }
                
                let resultText = 'Draw';
                if (game.winner === 1) resultText = `B+${game.margin || '?'}`;
                else if (game.winner === 2) resultText = `W+${game.margin || '?'}`;
                
                let resultColor = 'var(--text-muted)';
                if (game.is_eval && game.network_color !== undefined && game.winner !== 0) {
                    resultColor = (game.winner === game.network_color) ? 'var(--success)' : 'var(--danger)';
                }
                
                item.innerHTML = `
                    <span>${winnerIcon} ${label}</span>
                    <span class="meta"><span style="color: ${resultColor}; font-weight: 600;">${resultText}</span> &middot; ${game.num_moves} moves</span>
                `;
                item.addEventListener('click', () => {
                    window.location.href = `/training/review?game=${encodeURIComponent(game.filename)}`;
                });
                details.appendChild(item);
            });
            list.appendChild(details);
        });
    } catch (e) { /* ignore */ }
}

// ---- Learning Stats ----
function formatDuration(seconds) {
    if (seconds === null || seconds === undefined || isNaN(seconds)) return '—';
    if (seconds < 60) return seconds.toFixed(1) + 's';
    const m = Math.floor(seconds / 60);
    const s = (seconds % 60).toFixed(0);
    return `${m}m ${s}s`;
}

// Filter state for self-play & vs random bot graphs
let cachedLearningStats = null;
let currentSelfPlayFilter = 'all'; // 'all', '10', '20', '50', '100'
let currentRandomFilter = 'all';   // 'all', '10', '20', '50', '100'

function renderSelfPlayStats(stats) {
    if (!stats) return;
    const el = (id) => document.getElementById(id);
    let series = stats.self_play_series || [];

    if (currentSelfPlayFilter !== 'all') {
        const limit = parseInt(currentSelfPlayFilter, 10);
        if (!isNaN(limit) && limit > 0) {
            series = series.slice(-limit);
        }
    }

    let bWins = 0;
    let wWins = 0;
    let draws = 0;
    series.forEach(m => {
        if (m > 0) bWins++;
        else if (m < 0) wWins++;
        else draws++;
    });

    const total = series.length;
    if (total > 0) {
        const wrB = Math.round((bWins / total) * 100);
        const wrW = 100 - wrB;
        if (el('ls-wr-black')) el('ls-wr-black').textContent = wrB + '%';
        if (el('ls-wr-white')) el('ls-wr-white').textContent = wrW + '%';
        if (el('balance-black-fill')) el('balance-black-fill').style.width = wrB + '%';
        if (el('balance-white-fill')) el('balance-white-fill').style.width = wrW + '%';
    } else {
        if (el('ls-wr-black')) el('ls-wr-black').textContent = '—';
        if (el('ls-wr-white')) el('ls-wr-white').textContent = '—';
        if (el('balance-black-fill')) el('balance-black-fill').style.width = '50%';
        if (el('balance-white-fill')) el('balance-white-fill').style.width = '50%';
    }

    if (el('sp-black-wins')) el('sp-black-wins').textContent = bWins;
    if (el('sp-white-wins')) el('sp-white-wins').textContent = wWins;
    if (typeof updateSelfPlayMargins === 'function') {
        updateSelfPlayMargins(series);
    }
}

function renderRandomStats(stats) {
    if (!stats) return;
    const el = (id) => document.getElementById(id);
    let rSeries = stats.random_series || [];

    if (currentRandomFilter !== 'all') {
        const limit = parseInt(currentRandomFilter, 10);
        if (!isNaN(limit) && limit > 0) {
            rSeries = rSeries.slice(-limit);
        }
    }

    let rWins = 0;
    let rLoss = 0;
    let rDraws = 0;
    rSeries.forEach(m => {
        if (m > 0) rWins++;
        else if (m < 0) rLoss++;
        else rDraws++;
    });

    const rTotal = rSeries.length;
    if (rTotal > 0) {
        const wPct = Math.round((rWins / rTotal) * 100);
        const lPct = 100 - wPct;
        if (el('rand-wr')) el('rand-wr').textContent = wPct + '%';
        if (el('rand-lr')) el('rand-lr').textContent = lPct + '%';
        if (el('rand-win-fill')) el('rand-win-fill').style.width = wPct + '%';
        if (el('rand-loss-fill')) el('rand-loss-fill').style.width = lPct + '%';
    } else {
        if (el('rand-wr')) el('rand-wr').textContent = '—';
        if (el('rand-lr')) el('rand-lr').textContent = '—';
        if (el('rand-win-fill')) el('rand-win-fill').style.width = '0%';
        if (el('rand-loss-fill')) el('rand-loss-fill').style.width = '0%';
    }

    if (el('rand-wins')) el('rand-wins').textContent = rWins;
    if (el('rand-losses')) el('rand-losses').textContent = rLoss;
    if (typeof updateRandomMargins === 'function') {
        updateRandomMargins(rSeries);
    }
}

function setupGraphFilters() {
    const spBtns = document.querySelectorAll('#selfplay-filter-btns .filter-btn');
    spBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            spBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentSelfPlayFilter = btn.dataset.limit || 'all';
            renderSelfPlayStats(cachedLearningStats);
        });
    });

    const randBtns = document.querySelectorAll('#random-filter-btns .filter-btn');
    randBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            randBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentRandomFilter = btn.dataset.limit || 'all';
            renderRandomStats(cachedLearningStats);
        });
    });
}

async function loadLearningStats() {
    try {
        const res = await fetch('/training/api/learning_stats');
        const stats = await res.json();
        cachedLearningStats = stats;

        const el = (id) => document.getElementById(id);

        el('ls-avg-time-total').textContent = formatDuration(stats.avg_time_per_game_total);
        el('ls-iter-time-avg').textContent = formatDuration(stats.iter_time_avg);
        el('ls-game-length').textContent = (stats.avg_game_length !== null && stats.avg_game_length !== undefined)
            ? `${stats.avg_game_length} moves` : '—';
        el('ls-buffer').textContent = (stats.buffer_size !== null && stats.buffer_size !== undefined)
            ? `${stats.buffer_size.toLocaleString()} / ${stats.buffer_capacity.toLocaleString()}` : '—';
        el('ls-lr').textContent = (stats.learning_rate !== null && stats.learning_rate !== undefined)
            ? Number(stats.learning_rate.toPrecision(3)).toString() : '—';

        // --- Hero tiles ---
        if (el('t-device')) el('t-device').textContent = (stats.device || '—').toUpperCase();
        if (el('t-best-elo')) el('t-best-elo').textContent = (stats.best_elo != null) ? stats.best_elo : '—';
        if (el('t-wr-random')) {
            el('t-wr-random').textContent = (stats.latest_win_rate_vs_random != null)
                ? Math.round(stats.latest_win_rate_vs_random * 100) + '%' : '—';
        }

        // --- Render self-play & vs random bot with current active filter ---
        renderSelfPlayStats(stats);
        renderRandomStats(stats);
    } catch (e) { /* ignore */ }
}

// ---- Milestone Toast ----
function showMilestone(text) {
    const toast = document.getElementById('milestone-toast');
    document.getElementById('milestone-text').textContent = `🎯 ${text}`;
    toast.style.display = '';
    setTimeout(() => { toast.style.display = 'none'; }, 5000);
}

// ---- Lightweight action toast ----
function showToast(text) {
    const toast = document.getElementById('action-toast');
    if (!toast) return;
    toast.textContent = text;
    toast.style.display = '';
    clearTimeout(showToast._t);
    showToast._t = setTimeout(() => { toast.style.display = 'none'; }, 4000);
}

// ---- Parameter Tuner (bottom-bar) ----
const tunePanel = document.getElementById('tune-panel');
const btnTuneToggle = document.getElementById('btn-tune-toggle');
const btnTuneApply = document.getElementById('btn-tune-apply');
const btnTuneCancel = document.getElementById('btn-tune-cancel');

// Prefill the tuner inputs from the active model's stored training params.
async function prefillTuner() {
    try {
        const res = await fetch('/models/api/active');
        const model = await res.json();
        if (!model || !model.training) return;
        const t = model.training;
        const set = (id, v) => { const el = document.getElementById(id); if (el && v != null) el.value = v; };
        set('tune-sp', t.num_self_play_games);
        set('tune-eval', t.eval_games);
        set('tune-mcts', t.num_simulations);
        set('tune-cpuct', t.c_puct);
        set('tune-lr', t.learning_rate);
        set('tune-temp-thresh', t.temperature_threshold);
        set('tune-temp-init', t.temperature_init);
        set('tune-temp-final', t.temperature_final);
    } catch (e) { /* ignore */ }
}

function closeTuner() {
    if (tunePanel) tunePanel.hidden = true;
    if (btnTuneToggle) btnTuneToggle.classList.remove('active');
}

if (btnTuneToggle) {
    btnTuneToggle.addEventListener('click', () => {
        if (!tunePanel) return;
        const opening = tunePanel.hidden;
        tunePanel.hidden = !opening;
        btnTuneToggle.classList.toggle('active', opening);
        if (opening) prefillTuner();
    });
}

if (btnTuneCancel) btnTuneCancel.addEventListener('click', closeTuner);

if (btnTuneApply) {
    btnTuneApply.addEventListener('click', async () => {
        const num = (id) => {
            const el = document.getElementById(id);
            if (!el || el.value === '') return null;
            const v = parseFloat(el.value);
            return isNaN(v) ? null : v;
        };
        const payload = {
            num_self_play_games: num('tune-sp'),
            eval_games: num('tune-eval'),
            num_simulations: num('tune-mcts'),
            c_puct: num('tune-cpuct'),
            learning_rate: num('tune-lr'),
            temperature_threshold: num('tune-temp-thresh'),
            temperature_init: num('tune-temp-init'),
            temperature_final: num('tune-temp-final'),
        };

        btnTuneApply.disabled = true;
        btnTuneApply.textContent = 'Applying...';
        try {
            const res = await fetch('/training/api/apply_params', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await res.json();
            if (!res.ok) {
                showToast('⚠ ' + (data.error || 'Failed to apply'));
            } else {
                showToast('✓ ' + data.message);
                closeTuner();
                loadLearningStats(); // refresh the Learning Rate stat, etc.
            }
        } catch (e) {
            showToast('⚠ Failed to connect to server');
        }
        btnTuneApply.disabled = false;
        btnTuneApply.textContent = '✓ Apply';
    });
}

// ---- Metrics table (full history, scrollable) ----
// The charts cap at 100 points; the table keeps every iteration so you can
// scroll back to the very first ones and read exact values.
let allMetrics = [];

// Which model the currently-charted series belongs to; used to detect a switch.
let currentModelId = null;

function pushMetric(m) {
    if (m == null || m.iteration == null) return;
    // Key by iteration rather than only comparing against the last row: a
    // replayed or out-of-order iteration would otherwise be appended again and
    // show up twice in the table.
    const at = allMetrics.findIndex(x => x.iteration === m.iteration);
    if (at !== -1) {
        allMetrics[at] = m;
    } else {
        allMetrics.push(m);
    }
}

function fmtLoss(v) {
    if (v === null || v === undefined || isNaN(v)) return '—';
    return Number(v).toFixed(4);
}

function renderMetricsTable() {
    const body = document.getElementById('metrics-table-body');
    const empty = document.getElementById('metrics-table-empty');
    if (!body) return;
    if (!allMetrics.length) {
        body.innerHTML = '';
        if (empty) empty.style.display = '';
        return;
    }
    if (empty) empty.style.display = 'none';
    // Ascending by iteration so the earliest iterations sit at the top.
    const rows = allMetrics.slice().sort((a, b) => (a.iteration || 0) - (b.iteration || 0));
    body.innerHTML = rows.map(m => {
        const wr = (m.win_rate_vs_random != null) ? Math.round(m.win_rate_vs_random * 100) + '%' : '—';
        let total = m.total_loss;
        if (total == null && m.policy_loss != null && m.value_loss != null) total = m.policy_loss + m.value_loss;
        return `<tr>
            <td>${m.iteration != null ? m.iteration : '—'}</td>
            <td>${m.elo != null ? Math.round(m.elo) : '—'}</td>
            <td>${m.kyu_rank || '—'}</td>
            <td>${fmtLoss(m.policy_loss)}</td>
            <td>${fmtLoss(m.value_loss)}</td>
            <td>${fmtLoss(total)}</td>
            <td>${wr}</td>
        </tr>`;
    }).join('');
}

function metricsTableVisible() {
    const card = document.getElementById('metrics-table-card');
    return card && !card.hidden;
}

// ---- Chart / Table view toggle ----
function setupMetricsViewToggle() {
    const toggle = document.getElementById('metrics-view-toggle');
    if (!toggle) return;
    toggle.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            toggle.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const charts = document.getElementById('metrics-charts');
            const table = document.getElementById('metrics-table-card');
            if (btn.dataset.view === 'table') {
                if (charts) charts.hidden = true;
                if (table) table.hidden = false;
                renderMetricsTable();
            } else {
                if (table) table.hidden = true;
                if (charts) charts.hidden = false;
            }
        });
    });
}

// ---- Initial Load ----
async function loadHistoricalMetrics() {
    try {
        const res = await fetch('/training/api/metrics');
        const metrics = await res.json();
        allMetrics = Array.isArray(metrics) ? metrics.slice() : [];
        // Start from a clean series so a reload can never stack a second copy
        // of the history on top of what is already plotted.
        resetCharts();
        for (const m of metrics) {
            updateCharts(m);
        }
        renderMetricsTable();
        // Seed the hero Elo trend from the last two recorded iterations.
        if (metrics.length >= 2) {
            const last = metrics[metrics.length - 1].elo;
            const prev = metrics[metrics.length - 2].elo;
            if (last !== undefined && prev !== undefined) updateEloTrend(last - prev);
            prevEloForTrend = last;
        } else if (metrics.length === 1) {
            prevEloForTrend = metrics[0].elo;
        }
    } catch (e) { /* ignore */ }
}

// Request current status on load to fix blank UI on refresh
socket.emit('request_status');

loadHistoricalMetrics();
loadGamesList();
loadLearningStats();
setupGraphFilters();
setupMetricsViewToggle();
