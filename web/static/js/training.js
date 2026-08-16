/**
 * training_new.js — Training dashboard real-time updates (Candidate vs Champion Redesign).
 *
 * Connects via SocketIO for live training events.
 * Manages:
 * - Collapsible blocks with automatic Chart.js resize
 * - Candidate vs Champion metrics & hero stats
 * - Neural network health charts (Policy & Value loss)
 * - Pipeline stats & self-play balance
 * - Demoted, collapsible Time Metrics block
 * - Training controls, log feed, game browser, param tuner
 */

let replayBoard = null;
let replayData = null;
let replayMoveIndex = 0;

// ---- Button State Synchronizer ----
function syncButtonStates(isRunning, stopRequested = false) {
    const startBtn = document.getElementById('btn-start-training');
    const stopBtn = document.getElementById('btn-stop-training');
    const forceBtn = document.getElementById('btn-force-stop');

    if (!startBtn || !stopBtn) return;

    if (isRunning) {
        startBtn.style.display = 'none';
        startBtn.disabled = false;
        startBtn.textContent = '▶ Start Training';
        stopBtn.style.display = '';
        if (stopRequested) {
            stopBtn.disabled = true;
            stopBtn.textContent = '⏳ Stopping...';
        } else {
            stopBtn.disabled = false;
            stopBtn.textContent = '⏹ Stop Training';
        }
        if (forceBtn) {
            forceBtn.style.display = '';
            forceBtn.disabled = false;
            forceBtn.textContent = '⚡ Force Stop';
        }
    } else {
        startBtn.style.display = '';
        startBtn.disabled = false;
        startBtn.textContent = '▶ Start Training';
        stopBtn.style.display = 'none';
        stopBtn.disabled = false;
        if (forceBtn) forceBtn.style.display = 'none';
    }
}

// ---- Live Status & Pipeline Stepper & Parallel Games Renderer ----
function renderLiveStatus(data) {
    if (!data) return;
    const d = (data.type === 'status' && data.data) ? data.data : data;
    const stage = d.current_stage || data.current_stage || {};
    const isRunning = d.is_running !== undefined ? d.is_running : (stage.stage && stage.stage !== 'idle');
    const stopRequested = d.stop_requested || false;
    const iteration = d.iteration !== undefined ? d.iteration : stage.iteration;

    syncButtonStates(isRunning, stopRequested);

    // 1. Status Pill & Pulse Dot
    const statusPill = document.getElementById('status-pill');
    const statusDot = document.getElementById('status-dot');
    const statusMainLabel = document.getElementById('status-main-label');
    const heroDot = document.getElementById('t-status-dot');
    const heroLabel = document.getElementById('t-status-label');

    if (isRunning) {
        if (stopRequested) {
            if (statusPill) statusPill.className = 'status-live-pill is-stopping';
            if (statusDot) statusDot.className = 'status-pulse-dot is-stopping';
            if (statusMainLabel) statusMainLabel.textContent = 'Stopping...';
            if (heroDot) heroDot.className = 'status-dot is-stopping';
            if (heroLabel) heroLabel.textContent = 'Stopping...';
        } else {
            if (statusPill) statusPill.className = 'status-live-pill is-running';
            if (statusDot) statusDot.className = 'status-pulse-dot is-running';
            if (statusMainLabel) statusMainLabel.textContent = 'Running';
            if (heroDot) heroDot.className = 'status-dot active';
            if (heroLabel) heroLabel.textContent = 'Running';
        }
    } else {
        if (statusPill) statusPill.className = 'status-live-pill is-idle';
        if (statusDot) statusDot.className = 'status-pulse-dot is-idle';
        if (statusMainLabel) statusMainLabel.textContent = 'Idle';
        if (heroDot) heroDot.className = 'status-dot';
        if (heroLabel) heroLabel.textContent = 'Idle';
    }

    // 2. Stage Header & Iteration Badge
    const stageTitle = document.getElementById('status-stage-title');
    const stageLabel = document.getElementById('status-stage-label');
    const iterBadge = document.getElementById('status-iter-badge');
    const heroIter = document.getElementById('t-iter');
    const gamesBadge = document.getElementById('status-games-badge');
    const heroGames = document.getElementById('t-games');
    const workersBadge = document.getElementById('status-workers-count');
    const deviceBadge = document.getElementById('status-device-badge');
    const heroDevice = document.getElementById('t-device');

    if (stageTitle) {
        if (!isRunning) {
            stageTitle.textContent = 'Ready to train';
        } else {
            stageTitle.textContent = stage.stage_name || 'Learning in progress';
        }
    }
    if (stageLabel) {
        if (stage.stage_index && stage.stage_index > 0) {
            stageLabel.textContent = `Stage ${stage.stage_index} of ${stage.total_stages || 5}`;
        } else {
            stageLabel.textContent = isRunning ? 'Active Training' : 'System Standby';
        }
    }
    if (iterBadge && iteration != null) iterBadge.textContent = `#${iteration}`;
    if (heroIter && iteration != null) heroIter.textContent = iteration;
    if (gamesBadge && d.total_games != null) gamesBadge.textContent = d.total_games;
    if (heroGames && d.total_games != null) heroGames.textContent = d.total_games;

    const numWorkers = stage.num_workers || 0;
    if (workersBadge) {
        workersBadge.textContent = numWorkers > 0 ? `${numWorkers} active` : '—';
    }
    const dev = d.device || stage.device;
    if (dev) {
        if (deviceBadge) deviceBadge.textContent = dev.toUpperCase();
        if (heroDevice) heroDevice.textContent = dev.toUpperCase();
    }

    // 3. Pipeline Stepper (5 stages)
    const stageKey = stage.stage || (isRunning ? 'self_play' : 'idle');
    const stagesOverview = stage.stages_overview || [];
    const stepKeys = ['self_play', 'training', 'gate', 'eval', 'saving'];

    stepKeys.forEach((key, idx) => {
        const stepEl = document.getElementById(`step-${key}`);
        if (!stepEl) return;

        let status = 'pending';
        const found = stagesOverview.find(s => s.key === key);
        if (found) {
            status = found.status;
        } else {
            if (!isRunning) status = 'idle';
            else if (key === stageKey) status = 'active';
            else {
                const curIdx = stepKeys.indexOf(stageKey);
                if (curIdx > idx) status = 'completed';
                else status = 'pending';
            }
        }

        stepEl.classList.remove('is-completed', 'is-active', 'is-pending', 'is-skipped', 'is-idle');
        stepEl.classList.add(`is-${status}`);

        // Update connectors
        if (idx > 0) {
            const conn = document.getElementById(`conn-${idx}`);
            if (conn) {
                conn.classList.remove('conn-completed', 'conn-active', 'conn-pending');
                const prevStep = document.getElementById(`step-${stepKeys[idx - 1]}`);
                if (prevStep && prevStep.classList.contains('is-completed')) {
                    conn.classList.add('conn-completed');
                } else if (stepEl.classList.contains('is-active')) {
                    conn.classList.add('conn-active');
                } else {
                    conn.classList.add('conn-pending');
                }
            }
        }
    });

    // 4. Stage Arena Progress & Detail
    const detailEl = document.getElementById('stage-detail-text');
    const progFill = document.getElementById('stage-progress-fill');
    const progText = document.getElementById('stage-progress-text');
    const progPct = document.getElementById('stage-progress-pct');

    if (detailEl) {
        detailEl.textContent = stage.detail || (isRunning ? 'Processing...' : 'Ready for training · Press Start Training below to begin.');
    }
    const percent = Math.min(100, Math.max(0, stage.percent || 0));
    if (progFill) progFill.style.width = `${percent}%`;
    if (progPct) progPct.textContent = `${percent}%`;
    if (progText) {
        if (stage.total_items > 0) {
            progText.textContent = `${stage.completed_items || 0} / ${stage.total_items}`;
        } else {
            progText.textContent = isRunning ? '—' : 'Ready';
        }
    }

    // 5. Parallel Games Visualizer vs NN Step Readout
    const parallelArena = document.getElementById('parallel-games-arena');
    const nnArena = document.getElementById('nn-step-arena');
    const activeCountEl = document.getElementById('active-parallel-count');
    const activeContainer = document.getElementById('active-games-container');
    const compSummary = document.getElementById('completed-games-summary');
    const queueSummary = document.getElementById('queued-games-summary');

    const isGamePhase = ['self_play', 'gate', 'eval'].includes(stageKey);

    if (isGamePhase && isRunning && stage.total_items > 0) {
        if (parallelArena) parallelArena.style.display = 'block';
        if (nnArena) nnArena.style.display = 'none';

        const activeGames = stage.active_games || [];
        if (activeCountEl) activeCountEl.textContent = activeGames.length;
        if (compSummary) compSummary.textContent = `✓ ${stage.completed_items || 0} finished`;

        const remaining = Math.max(0, stage.total_items - (stage.completed_items || 0) - activeGames.length);
        if (queueSummary) queueSummary.textContent = `${remaining} queued`;

        if (activeContainer) {
            if (activeGames.length > 0) {
                activeContainer.innerHTML = activeGames.map(num => `
                    <div class="parallel-game-pill is-running">
                        <span class="game-spin-dot"></span>
                        <span class="game-pill-num">Game #${num}</span>
                        <span class="game-pill-tag">Running</span>
                    </div>
                `).join('');
            } else if (stage.completed_items >= stage.total_items) {
                activeContainer.innerHTML = `<div class="parallel-all-done">✓ All ${stage.total_items} games completed for this stage</div>`;
            } else {
                activeContainer.innerHTML = `<div class="parallel-waiting">Launching parallel workers...</div>`;
            }
        }
    } else if (stageKey === 'training' && isRunning) {
        if (parallelArena) parallelArena.style.display = 'none';
        if (nnArena) nnArena.style.display = 'block';

        const chipStep = document.getElementById('nn-chip-step');
        const chipLoss = document.getElementById('nn-chip-loss');
        const chipPolicy = document.getElementById('nn-chip-policy');
        const chipValue = document.getElementById('nn-chip-value');

        if (chipStep) chipStep.innerHTML = `Step: <strong>${stage.completed_items || 0} / ${stage.total_items || 0}</strong>`;
        const totLoss = d.total_loss !== undefined ? d.total_loss : (d.loss !== undefined ? d.loss : null);
        const polLoss = d.policy_loss;
        const valLoss = d.value_loss;

        if (chipLoss) chipLoss.innerHTML = `Combined Loss: <strong>${totLoss != null ? Number(totLoss).toFixed(4) : '—'}</strong>`;
        if (chipPolicy) chipPolicy.innerHTML = `Policy Loss: <strong>${polLoss != null ? Number(polLoss).toFixed(4) : '—'}</strong>`;
        if (chipValue) chipValue.innerHTML = `Value Loss: <strong>${valLoss != null ? Number(valLoss).toFixed(4) : '—'}</strong>`;
    } else {
        if (parallelArena) parallelArena.style.display = 'none';
        if (nnArena) nnArena.style.display = 'none';
    }
}

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

function updateHeroStatus(isRunning, subStatusText = null) {
    // Keep legacy helper for compatibility
    renderLiveStatus({ is_running: isRunning, current_stage: { stage_name: subStatusText } });
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
if (typeof socket !== 'undefined' && socket) {
    socket.on('training_update', (data) => {
        // Unwrap data object if type is status
        const d = (data.type === 'status' && data.data) ? data.data : data;

        // If the active model changed under us
        const incomingModel = data.model_id
            || (d && d.active_model && d.active_model.id)
            || null;
        if (incomingModel && currentModelId && incomingModel !== currentModelId) {
            currentModelId = incomingModel;
            resetCharts();
            allMetrics = [];
            loadHistoricalMetrics();
            loadGateHistory();
            loadResignStats();
        } else if (incomingModel && !currentModelId) {
            currentModelId = incomingModel;
        }

        // Render full live status & stage state
        renderLiveStatus(d);

        if (data.type === 'error') {
            showToast('⚠ ' + (data.message || 'Error occurred'));
        }

        // Live debounces on game events
        if (data.type === 'game_complete' || data.type === 'self_play_done' || data.type === 'gate_progress' || data.type === 'eval_progress') {
            debouncedRefreshStats();
        }

        // Populate recent logs & trigger full load if status event
        if (data.type === 'status' && data.data) {
            if (data.data.recent_logs) {
                const logEl = document.getElementById('training-log');
                if (logEl) {
                    logEl.innerHTML = '';
                    data.data.recent_logs.forEach(log => {
                        if (log.message) addLogEntry(log);
                    });
                }
            }
            loadLearningStats();
            loadGamesList();
            loadGateHistory();
            loadResignStats();
        }

        // Update charts on iteration_done
        if (data.type === 'iteration_done') {
            updateCharts(data);
            pushMetric(data);
            if (metricsTableVisible()) renderMetricsTable();
            loadGamesList();
            loadLearningStats();
            loadGateHistory();
            loadResignStats();
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
}


// ---- Log ----
function addLogEntry(data, append = false) {
    const log = document.getElementById('training-log');
    if (!log) return;
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
// The training sidebar is a live view of the run in progress, not an archive
// — older iterations live on the Review page, which pages through them.
const GAMES_ITERATIONS = 3;

function escapeAttr(str) {
    return String(str).replace(/&/g, '&amp;').replace(/"/g, '&quot;')
                      .replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/**
 * Mercy-rule marker on a game row. The server explains the resignation
 * (ai/resignation.py); the row shows the short form and carries the full
 * reason as a tooltip. `checked` games did NOT end early — the rule fired and
 * was deliberately overruled so it could be measured.
 */
function gameResignTag(game) {
    const info = game.resignation;
    if (!info) return '';

    const title = escapeAttr(info.reason || '');
    if (info.resigned) return `<span class="resign-tag" title="${title}">🏳</span>`;
    if (info.false_resign) return `<span class="resign-tag is-wrong" title="${title}">⚑!</span>`;
    return `<span class="resign-tag is-check" title="${title}">⚑</span>`;
}

/** How much of a phase ended early, on its header. */
function gamesPhaseResignBadge(phase) {
    const resigned = phase.resigned_count || 0;
    const checked = phase.resign_checked_count || 0;
    const wrong = phase.false_resign_count || 0;
    if (!resigned && !checked) return '';

    const bits = [];
    if (resigned) bits.push(`${resigned} of ${phase.count} games ended by resignation`);
    if (checked) bits.push(`${checked} played out as mercy-rule checks`);
    if (wrong) bits.push(`${wrong} of those checks show the rule would have been WRONG`);

    const cls = wrong ? 'phase-resign-note is-wrong' : 'phase-resign-note';
    const label = resigned ? `🏳 ${resigned}` : `⚑ ${checked}`;
    return `<span class="${cls}" title="${escapeAttr(bits.join('; '))}">${label}</span>`;
}

function gamesPhaseBadge(phase) {
    const color = (rate, threshold = 0.5) => {
        if (rate > threshold) return 'var(--success)';
        if (rate < threshold) return 'var(--danger)';
        return 'var(--warning)';
    };

    const resign = gamesPhaseResignBadge(phase);

    if (phase.phase === 'promotion') {
        const rate = phase.candidate_win_rate !== undefined && phase.candidate_win_rate !== null
            ? phase.candidate_win_rate
            : phase.gate_win_rate;
        if (rate === undefined || rate === null) return resign;

        const pct = Math.round(rate * 100);
        return `${resign}<span class="group-note" style="color: ${color(rate, phase.gate_threshold || 0.5)}; font-weight: 700; font-size: 0.95rem; white-space: nowrap;">
            ${pct}%
        </span>`;
    }

    if (phase.phase === 'eval' && phase.win_rate !== null && phase.win_rate !== undefined) {
        const pct = Math.round(phase.win_rate * 100);
        return `${resign}<span class="group-note" style="color: ${color(phase.win_rate)}; font-weight: 700; font-size: 0.95rem; white-space: nowrap;">
            ${pct}%
        </span>`;
    }

    return resign;
}

async function loadGamesList() {
    try {
        // Only the iterations you are actually watching. A long run has
        // thousands of stored games, and the sidebar showed every one of them.
        const res = await fetch(`/training/api/games?include_recorded=0&iterations=${GAMES_ITERATIONS}`);
        const payload = await res.json();
        const groupedGames = payload.groups || [];
        const list = document.getElementById('games-list');
        if (!list) return;
        list.innerHTML = '';

        const iterationGroups = (groupedGames || []).filter(g => g.kind === 'iteration' || (g.iteration !== undefined && g.kind !== 'recorded'));
        if (iterationGroups.length === 0) {
            list.innerHTML = '<p style="color: var(--text-muted); text-align: center; font-size: 0.85rem; padding: 0.5rem 0;">No training games stored yet.</p>';
            return;
        }

        iterationGroups.forEach((group, groupIdx) => {
            const details = document.createElement('details');
            details.className = 'iteration-group';
            if (groupIdx === 0) details.open = true; // Open the most recent by default

            const iterFolder = group.folder || `iter_${String(group.iteration).padStart(6, '0')}`;
            const summary = document.createElement('summary');
            summary.className = 'group-summary-2row';
            summary.innerHTML = `
                <div class="summary-row-top">
                    <span class="summary-title">Iteration ${group.iteration}</span>
                </div>
                <div class="summary-row-bottom">
                    <div class="summary-row-left">
                        <span class="group-note">${group.total_games} game${group.total_games === 1 ? '' : 's'}</span>
                    </div>
                    <button class="btn-group-delete" title="Delete Iteration ${group.iteration}" aria-label="Delete">✕</button>
                </div>
            `;
            const iterDelBtn = summary.querySelector('.btn-group-delete');
            if (iterDelBtn) {
                iterDelBtn.addEventListener('click', async (e) => {
                    e.stopPropagation();
                    e.preventDefault();
                    await fetch(`/training/api/games/${encodeURIComponent(iterFolder)}`, { method: 'DELETE' });
                    loadGamesList();
                });
            }
            details.appendChild(summary);

            (group.phases || []).forEach((phase, phaseIdx) => {
                const phaseEl = document.createElement('details');
                phaseEl.className = 'phase-group';
                if (groupIdx === 0 && (phaseIdx === 0 || phase.phase === 'promotion')) {
                    phaseEl.open = true;
                }

                const phaseFolder = phase.folder || `${iterFolder}/${phase.phase}`;
                const phaseSummary = document.createElement('summary');
                phaseSummary.className = 'group-summary-2row';
                phaseSummary.innerHTML = `
                    <div class="summary-row-top">
                        <span class="summary-title">${phase.label}</span>
                        <span class="summary-badge-wrap">${gamesPhaseBadge(phase)}</span>
                    </div>
                    <div class="summary-row-bottom">
                        <div class="summary-row-left">
                            <span class="group-note">${phase.count} game${phase.count === 1 ? '' : 's'}</span>
                        </div>
                        <button class="btn-group-delete" title="Delete ${phase.label}" aria-label="Delete">✕</button>
                    </div>
                `;
                const phaseDelBtn = phaseSummary.querySelector('.btn-group-delete');
                if (phaseDelBtn) {
                    phaseDelBtn.addEventListener('click', async (e) => {
                        e.stopPropagation();
                        e.preventDefault();
                        const enc = phaseFolder.split('/').map(encodeURIComponent).join('/');
                        await fetch(`/training/api/games/${enc}`, { method: 'DELETE' });
                        loadGamesList();
                    });
                }
                phaseEl.appendChild(phaseSummary);

                phase.games.forEach(game => {
                    const item = document.createElement('div');
                    item.className = 'game-item';

                    const winnerIcon = game.winner === 1 ? '⚫' : (game.winner === 2 ? '⚪' : '🤝');
                    const icon = (c) => (c === 1 ? '⚫' : '⚪');

                    let label = `Game ${game.game_index}`;
                    let resultColor = 'var(--text-muted)';

                    if (phase.phase === 'promotion') {
                        label = `Promo ${game.game_index} (${icon(game.candidate_color)} Cand vs ${icon(game.champion_color)} Champ)`;
                        if (game.winner !== 0) {
                            resultColor = game.candidate_won ? 'var(--success)' : 'var(--danger)';
                        }
                    } else if (phase.phase === 'eval' || game.is_eval) {
                        label = `Eval ${game.game_index} (${icon(game.network_color)} AI vs ${icon(game.network_color === 1 ? 2 : 1)} Rand)`;
                        if (game.network_color !== undefined && game.winner !== 0) {
                            resultColor = (game.winner === game.network_color) ? 'var(--success)' : 'var(--danger)';
                        }
                    }

                    let resultText = 'Draw';
                    if (game.winner === 1) resultText = `B+${game.margin || '?'}`;
                    else if (game.winner === 2) resultText = `W+${game.margin || '?'}`;
                    // Resigned games have no margin — B+R / W+R, as in Go.
                    if (game.resignation && game.resignation.resigned && game.resignation.result) {
                        resultText = game.resignation.result;
                    }

                    const body = document.createElement('div');
                    body.className = 'game-item-body';
                    body.style.cursor = 'pointer';
                    body.innerHTML = `
                        <div style="font-weight: 500;">${winnerIcon} ${label}${gameResignTag(game)}</div>
                        <div class="meta"><span style="color: ${resultColor}; font-weight: 600;">${resultText}</span> &middot; ${game.num_moves} moves</div>
                    `;
                    body.addEventListener('click', () => {
                        window.location.href = `/training/review?game=${encodeURIComponent(game.filename)}`;
                    });
                    item.appendChild(body);

                    const del = document.createElement('button');
                    del.className = 'btn-game-delete';
                    del.title = 'Delete this game';
                    del.textContent = '✕';
                    del.addEventListener('click', async (e) => {
                        e.stopPropagation();
                        const enc = game.filename.split('/').map(encodeURIComponent).join('/');
                        await fetch(`/training/api/games/${enc}`, { method: 'DELETE' });
                        loadGamesList();
                    });
                    item.appendChild(del);

                    phaseEl.appendChild(item);
                });

                details.appendChild(phaseEl);
            });

            list.appendChild(details);
        });

        // Say what is NOT here, so a missing iteration reads as a deliberate
        // cut-off rather than as data that went missing.
        const page = payload.pagination || {};
        if (page.has_more) {
            const note = document.createElement('p');
            note.className = 'games-list-note';
            note.innerHTML = `Showing the last ${page.returned} iterations &middot; ` +
                `<a href="/training/review">review all ${page.total_iterations}</a>`;
            list.appendChild(note);
        }
    } catch (e) { /* ignore */ }
}

// ---- Learning Stats ----
function formatDuration(seconds) {
    if (seconds === null || seconds === undefined || isNaN(seconds)) return '—';
    if (seconds < 60) return seconds.toFixed(1) + 's';
    if (seconds < 3600) {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}m ${s}s`;
    }
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    return `${h}h ${m}m ${s}s`;
}

let cachedLearningStats = null;
let currentSelfPlayFilter = 'all';

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
    if (typeof updateMarginDispersion === 'function') {
        updateMarginDispersion(series);
    }
}

function setupSelfPlayFilter() {
    const spBtns = document.querySelectorAll('#selfplay-filter-btns .filter-btn');
    spBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            spBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentSelfPlayFilter = btn.dataset.limit || 'all';
            renderSelfPlayStats(cachedLearningStats);
        });
    });
}

async function loadLearningStats() {
    try {
        const res = await fetch('/training/api/learning_stats');
        const stats = await res.json();
        cachedLearningStats = stats;

        const el = (id) => document.getElementById(id);

        if (el('ls-avg-time-total')) el('ls-avg-time-total').textContent = formatDuration(stats.avg_time_per_game_total);
        if (el('ls-iter-time-avg')) el('ls-iter-time-avg').textContent = formatDuration(stats.iter_time_avg);
        if (el('ls-game-length')) {
            el('ls-game-length').textContent = (stats.avg_game_length !== null && stats.avg_game_length !== undefined)
                ? `${stats.avg_game_length} moves` : '—';
        }
        if (el('ls-buffer')) {
            el('ls-buffer').textContent = (stats.buffer_size !== null && stats.buffer_size !== undefined)
                ? `${stats.buffer_size.toLocaleString()} / ${stats.buffer_capacity.toLocaleString()}` : '—';
        }
        if (el('ls-lr')) {
            el('ls-lr').textContent = (stats.learning_rate !== null && stats.learning_rate !== undefined)
                ? Number(stats.learning_rate.toPrecision(3)).toString() : '—';
        }

        // Hero tiles & Eval vs Random text line
        if (el('t-device')) el('t-device').textContent = (stats.device || '—').toUpperCase();
        
        if (el('ls-wr-random')) {
            if (stats.latest_win_rate_vs_random != null) {
                const pct = Math.round(stats.latest_win_rate_vs_random * 100);
                el('ls-wr-random').innerHTML = `<span class="eval-random-pill">${pct}%</span> <small>(${stats.random_ai_wins || 0}W / ${(stats.random_ai_wins || 0) + (stats.random_ai_losses || 0)}G)</small>`;
            } else {
                el('ls-wr-random').textContent = '—';
            }
        }

        renderSelfPlayStats(stats);
        renderTimeMetrics(stats);
    } catch (e) { /* ignore */ }
}

// ---- Milestone Toast ----
function showMilestone(text) {
    const toast = document.getElementById('milestone-toast');
    if (!toast) return;
    const msgEl = document.getElementById('milestone-text');
    if (msgEl) msgEl.textContent = `🎯 ${text}`;
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

let tunerParamBounds = null;

async function initTunerParamSliders(values = {}) {
    const container = document.getElementById('tuner-param-categories');
    if (!container) return;
    if (!tunerParamBounds) {
        tunerParamBounds = await getParamBounds();
    }
    if (!tunerParamBounds) return;

    container.innerHTML = buildParamSlidersHTML('tune', tunerParamBounds, values);
    bindParamSliders('tune', tunerParamBounds);
    setParamSliderValues('tune', tunerParamBounds, values);
}

async function prefillTuner() {
    try {
        const res = await fetch('/models/api/active');
        const model = await res.json();
        if (!model || !model.training) return;
        await initTunerParamSliders(model.training);
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
        const payload = extractParamSliderValues('tune', tunerParamBounds);

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
                loadLearningStats();
            }
        } catch (e) {
            showToast('⚠ Failed to connect to server');
        }
        btnTuneApply.disabled = false;
        btnTuneApply.textContent = '✓ Apply';
    });
}

// ---- Metrics table (full history, scrollable) ----
let allMetrics = [];
let currentModelId = null;

function pushMetric(m) {
    if (m == null || m.iteration == null) return;
    const at = allMetrics.findIndex(x => x.iteration === m.iteration);
    if (at !== -1) {
        allMetrics[at] = { ...allMetrics[at], ...m };
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
    const rows = allMetrics.slice().sort((a, b) => (a.iteration || 0) - (b.iteration || 0));
    body.innerHTML = rows.map(m => {
        let total = m.total_loss;
        if (total == null && m.policy_loss != null && m.value_loss != null) total = m.policy_loss + m.value_loss;

        // Gate Win Rate & Promotion Status
        let gateText = '—';
        if (m.gate_win_rate != null) {
            const gatePct = Math.round(m.gate_win_rate * 100) + '%';
            if (m.gate_promoted === true) {
                gateText = `<span class="gate-tag promo" title="Promoted to Champion">${gatePct} ✓</span>`;
            } else if (m.gate_promoted === false) {
                gateText = `<span class="gate-tag reject" title="Rejected by gate">${gatePct} ✗</span>`;
            } else {
                gateText = gatePct;
            }
        }

        // Mercy Resign Rate
        let mercyText = '—';
        if (m.resign_suppressed) {
            mercyText = '<span class="mercy-tag guard" title="Suppressed by collapse guard">Guard</span>';
        } else if (m.resign_rate != null) {
            mercyText = Math.round(m.resign_rate * 100) + '%';
        }

        // Wrong Resignation Rate
        let wrongText = '—';
        if (m.false_resign_rate != null) {
            const wrPct = Math.round(m.false_resign_rate * 100) + '%';
            wrongText = m.false_resign_rate > 0.05
                ? `<span class="wrong-tag danger" title="Wrong resignation rate exceeds 5% danger threshold">${wrPct}</span>`
                : `<span class="wrong-tag safe">${wrPct}</span>`;
        }

        // Value Head Spread (Black / White)
        let vstdText = '—';
        if (m.value_std_black != null && m.value_std_white != null) {
            vstdText = `${Number(m.value_std_black).toFixed(2)} / ${Number(m.value_std_white).toFixed(2)}`;
        } else if (m.value_std_black != null) {
            vstdText = `${Number(m.value_std_black).toFixed(2)} / —`;
        }

        // Duration
        const durText = m.elapsed_seconds != null ? formatDuration(m.elapsed_seconds) : '—';

        return `<tr>
            <td class="cell-iter">#${m.iteration != null ? m.iteration : '—'}</td>
            <td>${fmtLoss(m.policy_loss)}</td>
            <td>${fmtLoss(m.value_loss)}</td>
            <td>${fmtLoss(total)}</td>
            <td>${gateText}</td>
            <td>${mercyText}</td>
            <td>${wrongText}</td>
            <td>${vstdText}</td>
            <td>${durText}</td>
        </tr>`;
    }).join('');
}

function metricsTableVisible() {
    const card = document.getElementById('metrics-table-card');
    return card && !card.hidden && card.style.display !== 'none';
}

function setupMetricsViewToggle() {
    const toggle = document.getElementById('metrics-view-toggle');
    if (!toggle) return;
    toggle.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            toggle.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const charts = document.getElementById('metrics-charts');
            const table = document.getElementById('metrics-table-card');
            const resignCard = document.getElementById('resign-card');
            if (btn.dataset.view === 'table') {
                if (charts) {
                    charts.hidden = true;
                    charts.style.display = 'none';
                }
                if (resignCard) {
                    resignCard.hidden = true;
                    resignCard.style.display = 'none';
                }
                if (table) {
                    table.hidden = false;
                    table.style.display = 'block';
                }
                renderMetricsTable();
            } else {
                if (table) {
                    table.hidden = true;
                    table.style.display = 'none';
                }
                if (charts) {
                    charts.hidden = false;
                    charts.style.display = '';
                    if (typeof policyLossChart !== 'undefined' && policyLossChart) policyLossChart.resize();
                    if (typeof valueLossChart !== 'undefined' && valueLossChart) valueLossChart.resize();
                }
                loadResignStats();
            }
        });
    });
}

// ---- Champion Lineage (Promotion Gate) & Hero Updates ----
/**
 * Mercy-rule panel. Stays hidden unless the rule is on or has produced data
 * before, so a model that has never used it carries no dead UI.
 * Also hidden when the table view is selected (only visible with chart toggle on).
 */
async function loadResignStats() {
    const card = document.getElementById('resign-card');
    if (!card) return;
    try {
        const res = await fetch('/training/api/resign_stats');
        const { enabled, has_data, points, summary, danger_rate } = await res.json();

        const toggle = document.getElementById('metrics-view-toggle');
        const activeBtn = toggle ? toggle.querySelector('.filter-btn.active') : null;
        const isTableView = activeBtn && activeBtn.dataset.view === 'table';

        if (!enabled && !has_data || isTableView) {
            card.hidden = true;
            card.style.display = 'none';
            return;
        }
        card.hidden = false;
        card.style.display = '';

        const set = (id, txt) => {
            const el = document.getElementById(id);
            if (el) el.textContent = txt;
        };
        const pct = (v) => (v == null ? '—' : `${Math.round(v * 100)}%`);

        // Verdict banner — the one-line answer to "should this be on?"
        const banner = document.getElementById('resign-verdict');
        if (banner) banner.className = `resign-verdict is-${summary.verdict}`;
        set('resign-headline', summary.headline || '—');
        set('resign-detail', summary.detail || '');

        set('resign-rate', pct(summary.resign_rate));
        set('resign-counts', `${summary.total_resigned} of ${summary.total_games} games`);

        set('resign-false-rate', summary.false_resign_rate == null
            ? '—' : pct(summary.false_resign_rate));
        set('resign-false-counts', summary.checked_games
            ? `${summary.false_resigns} of ${summary.checked_games} checked`
                + (summary.ci_high != null ? ` · up to ${pct(summary.ci_high)}` : '')
            : 'no checks yet');

        set('resign-saved', summary.est_moves_saved == null
            ? '—' : `~${summary.est_moves_saved.toLocaleString()}`);
        set('resign-saved-avg', summary.avg_moves_saved == null
            ? 'measured from playout games'
            : `~${summary.avg_moves_saved} moves per resigned game`);

        set('resign-checks', `${summary.checked_games}`);
        set('resign-checks-hint', summary.checked_games < summary.min_checks
            ? `need ${summary.min_checks} to judge`
            : 'enough to judge');

        const hint = document.getElementById('resign-threshold-hint');
        if (hint && summary.threshold != null) {
            hint.textContent = summary.suppressed
                ? 'Suppressed (collapse guard)'
                : `Resign below ${Math.round((1 - summary.threshold) * 50)}% win rate`;
        }

        if (typeof updateResignChart === 'function') {
            updateResignChart(points || [], danger_rate);
        }
    } catch (e) { /* ignore */ }
}

async function loadGateHistory() {
    const panel = document.getElementById('gate-panel');
    if (!panel) return;
    try {
        const res = await fetch('/training/api/gate_history');
        const { points, summary } = await res.json();

        const empty = document.getElementById('gate-empty');
        const hasData = Array.isArray(points) && points.length > 0;
        if (empty) empty.style.display = hasData ? 'none' : '';

        if (typeof updateGateChart === 'function') {
            updateGateChart(points || [], summary.gate_threshold);
        }

        const set = (id, txt) => {
            const el = document.getElementById(id);
            if (el) el.textContent = txt;
        };

        if (!hasData) {
            ['gate-promotions', 'gate-avg', 'gate-streak'].forEach(id => set(id, '—'));
            set('gate-promo-rate', 'no gated iterations yet');
            set('gate-last-promo', '—');
            set('gate-champ-version', 'v1');
            
            // Hero defaults
            set('t-champ-version', 'v1');
            set('t-hero-promotions', '0 Promotions');
            set('t-hero-gate-elo', '+0 Elo');
            set('t-hero-promo-rate', '—');
            set('t-hero-streak', '—');
            return;
        }

        const elo = summary.gate_elo || 0;
        const eloTxt = `${elo >= 0 ? '+' : ''}${Math.round(elo)} Elo`;
        const promoCount = summary.promotions || 0;
        const totalGated = summary.gated_iterations || 0;
        const promoRatePct = Math.round((summary.promotion_rate || 0) * 100);
        const streak = summary.current_reject_streak || 0;
        const champVer = `v${summary.champion_version || 1}`;

        set('gate-promotions', `${promoCount}/${totalGated}`);
        set('gate-promo-rate', `${promoRatePct}% accepted`);
        set('gate-avg', `${Math.round((summary.avg_gate_win_rate || 0) * 100)}%`);
        set('gate-champ-version', champVer);

        set('gate-streak', streak === 0 ? 'just promoted' : `${streak} iter`);
        set('gate-last-promo',
            summary.last_promotion_iteration
                ? `last at iter ${summary.last_promotion_iteration}`
                : 'no promotions yet');

        // Update Hero Card & Tiles
        set('t-champ-version', champVer);
        set('t-hero-promotions', `${promoCount} Promotion${promoCount === 1 ? '' : 's'}`);
        set('t-hero-gate-elo', eloTxt);
        set('t-hero-promo-rate', `${promoRatePct}% (${promoCount}/${totalGated})`);
        set('t-hero-streak', streak === 0 ? 'Just Promoted 🚀' : `${streak} iter streak`);

        // Highlight streak if stalling
        const streakEl = document.getElementById('gate-streak');
        const heroStreakEl = document.getElementById('t-hero-streak');
        if (streakEl) {
            streakEl.style.color = streak >= 5 ? 'var(--danger)' : streak >= 3 ? 'var(--warning)' : '';
        }
        if (heroStreakEl) {
            heroStreakEl.style.color = streak >= 5 ? 'var(--danger)' : streak >= 3 ? 'var(--warning)' : '';
        }
    } catch (e) { /* ignore */ }
}

// ---- Initial Load ----
async function loadHistoricalMetrics() {
    try {
        const res = await fetch('/training/api/metrics');
        const metrics = await res.json();
        allMetrics = Array.isArray(metrics) ? metrics.slice() : [];
        resetCharts();
        for (const m of metrics) {
            updateCharts(m);
        }
        renderMetricsTable();
    } catch (e) { /* ignore */ }
}

let activeTimeGraphFilters = {
    self_play: true,
    nn_train: true,
    random_eval: true,
    champion_gate: true,
    total: true,
};

function renderTimeMetrics(stats) {
    if (!stats || !stats.time_metrics) return;
    const tm = stats.time_metrics;
    const summary = tm.summary || {};
    const history = tm.history || [];
    const el = (id) => document.getElementById(id);

    const lastIterTotal = summary.last_iter_total || 0.1;

    // Summary badges
    if (el('tm-last-iter-total')) el('tm-last-iter-total').textContent = formatDuration(summary.last_iter_total);
    if (el('tm-all-time-total')) el('tm-all-time-total').textContent = formatDuration(summary.all_time_total);

    // Tab 1: Overview Table
    const spLast = summary.sp_total_last || 0;
    const nnLast = summary.nn_total_last || 0;
    const randLast = summary.rand_total_last || 0;
    const champLast = summary.champ_total_last || 0;

    const spPct = Math.round((spLast / lastIterTotal) * 100);
    const nnPct = Math.round((nnLast / lastIterTotal) * 100);
    const randPct = Math.round((randLast / lastIterTotal) * 100);
    const champPct = Math.round((champLast / lastIterTotal) * 100);

    if (el('tm-sp-bar')) el('tm-sp-bar').style.width = `${Math.min(100, spPct)}%`;
    if (el('tm-sp-pct')) el('tm-sp-pct').textContent = `${spPct}%`;
    if (el('tm-sp-total-last')) el('tm-sp-total-last').textContent = formatDuration(summary.sp_total_last);
    if (el('tm-sp-total-all')) el('tm-sp-total-all').textContent = formatDuration(summary.sp_total_all);

    if (el('tm-nn-bar')) el('tm-nn-bar').style.width = `${Math.min(100, nnPct)}%`;
    if (el('tm-nn-pct')) el('tm-nn-pct').textContent = `${nnPct}%`;
    if (el('tm-nn-total-last')) el('tm-nn-total-last').textContent = formatDuration(summary.nn_total_last);
    if (el('tm-nn-total-all')) el('tm-nn-total-all').textContent = formatDuration(summary.nn_total_all);

    if (el('tm-rand-bar')) el('tm-rand-bar').style.width = `${Math.min(100, randPct)}%`;
    if (el('tm-rand-pct')) el('tm-rand-pct').textContent = `${randPct}%`;
    if (el('tm-rand-total-last')) el('tm-rand-total-last').textContent = formatDuration(summary.rand_total_last);
    if (el('tm-rand-total-all')) el('tm-rand-total-all').textContent = formatDuration(summary.rand_total_all);

    if (el('tm-champ-bar')) el('tm-champ-bar').style.width = `${Math.min(100, champPct)}%`;
    if (el('tm-champ-pct')) el('tm-champ-pct').textContent = `${champPct}%`;
    if (el('tm-champ-total-last')) el('tm-champ-total-last').textContent = formatDuration(summary.champ_total_last);
    if (el('tm-champ-total-all')) el('tm-champ-total-all').textContent = formatDuration(summary.champ_total_all);

    // Tab 2: Chart
    if (typeof updateTimeBreakdownChart === 'function') {
        updateTimeBreakdownChart(history, activeTimeGraphFilters);
    }

    // Tab 3: Time History Table
    const body = el('time-history-table-body');
    const empty = el('time-history-empty');
    if (body) {
        if (!history.length) {
            body.innerHTML = '';
            if (empty) empty.style.display = '';
        } else {
            if (empty) empty.style.display = 'none';
            const rows = history.slice().reverse();
            body.innerHTML = rows.map(h => `
                <tr>
                    <td><span class="iter-tag">Iter ${h.iteration}</span></td>
                    <td class="time-cell-total"><strong>${formatDuration(h.total_time)}</strong></td>
                    <td><span class="time-chip chip-sp">${formatDuration(h.self_play_time)}</span></td>
                    <td><span class="time-chip chip-nn">${formatDuration(h.nn_train_time)}</span></td>
                    <td><span class="time-chip chip-rand">${formatDuration(h.random_eval_time)}</span></td>
                    <td><span class="time-chip chip-champ">${formatDuration(h.champion_gate_time)}</span></td>
                </tr>
            `).join('');
        }
    }
}

function setupTimeMetricsTabsAndFilters() {
    const tabBtns = document.querySelectorAll('#time-metrics-tabs .time-tab-btn');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation(); // prevent collapsing panel when switching tabs
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const selected = btn.dataset.tab;
            const contents = {
                summary: document.getElementById('time-tab-summary'),
                graphs: document.getElementById('time-tab-graphs'),
                history: document.getElementById('time-tab-history'),
            };

            Object.keys(contents).forEach(key => {
                if (contents[key]) {
                    if (key === selected) {
                        contents[key].classList.add('active');
                        contents[key].style.display = 'block';
                    } else {
                        contents[key].classList.remove('active');
                        contents[key].style.display = 'none';
                    }
                }
            });

            if (selected === 'graphs' && typeof timeBreakdownChart !== 'undefined' && timeBreakdownChart) {
                timeBreakdownChart.resize();
                timeBreakdownChart.update();
            }
        });
    });

    const filterBtns = document.querySelectorAll('#time-graph-toggles .filter-btn');
    filterBtns.forEach(btn => {
        const seriesKey = btn.dataset.series;
        btn.classList.toggle('active', !!activeTimeGraphFilters[seriesKey]);

        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            btn.classList.toggle('active');
            activeTimeGraphFilters[seriesKey] = btn.classList.contains('active');
            if (cachedLearningStats && cachedLearningStats.time_metrics) {
                if (typeof updateTimeBreakdownChart === 'function') {
                    updateTimeBreakdownChart(cachedLearningStats.time_metrics.history, activeTimeGraphFilters);
                }
            }
        });
    });
}

// ---- Collapsible Blocks Logic ----
function setupCollapsibleBlocks() {
    const blocks = document.querySelectorAll('.collapsible-block');
    blocks.forEach(block => {
        const header = block.querySelector('.collapsible-header');
        const content = block.querySelector('.collapsible-content');
        const chevron = block.querySelector('.collapse-chevron');
        if (!header || !content) return;

        // Initialize state
        const startsCollapsed = block.classList.contains('is-collapsed') || block.dataset.collapsed === 'true';
        if (startsCollapsed) {
            block.classList.add('is-collapsed');
            content.style.display = 'none';
            if (chevron) chevron.textContent = '▶';
        } else {
            block.classList.remove('is-collapsed');
            content.style.display = 'block';
            if (chevron) chevron.textContent = '▼';
        }

        // Toggle handler
        header.addEventListener('click', (e) => {
            // Don't toggle if clicking inside an interactive child (e.g. tabs, view toggle buttons)
            if (e.target.closest('button:not(.collapse-toggle), .time-tabs, .graph-filter-buttons, a, input, select')) {
                return;
            }

            const isCurrentlyCollapsed = block.classList.contains('is-collapsed');
            if (isCurrentlyCollapsed) {
                // Expanding
                block.classList.remove('is-collapsed');
                content.style.display = 'block';
                if (chevron) chevron.textContent = '▼';

                // Re-render / resize any canvas inside this block so Chart.js isn't squished
                const canvases = block.querySelectorAll('canvas');
                canvases.forEach(canvas => {
                    const chart = (typeof Chart !== 'undefined') ? Chart.getChart(canvas) : null;
                    if (chart) {
                        chart.resize();
                        chart.update('none');
                    }
                });

                // Trigger specific charts if needed
                if (block.id === 'gate-panel' && typeof gateChart !== 'undefined' && gateChart) {
                    gateChart.resize();
                }
                if (block.id === 'nn-health-panel') {
                    if (typeof policyLossChart !== 'undefined' && policyLossChart) policyLossChart.resize();
                    if (typeof valueLossChart !== 'undefined' && valueLossChart) valueLossChart.resize();
                }
                if (block.id === 'time-metrics-panel' && typeof timeBreakdownChart !== 'undefined' && timeBreakdownChart) {
                    timeBreakdownChart.resize();
                }
                if (block.id === 'pipeline-stats-panel' && typeof selfplayMarginChart !== 'undefined' && selfplayMarginChart) {
                    selfplayMarginChart.resize();
                }
            } else {
                // Collapsing
                block.classList.add('is-collapsed');
                content.style.display = 'none';
                if (chevron) chevron.textContent = '▶';
            }
        });
    });
}

// Initial Status loader (prevents fallback on page refresh)
function loadInitialStatus() {
    const scriptEl = document.getElementById('initial-status-data');
    if (scriptEl && scriptEl.textContent) {
        try {
            const initialData = JSON.parse(scriptEl.textContent);
            if (initialData && Object.keys(initialData).length > 0) {
                renderLiveStatus(initialData);
            }
        } catch (e) { /* ignore */ }
    }
    // Also fetch status from REST API to ensure latest server state
    fetch('/training/api/status')
        .then(r => r.json())
        .then(data => {
            if (data && !data.error) {
                renderLiveStatus(data);
            }
        })
        .catch(() => {});
}

// Request current status on load via socket
if (typeof socket !== 'undefined' && socket) {
    socket.emit('request_status');
}

// Initialize components
loadInitialStatus();
setupCollapsibleBlocks();
loadHistoricalMetrics();
loadGamesList();
loadLearningStats();
loadGateHistory();
loadResignStats();
setupSelfPlayFilter();
setupTimeMetricsTabsAndFilters();
setupMetricsViewToggle();

