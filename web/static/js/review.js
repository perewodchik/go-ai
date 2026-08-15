/**
 * review.js — Handles the Review Games page (Game Titles & Territory Estimation).
 */

let reviewBoard = null;
let currentGameData = null;
let currentMoveIndex = 0;
let winrateChart = null;
let winrateBlack = [];   // base series: Black's win % per move
let winrateSide = 1;     // perspective shown: 1 = Black, 2 = White
let showTerritoryOverlay = false;  // coloured overlay on the board (numbers show regardless); off by default

document.addEventListener('DOMContentLoaded', () => {
    loadGamesList();

    document.getElementById('btn-refresh').addEventListener('click', loadGamesList);

    document.getElementById('btn-first').addEventListener('click', () => jumpToMove(0));
    document.getElementById('btn-prev').addEventListener('click', () => jumpToMove(currentMoveIndex - 1));
    document.getElementById('btn-next').addEventListener('click', () => jumpToMove(currentMoveIndex + 1));
    document.getElementById('btn-last').addEventListener('click', () => {
        if (currentGameData) jumpToMove(currentGameData.moves.length);
    });

    const btnSideBlack = document.getElementById('winrate-side-black');
    const btnSideWhite = document.getElementById('winrate-side-white');
    if (btnSideBlack) btnSideBlack.addEventListener('click', () => setWinrateSide(1));
    if (btnSideWhite) btnSideWhite.addEventListener('click', () => setWinrateSide(2));

    // Toggle the coloured territory overlay on the board without touching the
    // numbers in the Territory panel (those are always shown).
    const btnOverlay = document.getElementById('btn-territory-overlay');
    if (btnOverlay) {
        const syncOverlayBtn = () => {
            btnOverlay.classList.toggle('active', showTerritoryOverlay);
            btnOverlay.textContent = showTerritoryOverlay ? '👁 Overlay' : '🚫 Overlay';
        };
        syncOverlayBtn();  // reflect the default (off) on load
        btnOverlay.addEventListener('click', () => {
            showTerritoryOverlay = !showTerritoryOverlay;
            syncOverlayBtn();
            if (reviewBoard) {
                reviewBoard.showEstimate = showTerritoryOverlay;
                reviewBoard.draw();
            }
        });
    }

    // Check URL for game parameter
    const params = new URLSearchParams(window.location.search);
    const requestedGame = params.get('game');
    if (requestedGame) {
        loadGame(requestedGame);
    }
});

async function loadGamesList() {
    const list = document.getElementById('review-games-list');
    try {
        const res = await fetch('/training/api/games?include_recorded=1');
        const groupedGames = await res.json();
        list.innerHTML = '';

        if (groupedGames.length === 0) {
            list.innerHTML = '<p style="color: var(--text-muted); text-align: center;">No games available.</p>';
            return;
        }

        groupedGames.forEach((group, groupIdx) => {
            // Recorded human games are a flat list, not an iteration of phases.
            if (group.kind === 'recorded') {
                list.appendChild(buildRecordedGroup(group, groupIdx === 0));
                return;
            }

            // Bot vs bot matches: one collapsible section per series.
            if (group.kind === 'match') {
                list.appendChild(buildMatchGroup(group, groupIdx === 0));
                return;
            }

            const details = document.createElement('details');
            details.className = 'iteration-group';
            if (groupIdx === 0) details.open = true;

            const summary = document.createElement('summary');
            summary.innerHTML = `
                <span>Iteration ${group.iteration}</span>
                <span class="group-note">${group.total_games} games</span>
            `;
            details.appendChild(summary);

            // Second level: one collapsible section per phase of the iteration.
            (group.phases || []).forEach((phase, phaseIdx) => {
                const phaseEl = document.createElement('details');
                phaseEl.className = 'phase-group';
                // Open the promotion section by default on the newest iteration —
                // it's the match that decided whether the model moved forward.
                if (groupIdx === 0 && (phaseIdx === 0 || phase.phase === 'promotion')) {
                    phaseEl.open = true;
                }

                const phaseSummary = document.createElement('summary');
                phaseSummary.innerHTML = `
                    <span>${phase.label} <span class="group-note">${phase.count}</span></span>
                    ${phaseSummaryBadge(phase)}
                `;
                phaseEl.appendChild(phaseSummary);

                phase.games.forEach(game => {
                    phaseEl.appendChild(buildGameItem(game, phase.phase));
                });

                details.appendChild(phaseEl);
            });

            list.appendChild(details);
        });
    } catch (e) {
        list.innerHTML = '<p style="color: var(--text-muted); text-align: center;">Error loading games.</p>';
    }
}

/** Colour a win rate: above 50% good, below 50% bad. */
function winRateColor(rate, threshold = 0.5) {
    if (rate > threshold) return 'var(--success)';
    if (rate < threshold) return 'var(--danger)';
    return 'var(--warning)';
}

/** Right-hand badge on a phase header: candidate win rate, or AI win rate. */
function phaseSummaryBadge(phase) {
    if (phase.phase === 'promotion') {
        const rate = phase.candidate_win_rate !== undefined && phase.candidate_win_rate !== null
            ? phase.candidate_win_rate
            : phase.gate_win_rate;
        if (rate === undefined || rate === null) return '';

        const threshold = phase.gate_threshold || 0.5;
        const pct = Math.round(rate * 100);
        return `<span class="group-note" style="color: ${winRateColor(rate, threshold)}; font-weight: 700; font-size: 0.95rem; white-space: nowrap;">
            ${pct}%
        </span>`;
    }

    if (phase.phase === 'eval' && phase.win_rate !== null && phase.win_rate !== undefined) {
        const pct = Math.round(phase.win_rate * 100);
        return `<span class="group-note" style="color: ${winRateColor(phase.win_rate)}; font-weight: 700; font-size: 0.95rem; white-space: nowrap;">
            ${pct}%
        </span>`;
    }

    return '';
}

/** The "My Recorded Games" section — games saved from the Play page. */
function buildRecordedGroup(group, open) {
    const details = document.createElement('details');
    details.className = 'iteration-group recorded-group';
    details.open = open;

    const summary = document.createElement('summary');
    summary.innerHTML = `
        <span>🎮 ${group.label}</span>
        <span class="group-note">${group.total_games} games</span>
    `;
    details.appendChild(summary);

    group.games.forEach(game => details.appendChild(buildRecordedGameItem(game)));
    return details;
}

/** The "Bot vs Bot Matches" section — one sub-section per match series. */
function buildMatchGroup(group, open) {
    const details = document.createElement('details');
    details.className = 'iteration-group match-group';
    details.open = open;

    const summary = document.createElement('summary');
    summary.innerHTML = `
        <span>🤖 ${group.label}</span>
        <span class="group-note">${group.total_games} games</span>
    `;
    details.appendChild(summary);

    (group.series || []).forEach((series, idx) => {
        const seriesEl = document.createElement('details');
        seriesEl.className = 'phase-group match-series-group';
        if (open && idx === 0) seriesEl.open = true;

        const seriesSummary = document.createElement('summary');
        seriesSummary.innerHTML = `
            <span>${escapeHtml(series.name)} <span class="group-note">${series.count}</span></span>
            <span class="group-note">${matchSeriesScore(series)}</span>
        `;
        seriesEl.appendChild(seriesSummary);

        series.games.forEach(game => seriesEl.appendChild(buildMatchGameItem(game)));
        details.appendChild(seriesEl);
    });

    return details;
}

/**
 * Series score as "3–1", counted per PLAYER (colours alternate between games,
 * so a per-colour tally would be meaningless).
 */
function matchSeriesScore(series) {
    const names = [];
    const wins = {};
    series.games.forEach(game => {
        const black = (game.black_player || {}).name || 'Black';
        const white = (game.white_player || {}).name || 'White';
        [black, white].forEach(name => {
            if (!names.includes(name)) names.push(name);
            if (wins[name] === undefined) wins[name] = 0;
        });
        if (game.winner === 1) wins[black] += 1;
        else if (game.winner === 2) wins[white] += 1;
    });

    if (names.length < 2) return '';
    return `${wins[names[0]]}–${wins[names[1]]}`;
}

/** One match game row: who played which colour, and who won. */
function buildMatchGameItem(game) {
    const item = document.createElement('div');
    item.className = 'game-item review-list-item match-game-item';
    if (currentGameData && currentGameData.filename === game.filename) {
        item.classList.add('active');
    }

    const black = (game.black_player || {}).name || 'Black';
    const white = (game.white_player || {}).name || 'White';

    let resultText = 'Draw';
    let resultColor = 'var(--text-muted)';
    if (game.winner === 1 || game.winner === 2) {
        const winnerName = game.winner === 1 ? black : white;
        const how = game.resigned_by
            ? 'by resignation'
            : `+${game.margin !== undefined ? Number(game.margin).toFixed(1) : '?'}`;
        resultText = `${game.winner === 1 ? '⚫' : '⚪'} ${winnerName} won ${how}`;
        resultColor = 'var(--text-primary)';
    }

    item.innerHTML = `
        <div style="font-weight: 600; margin-bottom: 0.15rem;">Game ${(game.game_index || 0) + 1}</div>
        <div style="font-size: 0.85rem; color: ${resultColor}; font-weight: 600; margin-bottom: 0.15rem;">${escapeHtml(resultText)}</div>
        <div class="match-game-line">⚫ ${escapeHtml(black)} vs ⚪ ${escapeHtml(white)} &middot; ${game.num_moves} moves</div>
    `;
    item.addEventListener('click', () => selectGame(item, game.filename));

    return item;
}

/** One recorded game row: result from the human's point of view, plus delete. */
function buildRecordedGameItem(game) {
    const item = document.createElement('div');
    item.className = 'game-item review-list-item';
    if (currentGameData && currentGameData.filename === game.filename) {
        item.classList.add('active');
    }

    const humanColor = game.human_color === 2 ? 2 : 1;
    let resultText = 'Draw';
    let resultColor = 'var(--text-muted)';
    if (game.unfinished) {
        resultText = 'Unfinished';
    } else if (game.winner === 1 || game.winner === 2) {
        const won = game.winner === humanColor;
        const margin = game.resigned_by
            ? 'by resignation'
            : `by ${game.margin !== undefined ? Number(game.margin).toFixed(1) : '?'}`;
        resultText = `${won ? 'You won' : 'Bot won'} ${margin}`;
        resultColor = won ? 'var(--success)' : 'var(--danger)';
    }

    const title = game.name && game.name.length
        ? game.name
        : `${humanColor === 1 ? '⚫' : '⚪'} You vs Bot`;
    const when = game.timestamp ? new Date(game.timestamp).toLocaleString() : '';

    const body = document.createElement('div');
    body.className = 'recorded-item-body';
    body.innerHTML = `
        <div style="font-weight: 600; margin-bottom: 0.15rem;">${escapeHtml(title)}</div>
        <div style="font-size: 0.85rem; color: ${resultColor}; font-weight: 600; margin-bottom: 0.15rem;">${resultText}</div>
        <div style="font-size: 0.8rem; color: var(--text-muted);">${game.num_moves} moves${when ? ` &middot; ${when}` : ''}</div>
    `;
    body.addEventListener('click', () => selectGame(item, game.filename));
    item.appendChild(body);

    const del = document.createElement('button');
    del.className = 'btn-small recorded-delete';
    del.title = 'Delete this recorded game';
    del.textContent = '🗑';
    del.addEventListener('click', async (e) => {
        // The row itself opens the game — don't do both on a delete click.
        e.stopPropagation();
        if (!confirm(`Delete "${title}"? This cannot be undone.`)) return;
        await deleteRecordedGame(game.filename);
    });
    item.appendChild(del);

    return item;
}

async function deleteRecordedGame(filename) {
    const encodedPath = filename.split('/').map(encodeURIComponent).join('/');
    const res = await fetch(`/training/api/games/${encodedPath}`, { method: 'DELETE' });
    if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        alert(data.error || 'Failed to delete game.');
        return;
    }

    // If the deleted game was open, clear the viewer — its data is gone.
    if (currentGameData && currentGameData.filename === filename) {
        currentGameData = null;
        document.getElementById('review-viewer').style.display = 'none';
        document.getElementById('empty-state').style.display = '';
        const url = new URL(window.location);
        url.searchParams.delete('game');
        window.history.replaceState({}, '', url);
    }
    loadGamesList();
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

/** Open a game and mark its row active. */
function selectGame(item, filename) {
    document.querySelectorAll('.game-item').forEach(el => el.classList.remove('active'));
    item.classList.add('active');

    const url = new URL(window.location);
    url.searchParams.set('game', filename);
    window.history.pushState({}, '', url);

    loadGame(filename);
}

/** One clickable game row inside a phase section. */
function buildGameItem(game, phase) {
    const item = document.createElement('div');
    item.className = 'game-item review-list-item';

    if (currentGameData && currentGameData.filename === game.filename) {
        item.classList.add('active');
    }

    const winnerIcon = game.winner === 1 ? '⚫' : (game.winner === 2 ? '⚪' : '🤝');
    const colorStr = (c) => (c === 1 ? '⚫ Black' : '⚪ White');

    let resultColor = 'var(--text-muted)';
    let resultText = 'Draw';
    if (game.winner === 1) resultText = `B+${game.margin !== undefined ? Number(game.margin).toFixed(1) : '?'}`;
    else if (game.winner === 2) resultText = `W+${game.margin !== undefined ? Number(game.margin).toFixed(1) : '?'}`;

    if (phase === 'promotion') {
        if (game.winner !== 0) {
            resultColor = game.candidate_won ? 'var(--success)' : 'var(--danger)';
        }
        let outcomeLine = 'Draw';
        if (game.winner !== 0) {
            const winnerWho = game.candidate_won ? 'Candidate' : 'Champion';
            outcomeLine = `${winnerWho} won - ${resultText}`;
        }

        item.innerHTML = `
            <div style="font-weight: 600; margin-bottom: 0.15rem;">${winnerIcon} Promotion #${game.game_index}</div>
            <div style="font-size: 0.85rem; color: ${resultColor}; font-weight: 600; margin-bottom: 0.15rem;">${outcomeLine}</div>
            <div style="font-size: 0.85rem; color: var(--text-muted);">${game.num_moves} moves</div>
        `;
    } else {
        let label;
        if (phase === 'eval' || game.is_eval) {
            label = `Eval #${game.game_index} (AI as ${colorStr(game.network_color)} vs RandomBot)`;
            if (game.network_color !== undefined && game.winner !== 0) {
                resultColor = (game.winner === game.network_color) ? 'var(--success)' : 'var(--danger)';
            }
        } else {
            label = `Self-Play #${game.game_index}`;
        }

        item.innerHTML = `
            <div style="margin-bottom: 0.25rem;">
                <strong>${winnerIcon} ${label}</strong>
            </div>
            <div style="font-size: 0.85rem; color: var(--text-muted);">
                <span style="color: ${resultColor}; font-weight: 600;">${resultText}</span> &middot; ${game.num_moves} moves
            </div>
        `;
    }

    item.addEventListener('click', () => selectGame(item, game.filename));

    return item;
}

async function loadGame(filename) {
    try {
        const cacheBuster = new Date().getTime();
        // The id is a path under games/ (iter_000001/promotion/promo_0000.json),
        // so escape each segment but keep the separators.
        const encodedPath = filename.split('/').map(encodeURIComponent).join('/');
        const res = await fetch(`/training/api/games/${encodedPath}?t=${cacheBuster}`);
        if (!res.ok) throw new Error('Game not found');

        const data = await res.json();
        data.filename = filename;
        currentGameData = data;

        document.getElementById('empty-state').style.display = 'none';
        document.getElementById('review-viewer').style.display = 'block';

        const canvas = document.getElementById('review-board');
        const size = data.board_size || 9;

        reviewBoard = new GoBoardRenderer(canvas, size, null, { enableHover: false });

        // Informative Game Title & Meta
        let titleText = '';
        let resultText = 'Draw';
        let winnerIcon = '🤝';
        const komi = data.komi || 6.5;

        if (data.winner === 1) {
            resultText = `B+${data.margin !== undefined ? Number(data.margin).toFixed(1) : '?'}`;
            winnerIcon = '⚫';
        } else if (data.winner === 2) {
            resultText = `W+${data.margin !== undefined ? Number(data.margin).toFixed(1) : '?'}`;
            winnerIcon = '⚪';
        }

        if (data.phase === 'human') {
            const humanColor = data.human_color === 2 ? 2 : 1;
            const humanIcon = humanColor === 1 ? '⚫' : '⚪';
            const botIcon = humanColor === 1 ? '⚪' : '⚫';
            const humanWon = data.winner === humanColor;

            let outcomeStr, outcomeClass;
            if (data.unfinished) {
                outcomeStr = '⏸ Unfinished';
                outcomeClass = 'outcome-neutral';
            } else if (data.winner === 0) {
                outcomeStr = '🤝 Draw';
                outcomeClass = 'outcome-draw';
            } else {
                const how = data.resigned_by ? 'by resignation' : `(${resultText})`;
                outcomeStr = humanWon ? `🎉 You Won ${how}` : `❌ Bot Won ${how}`;
                outcomeClass = humanWon ? 'outcome-win' : 'outcome-loss';
            }

            const when = data.timestamp ? new Date(data.timestamp).toLocaleString() : '';
            titleText = data.name && data.name.length ? data.name : 'Recorded Game · You vs Bot';
            document.getElementById('review-meta').innerHTML = `
                <span class="review-meta-pill ${outcomeClass}">${outcomeStr}</span>
                <span class="review-meta-pill">You ${humanIcon} vs Bot ${botIcon}</span>
                <span class="review-meta-pill">♟️ ${data.moves.length} moves</span>
                <span class="review-meta-pill">⚖️ Komi ${komi}</span>
                ${when ? `<span class="review-meta-pill">🕒 ${when}</span>` : ''}
            `;
        } else if (data.phase === 'match') {
            const black = (data.black_player || {}).name || 'Black';
            const white = (data.white_player || {}).name || 'White';
            let outcomeStr = '🤝 Draw';
            let outcomeClass = 'outcome-draw';
            if (data.winner === 1 || data.winner === 2) {
                const winnerName = data.winner === 1 ? black : white;
                const how = data.resigned_by ? 'by resignation' : `(${resultText})`;
                outcomeStr = `${winnerIcon} ${winnerName} won ${how}`;
                outcomeClass = 'outcome-neutral';
            }

            const when = data.timestamp ? new Date(data.timestamp).toLocaleString() : '';
            titleText = `${data.match_name || 'Bot vs Bot'} · Game ${(data.game_index || 0) + 1}`;
            document.getElementById('review-meta').innerHTML = `
                <span class="review-meta-pill ${outcomeClass}">${escapeHtml(outcomeStr)}</span>
                <span class="review-meta-pill">⚫ ${escapeHtml(black)} vs ⚪ ${escapeHtml(white)}</span>
                <span class="review-meta-pill">♟️ ${data.moves.length} moves</span>
                <span class="review-meta-pill">⚖️ Komi ${komi}</span>
                ${when ? `<span class="review-meta-pill">🕒 ${when}</span>` : ''}
            `;
        } else if (data.phase === 'promotion') {
            const candIcon = data.candidate_color === 1 ? '⚫' : '⚪';
            const champIcon = data.champion_color === 1 ? '⚫' : '⚪';
            const outcomeStr = data.winner === 0
                ? '🤝 Draw'
                : (data.candidate_won ? `🎉 Candidate Won (${resultText})` : `🛡 Champion Held (${resultText})`);
            const outcomeClass = data.winner === 0 ? 'outcome-draw' : (data.candidate_won ? 'outcome-win' : 'outcome-loss');

            titleText = `Iteration ${data.iteration} · Promotion Match #${data.game_index || 0}`;
            document.getElementById('review-meta').innerHTML = `
                <span class="review-meta-pill ${outcomeClass}">${outcomeStr}</span>
                <span class="review-meta-pill">Cand ${candIcon} vs Champ ${champIcon}</span>
                <span class="review-meta-pill">♟️ ${data.moves.length} moves</span>
                <span class="review-meta-pill">⚖️ Komi ${komi}</span>
            `;
        } else if (data.is_eval) {
            const aiIcon = data.network_color === 1 ? '⚫' : '⚪';
            const randIcon = data.network_color === 1 ? '⚪' : '⚫';
            const aiWon = data.winner === data.network_color;
            const outcomeStr = (data.winner === 0)
                ? '🤝 Draw'
                : (aiWon ? `🎉 Go AI Won (${resultText})` : `❌ RandomBot Won (${resultText})`);
            const outcomeClass = data.winner === 0 ? 'outcome-draw' : (aiWon ? 'outcome-win' : 'outcome-loss');

            titleText = `Iteration ${data.iteration} · Eval Match #${data.game_index || 1}`;
            document.getElementById('review-meta').innerHTML = `
                <span class="review-meta-pill ${outcomeClass}">${outcomeStr}</span>
                <span class="review-meta-pill">Go AI ${aiIcon} vs RandomBot ${randIcon}</span>
                <span class="review-meta-pill">♟️ ${data.moves.length} moves</span>
                <span class="review-meta-pill">⚖️ Komi ${komi}</span>
            `;
        } else {
            const outcomeClass = data.winner === 0 ? 'outcome-draw' : 'outcome-neutral';
            titleText = `Iteration ${data.iteration} · Self-Play Game #${data.game_index || 1}`;
            document.getElementById('review-meta').innerHTML = `
                <span class="review-meta-pill ${outcomeClass}">${winnerIcon} ${resultText}</span>
                <span class="review-meta-pill">♟️ ${data.moves.length} moves</span>
                <span class="review-meta-pill">⚖️ Komi ${komi}</span>
            `;
        }

        document.getElementById('review-title').textContent = titleText;

        initWinrateChart(data);

        jumpToMove(0);

    } catch (e) {
        alert("Failed to load game: " + filename);
    }
}

// Palette per perspective so the curve's color reinforces whose win % is shown.
const WINRATE_STYLE = {
    1: { name: 'Black', line: '#c8956c', fill: 'rgba(200, 149, 108, 0.12)', point: '#dbb08a' },
    2: { name: 'White', line: '#cdd3dd', fill: 'rgba(205, 211, 221, 0.12)', point: '#e8ebf0' },
};

/** Series for the currently selected perspective (White = 100 − Black). */
function winrateSeries() {
    return winrateSide === 2
        ? winrateBlack.map((v) => Math.round((100 - v) * 10) / 10)
        : winrateBlack.slice();
}

/** Build (or rebuild) the per-move win-rate line chart for the loaded game. */
function initWinrateChart(data) {
    const container = document.getElementById('review-winrate-container');
    const canvas = document.getElementById('review-winrate-chart');
    if (!container || !canvas || typeof Chart === 'undefined') return;

    if (winrateChart) {
        winrateChart.destroy();
        winrateChart = null;
    }

    winrateBlack = Array.isArray(data.win_rates) ? data.win_rates : [];
    if (winrateBlack.length === 0) {
        // No eval data available (e.g. board-size mismatch) — hide the chart.
        container.style.display = 'none';
        return;
    }
    container.style.display = 'block';

    // Default perspective: show the side whose strength is being judged — the
    // AI in eval games, the candidate in promotion matches; else Black.
    if (data.phase === 'human' && (data.human_color === 1 || data.human_color === 2)) {
        winrateSide = data.human_color;
    } else if (data.phase === 'promotion' && (data.candidate_color === 1 || data.candidate_color === 2)) {
        winrateSide = data.candidate_color;
    } else if (data.is_eval && (data.network_color === 1 || data.network_color === 2)) {
        winrateSide = data.network_color;
    } else {
        winrateSide = 1;
    }

    const labels = winrateBlack.map((_, i) => i);
    const style = WINRATE_STYLE[winrateSide];
    const series = winrateSeries();
    const gridColor = 'rgba(255, 255, 255, 0.06)';
    const tickColor = '#9a9a9a';     // --text-secondary

    winrateChart = new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: `${style.name} Win %`,
                data: series,
                borderColor: style.line,
                backgroundColor: style.fill,
                borderWidth: 2,
                fill: true,
                tension: 0.35,
                pointRadius: series.map(() => 0),
                pointHoverRadius: 5,
                pointBackgroundColor: style.point,
                pointBorderColor: style.point,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            interaction: { mode: 'index', intersect: false },
            onClick: (evt, elements, chart) => {
                // `elements` is already resolved via the configured index/no-intersect
                // interaction; fall back to an explicit lookup just in case.
                let idx = (elements && elements.length) ? elements[0].index : null;
                if (idx === null) {
                    const pts = chart.getElementsAtEventForMode(evt, 'index', { intersect: false }, true);
                    if (pts.length > 0) idx = pts[0].index;
                }
                if (idx !== null) jumpToMove(idx);
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title: (items) => `Move ${items[0].label}`,
                        label: (item) => `${WINRATE_STYLE[winrateSide].name}: ${Number(item.raw).toFixed(1)}%`,
                    },
                },
            },
            scales: {
                x: {
                    title: { display: true, text: 'Move Number', color: tickColor },
                    grid: { color: gridColor },
                    ticks: { color: tickColor, maxTicksLimit: 12 },
                },
                y: {
                    min: 0,
                    max: 100,
                    title: { display: true, text: `${style.name} Win %`, color: tickColor },
                    grid: { color: gridColor },
                    ticks: { color: tickColor, stepSize: 25, callback: (v) => `${v}%` },
                },
            },
        },
    });

    updateWinrateSideButtons();
}

/** Switch the chart between Black's and White's perspective. */
function setWinrateSide(side) {
    if (side === winrateSide) return;
    winrateSide = side;
    if (!winrateChart) {
        updateWinrateSideButtons();
        return;
    }

    const style = WINRATE_STYLE[side];
    const ds = winrateChart.data.datasets[0];
    ds.data = winrateSeries();
    ds.label = `${style.name} Win %`;
    ds.borderColor = style.line;
    ds.backgroundColor = style.fill;
    ds.pointBorderColor = style.point;
    winrateChart.options.scales.y.title.text = `${style.name} Win %`;
    winrateChart.update('none');

    updateWinrateSideButtons();
    highlightWinrateMove(currentMoveIndex);
}

/** Reflect the active perspective on the toggle buttons. */
function updateWinrateSideButtons() {
    const btnB = document.getElementById('winrate-side-black');
    const btnW = document.getElementById('winrate-side-white');
    [[btnB, 1], [btnW, 2]].forEach(([btn, side]) => {
        if (btn) btn.classList.toggle('active', side === winrateSide);
    });
}

/** Emphasize the dot for the active move on the win-rate chart. */
function highlightWinrateMove(index) {
    const label = document.getElementById('winrate-current');
    if (!winrateChart) {
        if (label) label.textContent = '';
        return;
    }
    const style = WINRATE_STYLE[winrateSide];
    const data = winrateChart.data.datasets[0].data;
    // The chart has a point per move (0 .. n-1); the terminal position (index n)
    // has no eval, so clamp the highlight to the last available point.
    const active = Math.min(index, data.length - 1);

    winrateChart.data.datasets[0].pointRadius = data.map((_, i) => (i === active ? 5 : 0));
    winrateChart.data.datasets[0].pointBackgroundColor = data.map((_, i) => (i === active ? '#fff' : style.point));
    winrateChart.update('none');

    if (label && active >= 0 && active < data.length) {
        label.textContent = `${style.name} ${Number(data[active]).toFixed(1)}%`;
    } else if (label) {
        label.textContent = '';
    }
}

/** Render the always-visible territory estimate for the current position. */
function updateTerritoryPanel(est) {
    const panel = document.getElementById('territory-panel');
    if (!panel) return;

    const total = est.blackScore + est.whiteScore;
    const blackPct = total > 0 ? (est.blackScore / total) * 100 : 50;
    document.getElementById('territory-bar-black').style.width = `${blackPct.toFixed(1)}%`;
    document.getElementById('territory-bar-white').style.width = `${(100 - blackPct).toFixed(1)}%`;

    document.getElementById('territory-black-score').textContent = est.blackScore.toFixed(1);
    document.getElementById('territory-white-score').textContent = est.whiteScore.toFixed(1);
    // One <span> per component: laid out inline on desktop, stacked on mobile.
    document.getElementById('territory-black-detail').innerHTML =
        `<span>${est.blackStones} stones</span><span>${est.blackTerritory} territory</span>`;
    document.getElementById('territory-white-detail').innerHTML =
        `<span>${est.whiteStones} stones</span><span>${est.whiteTerritory} territory</span><span>${est.komi} komi</span>`;

    const leadEl = document.getElementById('territory-lead');
    leadEl.textContent = est.lead;
    leadEl.className = 'territory-lead ' + (
        est.lead.startsWith('B') ? 'lead-black' : (est.lead.startsWith('W') ? 'lead-white' : 'lead-tie')
    );
}

function jumpToMove(index) {
    if (!currentGameData || !reviewBoard) return;

    currentMoveIndex = Math.max(0, Math.min(index, currentGameData.moves.length));

    if (currentGameData.states && currentGameData.states[currentMoveIndex]) {
        reviewBoard.grid = currentGameData.states[currentMoveIndex];
    } else {
        const size = currentGameData.board_size;
        reviewBoard.grid = Array.from({ length: size }, () => Array(size).fill(0));
    }

    let lastMove = null;
    if (currentMoveIndex > 0) {
        const m = currentGameData.moves[currentMoveIndex - 1];
        if (m.move[0] >= 0 && m.move[1] >= 0) {
            lastMove = { row: m.move[0], col: m.move[1] };
        }
    }

    reviewBoard.lastMove = lastMove;

    // Territory numbers always update in the panel; the coloured board overlay
    // follows the user's toggle (see #btn-territory-overlay).
    reviewBoard.showEstimate = showTerritoryOverlay;
    updateTerritoryPanel(reviewBoard.computeTerritory(currentGameData.komi || 6.5));

    reviewBoard.draw();

    highlightWinrateMove(currentMoveIndex);

    const curStr = String(currentMoveIndex).padStart(3, '\u00A0');
    const totalStr = String(currentGameData.moves.length).padStart(3, '\u00A0');
    document.getElementById('move-counter').textContent = `${curStr} / ${totalStr}`;

    document.getElementById('btn-first').disabled = currentMoveIndex === 0;
    document.getElementById('btn-prev').disabled = currentMoveIndex === 0;
    document.getElementById('btn-next').disabled = currentMoveIndex === currentGameData.moves.length;
    document.getElementById('btn-last').disabled = currentMoveIndex === currentGameData.moves.length;
}

window.addEventListener('resize', () => {
    if (reviewBoard) {
        reviewBoard.resize();
    }
});

document.addEventListener('keydown', (e) => {
    if (!currentGameData || !reviewBoard) return;
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    if (e.key === 'ArrowLeft') {
        e.preventDefault();
        const step = e.shiftKey ? 10 : 1;
        jumpToMove(currentMoveIndex - step);
    } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        const step = e.shiftKey ? 10 : 1;
        jumpToMove(currentMoveIndex + step);
    }
});
