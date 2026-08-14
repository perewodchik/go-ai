/**
 * review.js — Handles the Review Games page (Game Titles & Territory Estimation).
 */

let reviewBoard = null;
let currentGameData = null;
let currentMoveIndex = 0;
let winrateChart = null;
let winrateBlack = [];   // base series: Black's win % per move
let winrateSide = 1;     // perspective shown: 1 = Black, 2 = White
let showTerritoryOverlay = true;  // coloured overlay on the board (numbers show regardless)

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
        btnOverlay.addEventListener('click', () => {
            showTerritoryOverlay = !showTerritoryOverlay;
            btnOverlay.classList.toggle('active', showTerritoryOverlay);
            btnOverlay.textContent = showTerritoryOverlay ? '👁 Overlay' : '🚫 Overlay';
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
        const res = await fetch('/training/api/games');
        const groupedGames = await res.json();
        list.innerHTML = '';

        if (groupedGames.length === 0) {
            list.innerHTML = '<p style="color: var(--text-muted); text-align: center;">No games available.</p>';
            return;
        }

        groupedGames.forEach((group, groupIdx) => {
            const details = document.createElement('details');
            details.className = 'iteration-group';
            if (groupIdx === 0) details.open = true;

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
                item.className = 'game-item review-list-item';

                // Highlight if currently selected
                if (currentGameData && currentGameData.filename === game.filename) {
                    item.classList.add('active');
                }

                const winnerIcon = game.winner === 1 ? '⚫' : (game.winner === 2 ? '⚪' : '🤝');
                let label = '';

                if (game.is_eval) {
                    const aiColorStr = game.network_color === 1 ? '⚫ Black' : '⚪ White';
                    label = `Eval #${game.game_index} (AI as ${aiColorStr} vs RandomBot)`;
                } else {
                    label = `Self-Play #${game.game_index}`;
                }

                let resultText = 'Draw';
                if (game.winner === 1) resultText = `B+${game.margin !== undefined ? Number(game.margin).toFixed(1) : '?'}`;
                else if (game.winner === 2) resultText = `W+${game.margin !== undefined ? Number(game.margin).toFixed(1) : '?'}`;

                let resultColor = 'var(--text-muted)';
                if (game.is_eval && game.network_color !== undefined && game.winner !== 0) {
                    resultColor = (game.winner === game.network_color) ? 'var(--success)' : 'var(--danger)';
                }

                item.innerHTML = `
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                        <strong>${winnerIcon} ${label}</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: var(--text-muted);">
                        <span><span style="color: ${resultColor}; font-weight: 600;">${resultText}</span> &middot; ${game.num_moves} moves</span>
                        <span>${new Date(game.timestamp).toLocaleTimeString()}</span>
                    </div>
                `;

                item.addEventListener('click', () => {
                    document.querySelectorAll('.game-item').forEach(el => el.classList.remove('active'));
                    item.classList.add('active');

                    const url = new URL(window.location);
                    url.searchParams.set('game', game.filename);
                    window.history.pushState({}, '', url);

                    loadGame(game.filename);
                });

                details.appendChild(item);
            });
            list.appendChild(details);
        });
    } catch (e) {
        list.innerHTML = '<p style="color: var(--text-muted); text-align: center;">Error loading games.</p>';
    }
}

async function loadGame(filename) {
    try {
        const cacheBuster = new Date().getTime();
        const res = await fetch(`/training/api/games/${filename}?t=${cacheBuster}`);
        if (!res.ok) throw new Error('Game not found');

        const data = await res.json();
        data.filename = filename;
        currentGameData = data;

        document.getElementById('empty-state').style.display = 'none';
        document.getElementById('review-viewer').style.display = 'block';

        const canvas = document.getElementById('review-board');
        const size = data.board_size || 9;

        reviewBoard = new GoBoardRenderer(canvas, size);

        // Informative Game Title & Meta
        let titleText = '';
        let resultText = '';
        const komi = data.komi || 6.5;

        if (data.winner === 1) resultText = `Black wins (B+${data.margin !== undefined ? Number(data.margin).toFixed(1) : '?'})`;
        else if (data.winner === 2) resultText = `White wins (W+${data.margin !== undefined ? Number(data.margin).toFixed(1) : '?'})`;
        else resultText = 'Draw';

        if (data.is_eval) {
            const aiColor = data.network_color === 1 ? '⚫ Black' : '⚪ White';
            const randColor = data.network_color === 1 ? '⚪ White' : '⚫ Black';
            const outcomeStr = (data.winner === data.network_color) ? '🎉 Go AI Won' : '❌ RandomBot Won';
            titleText = `Iteration ${data.iteration} · Eval Match #${data.game_index || 1}`;
            document.getElementById('review-meta').innerHTML = `<strong>${outcomeStr}</strong> (${resultText}) &middot; Go AI (${aiColor}) vs RandomBot (${randColor}) &middot; ${data.moves.length} moves &middot; Komi ${komi}`;
        } else {
            titleText = `Iteration ${data.iteration} · Self-Play Game #${data.game_index || 1}`;
            document.getElementById('review-meta').innerHTML = `<strong>${resultText}</strong> &middot; ${data.moves.length} moves &middot; Komi ${komi}`;
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

    // Default perspective: for eval games (AI vs RandomBot), show the AI's side;
    // otherwise default to Black.
    if (data.is_eval && (data.network_color === 1 || data.network_color === 2)) {
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

    // Stones in atari can flip the estimate next move, so flag them — but only then.
    const atariEl = document.getElementById('territory-atari');
    const parts = [];
    if (est.atariStonesBlack > 0) parts.push(`⚫ ${est.atariStonesBlack}`);
    if (est.atariStonesWhite > 0) parts.push(`⚪ ${est.atariStonesWhite}`);
    if (parts.length > 0) {
        atariEl.textContent = `⚠ In atari: ${parts.join('  ')}`;
        atariEl.style.display = 'block';
    } else {
        atariEl.style.display = 'none';
    }
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
