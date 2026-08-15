/**
 * game.js — Game interaction logic for human vs bot play.
 *
 * Manages: setup, move submission, pass/resign/undo, score display,
 * territory estimation toggle, easy/hard mode, game over.
 */

let board = null;
let gameId = null;
let gameMode = 'hard';
let playerColor = 'black';
// True while a human move is being processed and the bot is replying. Blocks
// further board/pass input so a fast double-click can't play the bot's move.
let inputLocked = false;
// Live win-rate readout: easy mode only, and off until the user asks for it.
let showWinRate = false;
let winrateChart = null;

// ---- Setup ----
document.querySelectorAll('.color-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.color-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        playerColor = btn.dataset.color;
    });
});

document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        gameMode = btn.dataset.mode;
    });
});

document.getElementById('simulations').addEventListener('input', (e) => {
    document.getElementById('sim-label').textContent = e.target.value;
});

document.getElementById('start-game').addEventListener('click', startGame);

async function startGame() {
    const numSim = parseInt(document.getElementById('simulations').value);

    const res = await fetch('/api/game/new', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            mode: gameMode,
            player_color: playerColor,
            num_simulations: numSim,
        }),
    });

    const data = await res.json();
    if (data.error) {
        alert(data.error);
        return;
    }
    
    gameId = data.game_id;

    // Initialize board renderer
    const canvas = document.getElementById('go-board');
    board = new GoBoardRenderer(canvas, data.board_size, onBoardClick);

    // Show/hide easy mode buttons
    document.querySelectorAll('.easy-only').forEach(el => {
        el.style.display = gameMode === 'easy' ? '' : 'none';
    });

    // Switch panels
    PlayViews.show('human-game');
    document.getElementById('btn-new-game').style.display = 'none';
    // Let the launcher offer to resume this game if the user navigates away.
    PlayViews.setHumanGame({
        label: `Your game vs ${data.model_name || 'the bot'}`,
        over: false,
    });

    // Fresh game — drop the previous game's curve rather than extending it.
    if (winrateChart) {
        winrateChart.destroy();
        winrateChart = null;
    }
    syncWinRatePanel();

    updateBoard(data.state);
    await refreshAnalysis();
}

// ---- Gameplay ----
async function onBoardClick(row, col) {
    // Ignore clicks until it's the human's turn again — this is what stops a
    // fast double-click from submitting a move for the bot's color.
    if (!gameId || inputLocked) return;
    inputLocked = true;
    try {
        const res = await fetch('/api/game/move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ game_id: gameId, row, col }),
        });

        const data = await res.json();
        if (data.error) {
            // Subtle flash — illegal move (or not your turn)
            document.getElementById('game-status').textContent = data.error;
            setTimeout(() => document.getElementById('game-status').textContent = 'Your turn', 1500);
            return;
        }

        updateBoard(data.state);
        if (data.scores) updateScores(data.scores);
        await refreshAnalysis();

        if (data.state.is_over) {
            showGameOver(data);
        } else {
            await fetchBotMove();
        }
    } finally {
        inputLocked = false;
    }
}

async function fetchBotMove() {
    document.getElementById('game-status').textContent = 'Bot thinking...';
    const res = await fetch('/api/game/bot_move', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ game_id: gameId }),
    });

    if (!res.ok) return;
    const data = await res.json();

    if (data.state) updateBoard(data.state);
    if (data.scores) updateScores(data.scores);
    await refreshAnalysis();
    if (data.state && data.state.is_over) showGameOver(data);
}

document.getElementById('btn-pass').addEventListener('click', async () => {
    if (!gameId || inputLocked) return;
    inputLocked = true;
    try {
        const res = await fetch('/api/game/pass', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ game_id: gameId }),
        });
        const data = await res.json();
        updateBoard(data.state);
        if (data.scores) updateScores(data.scores);
        await refreshAnalysis();

        if (data.result || (data.state && data.state.is_over)) {
            showGameOver(data);
        } else {
            await fetchBotMove();
        }
    } finally {
        inputLocked = false;
    }
});

document.getElementById('btn-resign').addEventListener('click', async () => {
    if (!confirm('Are you sure you want to resign?')) return;
    const res = await fetch('/api/game/resign', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ game_id: gameId }),
    });
    const data = await res.json();
    updateBoard(data.state);
    showGameOver(data);
});

document.getElementById('btn-undo').addEventListener('click', async () => {
    const res = await fetch('/api/game/undo', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ game_id: gameId }),
    });
    const data = await res.json();
    if (data.error) return;
    updateBoard(data.state);
    await refreshAnalysis();
});

document.getElementById('btn-suggest').addEventListener('click', async () => {
    document.getElementById('btn-suggest').textContent = 'Thinking...';
    const res = await fetch('/api/game/suggest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ game_id: gameId }),
    });
    const data = await res.json();
    document.getElementById('btn-suggest').textContent = 'Suggest';
    if (data.suggestion && data.suggestion !== 'pass') {
        board.suggestedMove = { row: data.suggestion[0], col: data.suggestion[1] };
        board.draw();
        // Clear after 3 seconds
        setTimeout(() => { board.suggestedMove = null; board.draw(); }, 3000);
    }
});

// Territory estimation toggle
document.getElementById('toggle-estimate').addEventListener('change', async (e) => {
    board.showEstimate = e.target.checked;
    if (e.target.checked) {
        await refreshEstimateIfEnabled();
    } else {
        board.setOwnershipMap(null);
    }
});

// Re-fetch and re-render the territory overlay after any move, if enabled.
// Without this, the overlay silently goes stale after a move — it only
// reflected the board state from when the toggle was last switched on.
async function refreshEstimateIfEnabled() {
    if (!board || !board.showEstimate || !gameId) return;

    const res = await fetch('/api/game/estimate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ game_id: gameId }),
    });
    if (!res.ok) return;
    const data = await res.json();
    board.setOwnershipMap(data.ownership_map);
    document.getElementById('black-score').textContent = data.black_estimate?.toFixed(1) || '0';
    document.getElementById('white-score').textContent = data.white_estimate?.toFixed(1) || '0';
}

/** Refresh every post-move analysis overlay that is currently switched on. */
async function refreshAnalysis() {
    await refreshEstimateIfEnabled();
    await refreshWinRateIfEnabled();
}

// ---- Live win rate (easy mode only, off by default) ----
document.getElementById('toggle-winrate')?.addEventListener('change', async (e) => {
    showWinRate = e.target.checked;
    syncWinRatePanel();
    if (showWinRate) await refreshWinRateIfEnabled();
});

/** Show the win-rate panel only while it's both enabled and available. */
function syncWinRatePanel() {
    const panel = document.getElementById('play-winrate-panel');
    if (!panel) return;
    const visible = showWinRate && gameMode === 'easy' && !!gameId;
    panel.style.display = visible ? '' : 'none';
}

async function refreshWinRateIfEnabled() {
    // Hard mode gets no evaluation at all — that's the point of hard mode.
    if (!showWinRate || gameMode !== 'easy' || !gameId) return;

    const res = await fetch('/api/game/winrate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ game_id: gameId }),
    });
    if (!res.ok) return;
    const data = await res.json();
    renderWinRate(data.win_rates || []);
}

/**
 * Draw the win-rate curve from the human's point of view.
 *
 * The server returns Black's win probability per position (the same series the
 * review page charts), so playing White is a straight 100 − x flip.
 */
function renderWinRate(blackRates) {
    const canvas = document.getElementById('play-winrate-chart');
    const label = document.getElementById('play-winrate-current');
    if (!canvas || typeof Chart === 'undefined') return;

    const series = playerColor === 'white'
        ? blackRates.map(v => Math.round((100 - v) * 10) / 10)
        : blackRates.slice();

    if (label) {
        label.textContent = series.length
            ? `You ${Number(series[series.length - 1]).toFixed(1)}%`
            : '';
    }

    if (winrateChart) {
        winrateChart.data.labels = series.map((_, i) => i);
        winrateChart.data.datasets[0].data = series;
        winrateChart.update('none');
        return;
    }

    const style = playerColor === 'white'
        ? { line: '#cdd3dd', fill: 'rgba(205, 211, 221, 0.12)' }
        : { line: '#c8956c', fill: 'rgba(200, 149, 108, 0.12)' };
    const tickColor = '#9a9a9a';
    const gridColor = 'rgba(255, 255, 255, 0.06)';

    winrateChart = new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: {
            labels: series.map((_, i) => i),
            datasets: [{
                label: 'Your Win %',
                data: series,
                borderColor: style.line,
                backgroundColor: style.fill,
                borderWidth: 2,
                fill: true,
                tension: 0.35,
                pointRadius: 0,
                pointHoverRadius: 4,
            }],
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
                        title: (items) => `Move ${items[0].label}`,
                        label: (item) => `You: ${Number(item.raw).toFixed(1)}%`,
                    },
                },
            },
            scales: {
                x: { grid: { color: gridColor }, ticks: { color: tickColor, maxTicksLimit: 8 } },
                y: {
                    min: 0,
                    max: 100,
                    grid: { color: gridColor },
                    ticks: { color: tickColor, stepSize: 25, callback: (v) => `${v}%` },
                },
            },
        },
    });
}

// ---- Recording ----
document.getElementById('btn-record')?.addEventListener('click', async () => {
    if (!gameId) return;

    const name = prompt('Name for this recorded game (optional):', defaultRecordName());
    if (name === null) return;  // cancelled

    const btn = document.getElementById('btn-record');
    const original = btn.textContent;
    btn.disabled = true;
    btn.textContent = 'Saving...';

    try {
        const res = await fetch('/api/game/record', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ game_id: gameId, name }),
        });
        const data = await res.json();
        if (!res.ok) {
            showToast(data.error || 'Failed to record game');
            return;
        }
        showToast('Game recorded — find it in Review Games');
    } finally {
        btn.disabled = false;
        btn.textContent = original;
    }
});

function defaultRecordName() {
    const when = new Date().toLocaleString();
    return `${playerColor === 'black' ? '⚫' : '⚪'} vs Bot — ${when}`;
}

function showToast(text) {
    const toast = document.getElementById('play-toast');
    if (!toast) return;
    toast.textContent = text;
    toast.style.display = '';
    clearTimeout(showToast._timer);
    showToast._timer = setTimeout(() => { toast.style.display = 'none'; }, 3500);
}

document.getElementById('btn-new-game')?.addEventListener('click', () => {
    gameId = null;
    PlayViews.setHumanGame(null);
    PlayViews.show('human-setup');
    syncWinRatePanel();
});

// ---- Helpers ----
function updateBoard(state) {
    board.updateState(state);
    document.getElementById('move-counter').textContent = `Move: ${state.move_number}`;
    document.getElementById('black-captures').textContent = `Cap: ${state.prisoners['1'] || 0}`;
    document.getElementById('white-captures').textContent = `Cap: ${state.prisoners['2'] || 0}`;

    if (state.is_over) {
        document.getElementById('game-status').textContent = 'Game Over';
        document.getElementById('btn-new-game').style.display = '';
        if (window.PlayViews && gameId) {
            PlayViews.setHumanGame({ label: 'Your game', over: true });
        }
    } else {
        const isMyTurn = (state.current_player === 1 && playerColor === 'black') ||
                         (state.current_player === 2 && playerColor === 'white');
        document.getElementById('game-status').textContent = isMyTurn ? 'Your turn' : 'Bot thinking...';
    }
}

function updateScores(scores) {
    document.getElementById('black-score').textContent = scores.black?.toFixed(1) || '0';
    document.getElementById('white-score').textContent = scores.white?.toFixed(1) || '0';
}

function showGameOver(data) {
    const modal = document.getElementById('game-over-modal');
    const result = data.result || {};

    let title = 'Game Over';
    let details = '';
    if (result.winner === 'draw') {
        title = 'Draw (Jigo)';
    } else if (result.reason === 'resignation') {
        title = `${result.winner === playerColor ? 'You Win!' : 'Bot Wins!'}`;
        details = 'By resignation';
    } else {
        title = `${result.winner === playerColor ? 'You Win!' : 'Bot Wins!'}`;
        details = `By ${result.margin?.toFixed(1) || '?'} points`;
    }

    document.getElementById('result-title').textContent = title;
    document.getElementById('result-details').textContent = details;
    document.getElementById('final-black').textContent = result.black_score?.toFixed(1) || data.scores?.black?.toFixed(1) || '?';
    document.getElementById('final-white').textContent = result.white_score?.toFixed(1) || data.scores?.white?.toFixed(1) || '?';
    document.getElementById('btn-new-game').style.display = '';
    modal.style.display = 'flex';
}

window.addEventListener('resize', () => {
    if (board) {
        board.resize();
    }
});
