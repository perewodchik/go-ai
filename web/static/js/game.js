/**
 * game.js — Game interaction logic for human vs bot play.
 *
 * Manages: setup (including which model you play), move submission,
 * pass/resign/undo, score display, territory estimation toggle, easy/hard
 * mode, game over.
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
// Opponent picker state: every model in the workspace, and the one this game
// is actually being played against.
let opponentModels = [];
let opponent = null;
let gameOver = false;

// ---- Opponent picker ----
//
// You can play ANY model, not just the one the Dashboard has active — that one
// is only the default selection.

async function loadPlayableModels() {
    const select = document.getElementById('play-model');
    if (!select || opponentModels.length) return;

    let data;
    try {
        const res = await fetch('/api/game/opponents');
        data = await res.json();
    } catch (err) {
        return;   // the setup panel still works with the active model default
    }

    opponentModels = data.models || [];
    select.innerHTML = '';
    opponentModels.forEach(model => {
        const opt = document.createElement('option');
        opt.value = model.model_id;
        opt.textContent = `${model.name} — ${Math.round(model.elo)} Elo (${model.kyu_rank})`;
        select.appendChild(opt);
    });

    if (data.active_model_id && opponentModels.some(m => m.model_id === data.active_model_id)) {
        select.value = data.active_model_id;
    }
    syncOpponent();
}

/** Mirror the selected model's locked board settings into the setup panel. */
function syncOpponent() {
    const select = document.getElementById('play-model');
    if (!select) return;
    const model = opponentModels.find(m => m.model_id === select.value);
    if (!model) return;

    document.getElementById('play-cfg-board').textContent =
        `${model.board_size}×${model.board_size}`;
    document.getElementById('play-cfg-komi').textContent = model.komi;
    document.getElementById('play-cfg-ruleset').textContent =
        model.ruleset.charAt(0).toUpperCase() + model.ruleset.slice(1);
    document.getElementById('play-model-meta').textContent =
        `Iteration ${model.iteration} · trained at ${model.default_simulations} simulations per move`;

    // Start from what the model itself trains at, clamped to the slider range.
    const slider = document.getElementById('simulations');
    const suggested = Math.min(Math.max(model.default_simulations, Number(slider.min)),
                               Number(slider.max));
    slider.value = suggested;
    document.getElementById('sim-label').textContent = slider.value;
}

document.getElementById('play-model')?.addEventListener('change', syncOpponent);

window.addEventListener('play-view-change', (event) => {
    if (event.detail.view === 'human-setup') loadPlayableModels();
});

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
    const modelSelect = document.getElementById('play-model');

    const res = await fetch('/api/game/new', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            mode: gameMode,
            player_color: playerColor,
            num_simulations: numSim,
            model_id: modelSelect ? modelSelect.value : null,
            // Let the bot give up a game it has already lost, rather than
            // playing a decided endgame out move by move.
            mercy_resign: document.getElementById('play-mercy')?.checked ?? true,
        }),
    });

    const data = await res.json();
    if (data.error) {
        alert(data.error);
        return;
    }

    gameId = data.game_id;
    gameOver = false;
    gameRecorded = false;
    opponent = data;

    // Initialize board renderer
    const canvas = document.getElementById('go-board');
    board = new GoBoardRenderer(canvas, data.board_size, onBoardClick);

    // Name both sides, so the panel says who is who without a status heading.
    const botName = data.model_name || 'Bot';
    document.getElementById('play-opponent-label').textContent = `vs ${botName}`;
    document.getElementById('play-name-black').textContent =
        playerColor === 'black' ? 'You' : botName;
    document.getElementById('play-name-white').textContent =
        playerColor === 'white' ? 'You' : botName;
    document.getElementById('play-meta-config').textContent =
        `${data.board_size}×${data.board_size} · komi ${data.komi}`;
    document.getElementById('play-summary').hidden = true;

    syncControls();

    // Switch panels
    PlayViews.show('human-game');
    // Let the launcher offer to resume this game if the user navigates away.
    PlayViews.setHumanGame({
        label: `Your game vs ${botName}`,
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

/**
 * Show only the controls that can actually do something right now.
 *
 * Pass/Resign/Undo/Suggest are meaningless once the game is over, so they go
 * away rather than sitting there disabled; what to do next lives in the
 * result banner instead.
 */
function syncControls() {
    const easy = gameMode === 'easy';
    setVisible(document.getElementById('btn-pass'), !gameOver);
    setVisible(document.getElementById('btn-resign'), !gameOver);
    setVisible(document.getElementById('btn-undo'), easy && !gameOver);
    setVisible(document.getElementById('btn-suggest'), easy && !gameOver);
    setVisible(document.getElementById('winrate-toggle-row'), easy);
}

function setVisible(el, visible) {
    if (el) el.hidden = !visible;
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
            // Illegal move (or not your turn) — a toast, since the panel no
            // longer carries a status line to flash it in.
            showToast(data.error);
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
    setThinking(true);
    try {
        await requestBotMove();
    } finally {
        setThinking(false);
    }
}

/** Mark the bot's own row as thinking while it searches. */
function setThinking(on) {
    const botRow = document.getElementById(
        playerColor === 'black' ? 'play-row-white' : 'play-row-black');
    if (botRow) botRow.classList.toggle('is-thinking', on && !gameOver);
}

async function requestBotMove() {
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
        if (data.error) {
            showToast(data.error);
            return;
        }
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
let gameRecorded = false;

async function autoRecordGame() {
    if (!gameId || gameRecorded) return;
    gameRecorded = true;

    try {
        const name = defaultRecordName();
        const res = await fetch('/api/game/record', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ game_id: gameId, name }),
        });
        const data = await res.json();
        if (!res.ok) {
            console.warn('Auto-record failed:', data.error);
            return;
        }
        showToast('Game recorded — find it in Review Games');
    } catch (err) {
        console.warn('Auto-record error:', err);
    }
}

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
    gameOver = false;
    gameRecorded = false;
    document.getElementById('play-summary').hidden = true;
    PlayViews.setHumanGame(null);
    PlayViews.show('human-setup');
    syncWinRatePanel();
});

// ---- Helpers ----
function updateBoard(state) {
    board.updateState(state);
    document.getElementById('move-counter').textContent = `Move ${state.move_number}`;
    document.getElementById('black-captures').textContent = `Captured: ${state.prisoners['1'] || 0}`;
    document.getElementById('white-captures').textContent = `Captured: ${state.prisoners['2'] || 0}`;

    // Whose turn it is: the highlighted row, not a heading above it.
    const blackToMove = !state.is_over && state.current_player === 1;
    const whiteToMove = !state.is_over && state.current_player === 2;
    document.getElementById('play-row-black').classList.toggle('to-move', blackToMove);
    document.getElementById('play-row-white').classList.toggle('to-move', whiteToMove);

    if (state.is_over) {
        gameOver = true;
        setThinking(false);
        syncControls();
        if (window.PlayViews && gameId) {
            const label = `Your game vs ${(opponent && opponent.model_name) || 'the bot'}`;
            PlayViews.setHumanGame({ label, over: true });
        }
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
        // The bot resigns on its own once the mercy rule fires, so say which
        // side gave up rather than a bare "by resignation".
        details = result.resigned_by === 'bot' ? 'The bot resigned' : 'By resignation';
    } else {
        title = `${result.winner === playerColor ? 'You Win!' : 'Bot Wins!'}`;
        details = `By ${result.margin?.toFixed(1) || '?'} points`;
    }

    // A resignation ends the game without a score — leave those blank rather
    // than printing question marks.
    const black = result.black_score?.toFixed(1) || data.scores?.black?.toFixed(1) || null;
    const white = result.white_score?.toFixed(1) || data.scores?.white?.toFixed(1) || null;

    document.getElementById('result-title').textContent = title;
    document.getElementById('result-details').textContent = details;
    document.getElementById('final-black').textContent = black || '—';
    document.getElementById('final-white').textContent = white || '—';

    // The panel keeps the result after the modal is closed — that is where the
    // "New Game" button lives, rather than loose at the bottom of the panel.
    gameOver = true;
    syncControls();
    const banner = document.getElementById('play-summary');
    banner.hidden = false;
    if (!result.winner || result.winner === 'draw') {
        banner.dataset.outcome = 'draw';
    } else {
        banner.dataset.outcome = result.winner === playerColor ? 'win' : 'loss';
    }
    document.getElementById('play-summary-badge').textContent = 'Game over';
    document.getElementById('play-summary-title').textContent = title;
    const scoreLine = (black && white) ? `⚫ ${black} — ${white} ⚪` : '';
    document.getElementById('play-summary-score').textContent =
        [details, scoreLine].filter(Boolean).join(' · ');

    modal.style.display = 'flex';
    autoRecordGame();
}

window.addEventListener('resize', () => {
    if (board) {
        board.resize();
    }
});
