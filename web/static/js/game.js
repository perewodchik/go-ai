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
    document.getElementById('setup-panel').style.display = 'none';
    document.getElementById('game-area').style.display = 'grid';
    document.getElementById('btn-new-game').style.display = 'none';

    updateBoard(data.state);
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
        await refreshEstimateIfEnabled();

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
    await refreshEstimateIfEnabled();
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
        await refreshEstimateIfEnabled();

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
    await refreshEstimateIfEnabled();
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

document.getElementById('btn-new-game')?.addEventListener('click', () => {
    document.getElementById('game-area').style.display = 'none';
    document.getElementById('setup-panel').style.display = '';
    gameId = null;
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
