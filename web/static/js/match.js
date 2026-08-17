/**
 * match.js — Bot vs Bot mode on the Play page.
 *
 * Sets up a match between two bots, then renders it live in the same layout a
 * human game uses (board on the left, info panel on the right).
 *
 * Everything lives in an IIFE: game.js owns the human-game half of this page
 * and declares its own globals (`board`, `gameId`, ...), so nothing here may
 * leak into the global scope.
 *
 * The server drives the match; this file only polls /api/match/<id> for a
 * snapshot and paints it. That means a reload never desynchronises the match,
 * and the same snapshot shape will serve an online (OGS) game unchanged.
 */
(function () {
    'use strict';

    const POLL_INTERVAL_MS = 500;

    let opponents = null;         // /api/match/opponents payload
    let matchBoard = null;        // GoBoardRenderer for the spectator board
    let matchId = null;
    let pollTimer = null;
    let matchChart = null;
    let showEstimate = false;
    let showWinRate = true;
    let lastStatus = null;
    // Last snapshot painted — a rematch is rebuilt from it, so a match picked
    // up from the launcher can be replayed without the original request.
    let lastSnapshot = null;

    function el(id) {
        return document.getElementById(id);
    }

    // ---- View integration ------------------------------------------------
    //
    // play_views.js owns which panel is on screen. This half only reacts:
    // load the opponent list when the setup view opens, and resize the board
    // when the live view opens (a canvas sized while hidden measures zero).

    window.addEventListener('play-view-change', (event) => {
        const view = event.detail.view;
        if (view === 'match-setup') {
            loadOpponents();
        } else if (view === 'match-live' && matchBoard) {
            matchBoard.resize();
        }
    });

    /** Open a match from the launcher's in-progress list. */
    async function watchMatch(id) {
        try {
            const res = await fetch(`/api/match/${id}`);
            if (!res.ok) {
                showMatchError('That match is no longer available.');
                return;
            }
            enterMatch(await res.json());
        } catch (err) {
            showMatchError('Could not open that match.');
        }
    }

    window.MatchView = { watch: watchMatch };

    // ---- Opponent picker -------------------------------------------------

    async function loadOpponents() {
        if (opponents) return;
        try {
            const res = await fetch('/api/match/opponents');
            opponents = await res.json();
        } catch (err) {
            showMatchError('Could not load the list of bots.');
            return;
        }
        populateSelect('match-player-a', 0);
        populateSelect('match-player-b', 1);
        syncSides();
    }

    /** Options: every model, the random bot, and any future (disabled) types. */
    function populateSelect(selectId, defaultIndex) {
        const select = el(selectId);
        if (!select) return;
        select.innerHTML = '';

        const models = opponents.models || [];
        if (models.length) {
            const group = document.createElement('optgroup');
            group.label = 'Trained models';
            models.forEach(m => {
                const opt = document.createElement('option');
                opt.value = JSON.stringify({ type: 'model', model_id: m.model_id });
                opt.textContent = `${m.name} — ${Math.round(m.elo)} Elo (${m.kyu_rank}), ${m.board_size}×${m.board_size}`;
                opt.dataset.boardSize = m.board_size;
                opt.dataset.modelId = m.model_id;
                group.appendChild(opt);
            });
            select.appendChild(group);
        }

        const baseline = document.createElement('optgroup');
        baseline.label = 'Baselines';
        const randomOpt = document.createElement('option');
        randomOpt.value = JSON.stringify({ type: 'random' });
        randomOpt.textContent = 'Random Bot — 500 Elo anchor';
        baseline.appendChild(randomOpt);
        select.appendChild(baseline);

        // Live bots on online-go.com. Listing them needs no account; playing
        // one does, which syncSides() explains rather than failing at start.
        const ogs = opponents.ogs || {};
        if (ogs.available && (ogs.bots || []).length) {
            const online = document.createElement('optgroup');
            online.label = 'Online — OGS bots';
            ogs.bots.forEach(bot => {
                const opt = document.createElement('option');
                opt.value = JSON.stringify({ type: 'ogs', bot_id: bot.bot_id });
                opt.textContent = `${bot.name} — ${bot.rank} on OGS`;
                opt.dataset.ogsId = bot.bot_id;
                online.appendChild(opt);
            });
            select.appendChild(online);
        }

        (opponents.player_types || [])
            .filter(t => !t.available)
            .forEach(t => {
                const opt = document.createElement('option');
                opt.value = JSON.stringify({ type: t.type });
                opt.textContent = `${t.label} — ${t.note}`;
                opt.disabled = true;
                baseline.appendChild(opt);
            });

        // Default: the active model on one side, and the next model (or the
        // random bot when there is only one) on the other.
        const optionCount = select.options.length;
        let index = Math.min(defaultIndex, optionCount - 1);
        if (defaultIndex === 0 && opponents.active_model_id) {
            const found = Array.from(select.options)
                .findIndex(o => o.dataset.modelId === opponents.active_model_id);
            if (found >= 0) index = found;
        }
        select.selectedIndex = Math.max(0, index);
    }

    function selectedSpec(selectId) {
        const select = el(selectId);
        if (!select || !select.value) return null;
        try {
            const spec = JSON.parse(select.value);
            return spec;
        } catch (err) {
            return null;
        }
    }

    function modelFor(spec) {
        if (!spec || spec.type !== 'model') return null;
        return (opponents.models || []).find(m => m.model_id === spec.model_id) || null;
    }

    function ogsBotFor(spec) {
        if (!spec || spec.type !== 'ogs') return null;
        const bots = (opponents.ogs || {}).bots || [];
        return bots.find(b => b.bot_id === spec.bot_id) || null;
    }

    /** Describe each side under its picker, and warn about invalid pairings. */
    function syncSides() {
        const specA = selectedSpec('match-player-a');
        const specB = selectedSpec('match-player-b');
        const modelA = modelFor(specA);
        const modelB = modelFor(specB);
        const ogsA = ogsBotFor(specA);
        const ogsB = ogsBotFor(specB);

        el('match-player-a-meta').textContent = sideDescription(specA, modelA, ogsA);
        el('match-player-b-meta').textContent = sideDescription(specB, modelB, ogsB);

        // The board a match is played on comes from whichever side is a model.
        const boardSize = (modelA || modelB || {}).board_size;
        const ogs = opponents.ogs || {};

        let warning = '';
        let invalid = false;

        if (modelA && modelB && modelA.board_size !== modelB.board_size) {
            warning = `${modelA.name} plays ${modelA.board_size}×${modelA.board_size} and ` +
                      `${modelB.name} plays ${modelB.board_size}×${modelB.board_size} — ` +
                      'they cannot play each other.';
            invalid = true;
        } else if ((ogsA || ogsB) && !modelA && !modelB) {
            warning = 'An OGS bot needs one of your models on the other side.';
            invalid = true;
        } else if (ogsA && ogsB) {
            warning = 'Two OGS bots would play each other on OGS, not here.';
            invalid = true;
        } else if ((ogsA || ogsB) && !ogs.credentials_ok) {
            warning = ogs.credentials_error ||
                      'Playing an OGS bot needs OGS credentials on this machine.';
            invalid = true;
        } else if ((ogsA || ogsB) && boardSize) {
            const bot = ogsA || ogsB;
            if (bot.board_sizes && !bot.board_sizes.includes(boardSize)) {
                warning = `${bot.name} does not play ${boardSize}×${boardSize} — ` +
                          `it plays ${bot.board_sizes.join(', ')}.`;
                invalid = true;
            } else if (!bot.accepting) {
                warning = `${bot.name} is not accepting new challenges right now.`;
                invalid = true;
            } else {
                warning = `Games against ${bot.name} are played on online-go.com, ` +
                          'unranked, on their clock. Only your model\'s Elo moves.';
            }
        } else if (modelA && modelB && modelA.model_id === modelB.model_id) {
            warning = 'A model playing itself: the games are recorded and shown, ' +
                      'but no Elo changes — a mirror match says nothing about strength.';
        } else if (specA && specB && specA.type === 'random' && specB.type === 'random') {
            warning = 'Two random bots. Watchable, but nothing is rated — the ' +
                      'random bot is the fixed Elo anchor.';
        }

        const box = el('match-setup-warning');
        box.textContent = warning;
        box.style.display = warning ? '' : 'none';

        el('match-start').disabled = invalid;
        syncOGSLimits(!!(ogsA || ogsB));
    }

    /**
     * An OGS series is capped and runs on OGS's clock, so the two controls that
     * do not apply are taken out of the way rather than silently ignored.
     */
    function syncOGSLimits(isOGS) {
        const games = el('match-games');
        const delayGroup = el('match-delay')?.closest('.form-group');
        if (!games) return;

        const cap = isOGS ? 5 : 20;
        games.max = cap;
        if (Number(games.value) > cap) {
            games.value = cap;
            el('match-games-label').textContent = games.value;
        }
        if (delayGroup) delayGroup.hidden = isOGS;
    }

    function sideDescription(spec, model, ogsBot) {
        if (!spec) return '';
        if (spec.type === 'random') return 'Uniform random legal moves · fixed 500 Elo';
        if (spec.type === 'ogs') {
            if (!ogsBot) return '';
            const sizes = ogsBot.board_sizes ? ogsBot.board_sizes.join(', ') : 'any size';
            return `Live on OGS · ${ogsBot.rank} (≈${Math.round(ogsBot.elo)} Elo here) · ` +
                   `plays ${sizes}` +
                   (ogsBot.settings_published ? '' : ' · publishes no settings');
        }
        if (spec.type !== 'model') return '';
        if (!model) return '';
        return `${model.board_size}×${model.board_size} · komi ${model.komi} · ` +
               `${model.ruleset} · iteration ${model.iteration} · ${model.default_simulations} sims`;
    }

    ['match-player-a', 'match-player-b'].forEach(id => {
        const select = el(id);
        if (select) select.addEventListener('change', syncSides);
    });

    // ---- Setup controls --------------------------------------------------

    const gamesSlider = el('match-games');
    if (gamesSlider) {
        gamesSlider.addEventListener('input', e => {
            el('match-games-label').textContent = e.target.value;
        });
    }

    function formatDelay(ms) {
        return `${(Number(ms) / 1000).toFixed(1)}s`;
    }

    const delaySlider = el('match-delay');
    if (delaySlider) {
        delaySlider.addEventListener('input', e => {
            el('match-delay-label').textContent = formatDelay(e.target.value);
        });
    }

    const liveDelaySlider = el('match-live-delay');
    if (liveDelaySlider) {
        liveDelaySlider.addEventListener('input', async e => {
            el('match-live-delay-label').textContent = formatDelay(e.target.value);
            if (!matchId) return;
            await fetch(`/api/match/${matchId}/speed`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ move_delay: Number(e.target.value) / 1000 }),
            });
        });
    }

    // ---- Starting / stopping a match ------------------------------------

    el('match-start')?.addEventListener('click', async () => {
        const specA = selectedSpec('match-player-a');
        const specB = selectedSpec('match-player-b');
        if (!specA || !specB) {
            showMatchError('Pick a bot for each side.');
            return;
        }

        const btn = el('match-start');
        btn.disabled = true;
        btn.textContent = 'Starting…';

        try {
            const res = await fetch('/api/match/new', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    player_a: specA,
                    player_b: specB,
                    num_games: parseInt(el('match-games').value, 10),
                    move_delay: Number(el('match-delay').value) / 1000,
                    record_games: el('match-record').checked,
                    update_ratings: el('match-rate').checked,
                    mercy_resign: el('match-mercy').checked,
                }),
            });
            const data = await res.json();
            if (!res.ok) {
                showMatchError(data.error || 'Could not start the match.');
                return;
            }
            enterMatch(data);
        } finally {
            btn.disabled = false;
            btn.textContent = 'Start Match';
        }
    });

    function enterMatch(snapshot) {
        matchId = snapshot.match_id;
        lastSnapshot = snapshot;

        el('match-summary').hidden = true;
        PlayViews.show('match-live');

        // Spectator board: no click handler, so no hover preview either.
        matchBoard = new GoBoardRenderer(el('match-board'), snapshot.board_size, null);
        if (matchChart) {
            matchChart.destroy();
            matchChart = null;
        }

        // Match the live delay slider to what the match actually started with.
        const ms = Math.round((snapshot.move_delay || 0) * 1000);
        el('match-live-delay').value = ms;
        el('match-live-delay-label').textContent = formatDelay(ms);

        render(snapshot);
        startPolling();
    }

    el('match-pause')?.addEventListener('click', async () => {
        if (!matchId) return;
        const paused = el('match-pause').dataset.paused === '1';
        const res = await fetch(`/api/match/${matchId}/pause`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ paused: !paused }),
        });
        if (res.ok) render(await res.json());
    });

    el('match-stop')?.addEventListener('click', async () => {
        if (!matchId) return;
        if (!confirm('Stop this match? The game in progress is discarded; finished games are kept.')) return;
        const res = await fetch(`/api/match/${matchId}/stop`, { method: 'POST' });
        if (res.ok) render(await res.json());
    });

    el('match-new')?.addEventListener('click', () => {
        stopPolling();
        matchId = null;
        opponents = null;             // model Elo moved — reload the list
        el('match-summary').hidden = true;
        PlayViews.show('match-setup');
    });

    /**
     * Run the same series again, same two bots and same settings.
     *
     * Rebuilt from the snapshot rather than from the form: the match being
     * looked at may have been started in another tab, or picked up from the
     * launcher, and the setup panel would then describe something else.
     */
    el('match-rematch')?.addEventListener('click', async () => {
        if (!lastSnapshot) return;
        const btn = el('match-rematch');
        btn.disabled = true;
        btn.textContent = 'Starting…';

        try {
            const res = await fetch('/api/match/new', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    player_a: specFromPlayer((lastSnapshot.players || {}).a),
                    player_b: specFromPlayer((lastSnapshot.players || {}).b),
                    num_games: lastSnapshot.num_games,
                    move_delay: lastSnapshot.move_delay,
                    record_games: lastSnapshot.record_games !== false,
                    update_ratings: !!lastSnapshot.rated,
                    mercy_resign: el('match-mercy')?.checked ?? true,
                }),
            });
            const data = await res.json();
            if (!res.ok) {
                showToastMessage(data.error || 'Could not start the rematch.');
                return;
            }
            stopPolling();
            opponents = null;         // ratings moved — the picker is stale
            enterMatch(data);
        } finally {
            btn.disabled = false;
            btn.textContent = 'Rematch';
        }
    });

    /** The /api/match/new spec that would recreate one side of a match. */
    function specFromPlayer(player) {
        if (!player) return null;
        if (player.kind === 'model') {
            return {
                type: 'model',
                model_id: player.model_id,
                // Same search strength, or the rematch is not the same match.
                num_simulations: player.num_simulations,
            };
        }
        if (player.kind === 'ogs') {
            // Without the bot id this rebuilds as "an OGS opponent needs a
            // bot_id" — which is what Rematch used to do after a failed match.
            return {
                type: 'ogs',
                bot_id: player.ogs_id,
                ranked: !!player.ranked,
            };
        }
        return { type: player.kind || 'random' };
    }

    /** The Play page's toast, shared with game.js. */
    function showToastMessage(text) {
        if (typeof showToast === 'function') {
            showToast(text);
            return;
        }
        showMatchError(text);
    }

    // Leaving the match view does not stop the match — it is a server-side job,
    // and the launcher lists it so it can be picked back up.
    window.addEventListener('play-view-change', (event) => {
        if (event.detail.view !== 'match-live') stopPolling();
    });

    // ---- Polling ---------------------------------------------------------

    function startPolling() {
        stopPolling();
        pollTimer = setInterval(poll, POLL_INTERVAL_MS);
    }

    function stopPolling() {
        if (pollTimer) clearInterval(pollTimer);
        pollTimer = null;
    }

    async function poll() {
        if (!matchId) return;
        let snapshot;
        try {
            const res = await fetch(`/api/match/${matchId}`);
            if (!res.ok) {
                stopPolling();
                return;
            }
            snapshot = await res.json();
        } catch (err) {
            return;   // transient — the next tick retries
        }

        render(snapshot);
        if (snapshot.status !== 'running' && snapshot.status !== 'pending') {
            stopPolling();
        }
        await refreshEstimate();
    }

    // ---- Rendering -------------------------------------------------------

    function render(snapshot) {
        if (!snapshot) return;
        lastStatus = snapshot.status;
        lastSnapshot = snapshot;

        if (snapshot.state && matchBoard) {
            matchBoard.updateState(snapshot.state);
            el('match-move-counter').textContent = `Move ${snapshot.state.move_number}`;
            const prisoners = snapshot.state.prisoners || {};
            el('match-black-captures').textContent = `Captured: ${prisoners['1'] || 0}`;
            el('match-white-captures').textContent = `Captured: ${prisoners['2'] || 0}`;
        }

        // The territory overlay writes its own (estimated) numbers into these
        // fields, so only fill in the scored ones when it is switched off.
        if (snapshot.scores && !showEstimate) {
            el('match-black-score').textContent = Number(snapshot.scores.black || 0).toFixed(1);
            el('match-white-score').textContent = Number(snapshot.scores.white || 0).toFixed(1);
        }

        const over = snapshot.status !== 'running' && snapshot.status !== 'pending';

        renderPlayers(snapshot, over);
        renderResults(snapshot);
        renderWinRate(snapshot);

        renderOGSStatus(snapshot, over);

        const pauseBtn = el('match-pause');
        pauseBtn.dataset.paused = snapshot.paused ? '1' : '0';
        pauseBtn.textContent = snapshot.paused ? '▶ Resume' : '⏸ Pause';

        // A finished series has nothing left to pause, stop or slow down —
        // those controls go away instead of sitting there disabled.
        el('match-controls').hidden = over;
        el('match-speed-group').hidden = over;
        renderSummary(snapshot, over);
    }

    /**
     * The end-of-series banner: who won, and the obvious next thing to do.
     *
     * Deliberately thin. The scoreboard right below it already carries both
     * names, both ratings and the running score, and the results list carries
     * every game — repeating all of that here was the bulk of what made the
     * old banner a wall of numbers.
     */
    function renderSummary(snapshot, over) {
        const banner = el('match-summary');
        if (!banner) return;

        banner.hidden = !over;
        if (!over) return;

        banner.dataset.outcome = snapshot.status;
        const rematch = el('match-rematch');

        if (snapshot.status === 'error') {
            el('match-summary-icon').textContent = '⚠️';
            el('match-summary-title').textContent =
                snapshot.error || 'The match could not finish.';
            el('match-summary-elo').textContent = '';
            if (rematch) rematch.textContent = 'Try again';
            return;
        }

        const players = snapshot.players || {};
        const series = snapshot.series || {};
        const nameA = (players.a || {}).name || 'A';
        const nameB = (players.b || {}).name || 'B';
        const tied = (series.a ?? 0) === (series.b ?? 0);
        const winner = (series.a ?? 0) > (series.b ?? 0) ? nameA : nameB;
        const high = Math.max(series.a ?? 0, series.b ?? 0);
        const low = Math.min(series.a ?? 0, series.b ?? 0);

        el('match-summary-icon').textContent = tied ? '🤝' : '🏆';
        el('match-summary-title').textContent = tied
            ? `${nameA} and ${nameB} draw the series ${high}–${low}`
            : `${winner} wins ${high}–${low}`;
        if (rematch) rematch.textContent = 'Rematch';

        // The one number the scoreboard cannot show afterwards: what the
        // series did to the ratings.
        el('match-summary-elo').innerHTML = eloMovement(snapshot);
    }

    /**
     * What the online opponent is doing, and how long it has been doing it.
     *
     * Only a networked player reports a status, so this stays hidden for a
     * match between two local models.
     */
    function renderOGSStatus(snapshot, over) {
        const row = el('ogs-status');
        if (!row) return;

        const players = snapshot.players || {};
        const status = (players.a || {}).status || (players.b || {}).status;
        if (!status) {
            row.hidden = true;
            return;
        }

        row.hidden = false;
        row.dataset.state = status.state;

        const seconds = Math.round(status.waiting_seconds || 0);
        const stale = status.state === 'waiting' && seconds > 20;
        let text = status.detail || status.state;

        if (status.state === 'waiting') {
            text = `${status.detail} · ${seconds}s`;
            if (stale) text += ' — it may be busy; you can stop the match';
        } else if (status.state === 'connecting') {
            text = status.detail;
        } else if (status.state === 'playing') {
            text = over ? 'Game finished on OGS' : 'Playing on OGS';
        } else if (status.state === 'finished') {
            text = 'Game finished on OGS';
        } else if (status.state === 'failed' || status.state === 'cancelled') {
            text = status.detail || status.state;
        }
        if (status.online === false) {
            text = 'Disconnected from OGS — reconnecting…';
            row.dataset.state = 'offline';
        }

        el('ogs-status-text').textContent = text;

        const link = el('ogs-status-link');
        link.hidden = !status.game_url;
        if (status.game_url) link.href = status.game_url;
    }

    /** "hot boi 812 (+7.4) · lucky thirteen 494 (−7.4)", or nothing to say. */
    function eloMovement(snapshot) {
        if (!snapshot.rated) return 'Unrated — no ratings changed.';
        const players = snapshot.players || {};
        return ['a', 'b'].map(slot => {
            const p = players[slot] || {};
            if (p.rating_is_fixed) return `${escapeHtml(p.name || slot)}: anchor`;
            const delta = Number(p.rating_delta || 0);
            const cls = delta > 0 ? 'match-elo-up' : (delta < 0 ? 'match-elo-down' : '');
            const sign = delta > 0 ? '+' : '';
            return `${escapeHtml(p.name || slot)} ${Math.round(p.rating ?? 0)} ` +
                   `<span class="${cls}">(${sign}${delta.toFixed(1)})</span>`;
        }).join(' &nbsp;·&nbsp; ');
    }

    function escapeHtml(str) {
        return String(str).replace(/[&<>"']/g, c => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
        })[c]);
    }

    function renderPlayers(snapshot, over) {
        const players = snapshot.players || {};
        ['a', 'b'].forEach(slot => {
            const player = players[slot] || {};
            el(`match-color-${slot}`).textContent = player.color === 2 ? '⚪' : '⚫';
            el(`match-name-${slot}`).textContent = player.name || '—';
            el(`match-wins-${slot}`).textContent = (snapshot.series || {})[slot] ?? 0;

            const elo = Math.round(player.rating ?? 0);
            const delta = Number(player.rating_delta || 0);
            let deltaText = '';
            if (Math.abs(delta) >= 0.05) {
                deltaText = ` <span class="${delta > 0 ? 'match-elo-up' : 'match-elo-down'}">` +
                            `${delta > 0 ? '+' : ''}${delta.toFixed(1)}</span>`;
            } else if (player.rating_is_fixed) {
                deltaText = ' <span class="match-elo-fixed">anchor</span>';
            }
            el(`match-elo-${slot}`).innerHTML = `${elo} Elo${deltaText}`;

            // Highlight whoever is to move, so the board reads at a glance.
            const toMove = snapshot.state && snapshot.state.current_player === player.color;
            el(`match-row-${slot}`).classList.toggle('to-move',
                !!toMove && snapshot.status === 'running');
        });

        const draws = (snapshot.series || {}).draw || 0;
        const rated = snapshot.rated ? '' : ' · unrated';
        const paused = (!over && snapshot.paused) ? ' · paused' : '';
        el('match-series-line').textContent =
            `Game ${Math.min(snapshot.current_game, snapshot.num_games)} of ${snapshot.num_games}` +
            (draws ? ` · ${draws} draw${draws === 1 ? '' : 's'}` : '') + rated + paused;

        // What the removed status heading used to say, minus the parts the
        // summary banner and the highlighted row already carry.
        const last = snapshot.last_move;
        el('match-last-move').textContent =
            last ? `Last: ${last.player_name} ${moveText(last.move)}` : '';
    }

    function moveText(move) {
        if (!move) return '';
        if (move[0] === -1) return 'pass';
        if (move[0] === -2) return 'resign';
        // Column letters skip I, the usual Go convention.
        const letters = 'ABCDEFGHJKLMNOPQRST';
        const size = matchBoard ? matchBoard.boardSize : 9;
        return `${letters[move[1]] || '?'}${size - move[0]}`;
    }

    function renderResults(snapshot) {
        const list = el('match-results-list');
        const results = snapshot.results || [];
        if (!results.length) {
            list.innerHTML = '<span class="match-results-empty">No games finished yet.</span>';
            return;
        }

        list.innerHTML = '';
        results.slice().reverse().forEach(result => {
            // A recorded game is a link straight to itself in Review Games;
            // one that was not recorded stays a plain row.
            const saved = (result.saved_as || [])[0];
            const row = document.createElement(saved ? 'a' : 'div');
            row.className = 'match-result-row';
            if (saved) {
                row.href = `/training/review?game=${encodeURIComponent(saved)}`;
                row.title = 'Open this game in Review Games';
                row.classList.add('is-link');
            }

            let outcome;
            if (result.winner === 1) outcome = `⚫ ${result.black_name}`;
            else if (result.winner === 2) outcome = `⚪ ${result.white_name}`;
            else outcome = 'Draw';

            const how = result.resigned_by
                ? 'by resignation'
                : (result.winner ? `+${Number(result.margin).toFixed(1)}` : '');

            row.innerHTML = `
                <span class="match-result-index">#${result.index + 1}</span>
                <span class="match-result-winner">${escapeHtml(outcome)}</span>
                <span class="match-result-margin">${escapeHtml(how)}</span>
                <span class="match-result-moves">${result.num_moves} moves</span>
            `;
            list.appendChild(row);
        });
    }

    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str == null ? '' : String(str);
        return div.innerHTML;
    }

    // ---- Territory overlay ----------------------------------------------

    el('match-toggle-estimate')?.addEventListener('change', async e => {
        showEstimate = e.target.checked;
        if (matchBoard) matchBoard.showEstimate = showEstimate;
        if (showEstimate) {
            await refreshEstimate();
        } else if (matchBoard) {
            matchBoard.setOwnershipMap(null);
        }
    });

    async function refreshEstimate() {
        if (!showEstimate || !matchId || !matchBoard) return;
        try {
            const res = await fetch(`/api/match/${matchId}/estimate`, { method: 'POST' });
            if (!res.ok) return;
            const data = await res.json();
            matchBoard.setOwnershipMap(data.ownership_map);
            el('match-black-score').textContent = (data.black_estimate ?? 0).toFixed(1);
            el('match-white-score').textContent = (data.white_estimate ?? 0).toFixed(1);
        } catch (err) {
            /* transient */
        }
    }

    // ---- Win-rate curve --------------------------------------------------

    el('match-toggle-winrate')?.addEventListener('change', e => {
        showWinRate = e.target.checked;
        el('match-winrate-panel').style.display = showWinRate ? '' : 'none';
    });

    /**
     * Black's win probability per position, as evaluated by one of the two
     * networks (the server picks one for the whole series, so the curve stays
     * on a single scale instead of alternating between two opinions).
     */
    function renderWinRate(snapshot) {
        if (!showWinRate) return;
        const canvas = el('match-winrate-chart');
        const panel = el('match-winrate-panel');
        const series = snapshot.win_rates || [];

        if (!canvas || typeof Chart === 'undefined' || !series.length) {
            // Nothing to chart (e.g. two random bots — no network to ask).
            if (panel) panel.style.display = series.length ? '' : 'none';
            return;
        }
        panel.style.display = '';

        const current = series[series.length - 1];
        el('match-winrate-current').textContent =
            `⚫ ${Number(current).toFixed(1)}% / ⚪ ${(100 - Number(current)).toFixed(1)}%`;

        if (matchChart) {
            matchChart.data.labels = series.map((_, i) => i);
            matchChart.data.datasets[0].data = series;
            matchChart.update('none');
            return;
        }

        matchChart = new Chart(canvas.getContext('2d'), {
            type: 'line',
            data: {
                labels: series.map((_, i) => i),
                datasets: [{
                    label: 'Black Win %',
                    data: series,
                    borderColor: '#c8956c',
                    backgroundColor: 'rgba(200, 149, 108, 0.12)',
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
                            title: items => `Move ${items[0].label}`,
                            label: item => `Black: ${Number(item.raw).toFixed(1)}%`,
                        },
                    },
                },
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.06)' }, ticks: { color: '#9a9a9a', maxTicksLimit: 8 } },
                    y: {
                        min: 0,
                        max: 100,
                        grid: { color: 'rgba(255,255,255,0.06)' },
                        ticks: { color: '#9a9a9a', stepSize: 25, callback: v => `${v}%` },
                    },
                },
            },
        });
    }

    function showMatchError(message) {
        const box = el('match-setup-warning');
        if (!box) return;
        box.textContent = message;
        box.style.display = '';
    }

    window.addEventListener('resize', () => {
        if (matchBoard) matchBoard.resize();
    });

    // A match keeps running server-side when the tab is hidden; stop polling
    // while it is, and catch up on return.
    document.addEventListener('visibilitychange', () => {
        if (!matchId) return;
        if (document.hidden) {
            stopPolling();
        } else if (lastStatus === 'running' || lastStatus === 'pending') {
            poll();
            startPolling();
        }
    });
})();
