/**
 * play_views.js — View router for the Play page.
 *
 * The Play page is a launcher, not a pair of tabs. Tabs implied the two modes
 * were alternative views of the same thing, but a bot-vs-bot match is a
 * server-side job that keeps running whether or not anyone is watching it —
 * so it belongs in an "in progress" list you can leave and come back to,
 * not behind a tab that silently hides it.
 *
 * Views:
 *   launcher     — pick what to start; lists anything already running
 *   human-setup  — configure a game against the bot
 *   human-game   — the human game board
 *   match-setup  — configure a bot-vs-bot series
 *   match-live   — watch a series (running or finished)
 *
 * game.js and match.js drive their own halves; both route through show() so
 * exactly one view is visible and nothing has to guess what the other is doing.
 */

(function () {
    // Each view is one wrapper: a "← Back" header followed by the view's own
    // content. Everything inside is laid out by CSS, so switching views only
    // ever shows or hides these five elements.
    const VIEW_PANELS = {
        'launcher': ['play-launcher'],
        'human-setup': ['human-setup-view'],
        'human-game': ['game-area'],
        'match-setup': ['match-setup-view'],
        'match-live': ['match-area'],
    };

    const LIST_POLL_MS = 2000;

    let currentView = 'launcher';
    let humanGame = null;      // {label, over} while a human game exists
    let listTimer = null;

    function el(id) {
        return document.getElementById(id);
    }

    function show(view) {
        if (!VIEW_PANELS[view]) return;
        currentView = view;

        Object.entries(VIEW_PANELS).forEach(([name, ids]) => {
            ids.forEach(id => {
                const node = el(id);
                if (!node) return;
                node.style.display = (name === view) ? '' : 'none';
            });
        });

        // The in-progress list only needs polling while it is on screen.
        if (view === 'launcher') {
            refreshInProgress();
            startListPolling();
        } else {
            stopListPolling();
        }

        window.dispatchEvent(new CustomEvent('play-view-change', { detail: { view } }));
    }

    function getView() {
        return currentView;
    }

    /**
     * Called by game.js so the launcher can offer to resume a human game
     * instead of stranding it behind a view the user navigated away from.
     */
    function setHumanGame(info) {
        humanGame = info || null;
        if (currentView === 'launcher') refreshInProgress();
    }

    // ---- In-progress list ------------------------------------------------

    function startListPolling() {
        stopListPolling();
        listTimer = setInterval(refreshInProgress, LIST_POLL_MS);
    }

    function stopListPolling() {
        if (listTimer) clearInterval(listTimer);
        listTimer = null;
    }

    async function refreshInProgress() {
        let matches = [];
        try {
            const res = await fetch('/api/match/list');
            if (res.ok) matches = await res.json();
        } catch (err) {
            // Offline or server restarting — just show the human game, if any.
        }

        const section = el('play-inprogress');
        const list = el('play-inprogress-list');
        if (!section || !list) return;

        const rows = [];

        if (humanGame) {
            rows.push(humanRow(humanGame));
        }
        matches.forEach(m => rows.push(matchRow(m)));

        if (!rows.length) {
            section.hidden = true;
            list.innerHTML = '';
            return;
        }

        section.hidden = false;
        list.innerHTML = '';
        rows.forEach(row => list.appendChild(row));
    }

    function humanRow(info) {
        const row = document.createElement('div');
        row.className = 'play-progress-row';
        row.innerHTML = `
            <span class="play-progress-dot ${info.over ? 'is-done' : 'is-live'}"></span>
            <span class="play-progress-name">👤 ${escapeHtml(info.label || 'Your game')}</span>
            <span class="play-progress-meta">${info.over ? 'finished' : 'your move'}</span>
        `;
        const btn = document.createElement('button');
        btn.className = 'btn-secondary play-progress-btn';
        btn.textContent = info.over ? 'View' : 'Resume';
        btn.addEventListener('click', () => show('human-game'));
        row.appendChild(btn);
        return row;
    }

    function matchRow(m) {
        const live = m.status === 'running' || m.status === 'pending';
        const series = m.series || {};
        const players = m.players || {};
        const nameA = (players.a || {}).name || 'A';
        const nameB = (players.b || {}).name || 'B';

        const row = document.createElement('div');
        row.className = 'play-progress-row';

        let meta;
        if (live) {
            meta = `game ${Math.min(m.games_completed + 1, m.num_games)}/${m.num_games} · ` +
                   `${series.a ?? 0}–${series.b ?? 0}`;
        } else if (m.status === 'error') {
            meta = 'error';
        } else {
            meta = `${m.status} · ${series.a ?? 0}–${series.b ?? 0}`;
        }

        row.innerHTML = `
            <span class="play-progress-dot ${live ? 'is-live' : 'is-done'}"></span>
            <span class="play-progress-name">🤖 ${escapeHtml(nameA)} vs ${escapeHtml(nameB)}</span>
            <span class="play-progress-meta">${escapeHtml(meta)}</span>
        `;

        const btn = document.createElement('button');
        btn.className = 'btn-secondary play-progress-btn';
        btn.textContent = live ? 'Watch' : 'Results';
        btn.addEventListener('click', () => {
            if (window.MatchView && window.MatchView.watch) {
                window.MatchView.watch(m.match_id);
            }
        });
        row.appendChild(btn);

        // Finished matches can be cleared out so the list stays useful.
        if (!live) {
            const dismiss = document.createElement('button');
            dismiss.className = 'btn-icon play-progress-dismiss';
            dismiss.title = 'Remove from this list';
            dismiss.textContent = '✕';
            dismiss.addEventListener('click', async () => {
                try {
                    await fetch(`/api/match/${m.match_id}`, { method: 'DELETE' });
                } catch (err) { /* refresh will show whether it worked */ }
                refreshInProgress();
            });
            row.appendChild(dismiss);
        }

        return row;
    }

    function escapeHtml(str) {
        return String(str).replace(/[&<>"']/g, c => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
        })[c]);
    }

    // ---- Wiring ----------------------------------------------------------

    document.addEventListener('DOMContentLoaded', () => {
        if (!el('play-launcher')) return;   // not the play page

        document.querySelectorAll('[data-launch]').forEach(card => {
            card.addEventListener('click', () => {
                show(card.dataset.launch === 'match' ? 'match-setup' : 'human-setup');
            });
        });

        document.querySelectorAll('[data-play-back]').forEach(btn => {
            btn.addEventListener('click', () => show('launcher'));
        });

        show('launcher');
    });

    window.PlayViews = { show, getView, setHumanGame, refreshInProgress };
})();
