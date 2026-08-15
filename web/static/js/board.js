/**
 * board.js — Canvas-based Go board renderer.
 *
 * Draws a Go board with:
 * - Wood-texture background gradient
 * - Anti-aliased stones with 3D gradient effect
 * - Star points (hoshi) at standard positions
 * - Coordinate labels (A-T, 1-19)
 * - Last-move indicator (small dot)
 * - Hover preview (semi-transparent stone)
 * - Territory estimation overlay (semi-transparent colors)
 * - Click-to-play with snap-to-intersection
 */

class GoBoardRenderer {
    /**
     * @param {HTMLCanvasElement} canvas - The canvas element to draw on.
     * @param {number} boardSize - Board dimension (7, 9, 13, 15, or 19).
     * @param {function} onClickCallback - Called with (row, col) when user clicks.
     * @param {object} options - Configuration options (e.g. { enableHover: false }).
     */
    constructor(canvas, boardSize = 9, onClickCallback = null, options = {}) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.boardSize = boardSize;
        this.onClick = onClickCallback;
        this.enableHover = options.enableHover !== undefined ? options.enableHover : (onClickCallback !== null);

        // Layout constants
        this.padding = 48;          // Space for labels (kept clear of edge stones)
        this.cellSize = 0;          // Computed in resize
        this.stoneRadius = 0;

        // State
        this.grid = [];             // 2D array: 0=empty, 1=black, 2=white
        this.lastMove = null;       // {row, col} of last move
        this.hoverPos = null;       // {row, col} under cursor
        this.currentPlayer = 1;     // For hover preview color
        this.ownershipMap = null;   // Territory estimation overlay
        this.showEstimate = false;
        this.suggestedMove = null;  // {row, col} for move suggestion

        this._initGrid();
        this.resize();
        this._bindEvents();
        this.draw();
    }

    _initGrid() {
        this.grid = Array.from({ length: this.boardSize }, () =>
            Array(this.boardSize).fill(0)
        );
    }

    resize(forceSize = null) {
        let totalSize;
        if (forceSize) {
            totalSize = forceSize;
        } else if (this.canvas.parentElement) {
            totalSize = this.canvas.parentElement.clientWidth;
        } else {
            totalSize = Math.min(600, window.innerWidth - 20);
        }

        // Match the CSS clamp (#go-board { max-width: 600px }) so the logical
        // coordinate space stays identical to the on-screen size — otherwise
        // hover/click positions map to the wrong intersection.
        totalSize = Math.min(totalSize, 600);

        // A hidden panel measures zero, which would make the cell size (and the
        // stone radius) negative and throw out of the next draw. Keep the last
        // good geometry; the view is redrawn when it comes back on screen.
        if (totalSize < 2 * this.padding + this.boardSize) return;

        // Handle high-DPI displays
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = totalSize * dpr;
        this.canvas.height = totalSize * dpr;

        // Pin the displayed (CSS) size to the logical size as well.
        this.canvas.style.width = totalSize + 'px';
        this.canvas.style.height = totalSize + 'px';

        // Scale context to match CSS pixels
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        this.cellSize = (totalSize - 2 * this.padding) / (this.boardSize - 1);
        this.stoneRadius = this.cellSize * 0.43;
        
        // Store logical size for draw calculations
        this.logicalWidth = totalSize;
        this.logicalHeight = totalSize;
        
        // Redraw if grid is initialized
        if (this.grid.length > 0) {
            this.draw();
        }
    }

    _bindEvents() {
        this.canvas.addEventListener('mousemove', (e) => {
            if (!this.enableHover) return;
            const pos = this._canvasToBoard(e.clientX, e.clientY);
            if (pos && (this.hoverPos?.row !== pos.row || this.hoverPos?.col !== pos.col)) {
                this.hoverPos = pos;
                this.draw();
            }
        });

        this.canvas.addEventListener('mouseleave', () => {
            if (!this.enableHover) return;
            if (this.hoverPos !== null) {
                this.hoverPos = null;
                this.draw();
            }
        });

        this.canvas.addEventListener('click', (e) => {
            if (!this.onClick) return;
            const pos = this._canvasToBoard(e.clientX, e.clientY);
            if (pos) {
                this.onClick(pos.row, pos.col);
            }
        });
    }

    /**
     * Convert viewport (clientX, clientY) coords to board (row, col).
     * Uses the canvas's rendered rectangle and scales into logical space so
     * the mapping is correct regardless of CSS scaling or device pixel ratio.
     */
    _canvasToBoard(clientX, clientY) {
        const rect = this.canvas.getBoundingClientRect();
        if (!rect.width || !rect.height) return null;
        const scaleX = (this.logicalWidth || rect.width) / rect.width;
        const scaleY = (this.logicalHeight || rect.height) / rect.height;
        const x = (clientX - rect.left) * scaleX;
        const y = (clientY - rect.top) * scaleY;
        const col = Math.round((x - this.padding) / this.cellSize);
        const row = Math.round((y - this.padding) / this.cellSize);
        if (row >= 0 && row < this.boardSize && col >= 0 && col < this.boardSize) {
            return { row, col };
        }
        return null;
    }

    /** Convert board (row, col) to canvas pixel coords. */
    _boardToCanvas(row, col) {
        return {
            x: this.padding + col * this.cellSize,
            y: this.padding + row * this.cellSize,
        };
    }

    /** Update the board state from a game state dict. */
    updateState(state) {
        this.boardSize = state.board_size;
        this.grid = state.grid;
        this.currentPlayer = state.current_player;

        // Find last move
        if (state.move_history && state.move_history.length > 0) {
            const last = state.move_history[state.move_history.length - 1];
            if (last.move[0] >= 0 && last.move[1] >= 0) {
                this.lastMove = { row: last.move[0], col: last.move[1] };
            } else {
                this.lastMove = null;
            }
        } else {
            this.lastMove = null;
        }

        this.resize();
    }

    /** Set territory estimation overlay data. */
    setOwnershipMap(map) {
        this.ownershipMap = map;
        this.draw();
    }

    /** Compute territory and area scoring dynamically from grid state. */
    computeTerritory(komi = 6.5) {
        const s = this.boardSize;
        const ownershipMap = Array.from({ length: s }, () => Array(s).fill(0));
        const visited = Array.from({ length: s }, () => Array(s).fill(false));
        let blackStones = 0;
        let whiteStones = 0;
        let blackTerritory = 0;
        let whiteTerritory = 0;

        for (let r = 0; r < s; r++) {
            for (let c = 0; c < s; c++) {
                const val = this.grid[r]?.[c] || 0;
                if (val === 1) blackStones++;
                else if (val === 2) whiteStones++;
            }
        }

        for (let r = 0; r < s; r++) {
            for (let c = 0; c < s; c++) {
                if ((this.grid[r]?.[c] || 0) === 0 && !visited[r][c]) {
                    const region = [];
                    const borderColors = new Set();
                    const queue = [[r, c]];
                    visited[r][c] = true;
                    region.push([r, c]);

                    while (queue.length > 0) {
                        const [cr, cc] = queue.pop();
                        const neighbors = [
                            [cr - 1, cc], [cr + 1, cc],
                            [cr, cc - 1], [cr, cc + 1]
                        ];
                        for (const [nr, nc] of neighbors) {
                            if (nr >= 0 && nr < s && nc >= 0 && nc < s) {
                                const nVal = this.grid[nr]?.[nc] || 0;
                                if (nVal === 0) {
                                    if (!visited[nr][nc]) {
                                        visited[nr][nc] = true;
                                        region.push([nr, nc]);
                                        queue.push([nr, nc]);
                                    }
                                } else {
                                    borderColors.add(nVal);
                                }
                            }
                        }
                    }

                    if (borderColors.size === 1) {
                        const owner = borderColors.has(1) ? -1.0 : 1.0;
                        if (owner === -1.0) blackTerritory += region.length;
                        else whiteTerritory += region.length;

                        for (const [pr, pc] of region) {
                            ownershipMap[pr][pc] = owner;
                        }
                    }
                }
            }
        }

        this.ownershipMap = ownershipMap;
        const blackScore = blackStones + blackTerritory;
        const whiteScore = whiteStones + whiteTerritory + komi;
        const diff = blackScore - whiteScore;

        return {
            blackStones,
            whiteStones,
            blackTerritory,
            whiteTerritory,
            blackScore,
            whiteScore,
            komi,
            lead: diff > 0 ? `B+${diff.toFixed(1)}` : (diff < 0 ? `W+${Math.abs(diff).toFixed(1)}` : 'Tie')
        };
    }

    /** Main draw function. */
    draw() {
        const ctx = this.ctx;
        const w = this.logicalWidth || this.canvas.width;
        const h = this.logicalHeight || this.canvas.height;

        // Clear
        ctx.clearRect(0, 0, w, h);

        // Board background (wood gradient)
        const grad = ctx.createLinearGradient(0, 0, w, h);
        grad.addColorStop(0, '#dcb35c');
        grad.addColorStop(0.5, '#d4a84a');
        grad.addColorStop(1, '#c49a3c');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.roundRect(0, 0, w, h, 12);
        ctx.fill();

        // Wood grain effect (subtle lines)
        ctx.strokeStyle = 'rgba(139, 98, 68, 0.08)';
        ctx.lineWidth = 1;
        for (let i = 0; i < h; i += 7) {
            ctx.beginPath();
            ctx.moveTo(0, i + Math.sin(i * 0.05) * 3);
            ctx.lineTo(w, i + Math.sin(i * 0.05 + 2) * 3);
            ctx.stroke();
        }

        // Grid lines
        ctx.strokeStyle = '#3a2a0a';
        ctx.lineWidth = 1;
        for (let i = 0; i < this.boardSize; i++) {
            const p = this.padding + i * this.cellSize;
            // Vertical
            ctx.beginPath();
            ctx.moveTo(p, this.padding);
            ctx.lineTo(p, this.padding + (this.boardSize - 1) * this.cellSize);
            ctx.stroke();
            // Horizontal
            ctx.beginPath();
            ctx.moveTo(this.padding, p);
            ctx.lineTo(this.padding + (this.boardSize - 1) * this.cellSize, p);
            ctx.stroke();
        }

        // Star points (hoshi)
        this._drawStarPoints(ctx);

        // Coordinate labels
        this._drawLabels(ctx);

        // Territory estimation overlay
        if (this.showEstimate && this.ownershipMap) {
            this._drawOwnership(ctx);
        }

        // Stones
        for (let r = 0; r < this.boardSize; r++) {
            for (let c = 0; c < this.boardSize; c++) {
                if (this.grid[r] && this.grid[r][c] !== 0) {
                    this._drawStone(ctx, r, c, this.grid[r][c]);
                }
            }
        }

        // Last move indicator
        if (this.lastMove) {
            this._drawLastMoveMarker(ctx, this.lastMove.row, this.lastMove.col);
        }

        // Suggested move
        if (this.suggestedMove) {
            this._drawSuggestion(ctx, this.suggestedMove.row, this.suggestedMove.col);
        }

        // Hover preview
        if (this.enableHover && this.hoverPos && this.grid[this.hoverPos.row]?.[this.hoverPos.col] === 0) {
            this._drawHoverPreview(ctx, this.hoverPos.row, this.hoverPos.col);
        }
    }

    _drawStone(ctx, row, col, color) {
        const { x, y } = this._boardToCanvas(row, col);
        const r = this.stoneRadius;

        // Shadow
        ctx.beginPath();
        ctx.arc(x + 1.5, y + 1.5, r, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0,0,0,0.3)';
        ctx.fill();

        // Stone body
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        if (color === 1) {
            // Black stone gradient
            const g = ctx.createRadialGradient(x - r * 0.3, y - r * 0.3, r * 0.1, x, y, r);
            g.addColorStop(0, '#555');
            g.addColorStop(0.5, '#222');
            g.addColorStop(1, '#111');
            ctx.fillStyle = g;
        } else {
            // White stone gradient
            const g = ctx.createRadialGradient(x - r * 0.3, y - r * 0.3, r * 0.1, x, y, r);
            g.addColorStop(0, '#fff');
            g.addColorStop(0.5, '#eee');
            g.addColorStop(1, '#ccc');
            ctx.fillStyle = g;
        }
        ctx.fill();

        // Shine highlight
        ctx.beginPath();
        ctx.arc(x - r * 0.25, y - r * 0.25, r * 0.2, 0, Math.PI * 2);
        ctx.fillStyle = color === 1 ? 'rgba(255,255,255,0.15)' : 'rgba(255,255,255,0.5)';
        ctx.fill();
    }

    _drawStarPoints(ctx) {
        const points = this._getStarPoints();
        ctx.fillStyle = '#3a2a0a';
        for (const [r, c] of points) {
            const { x, y } = this._boardToCanvas(r, c);
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    _getStarPoints() {
        const s = this.boardSize;
        if (s === 9) return [[2,2],[2,6],[4,4],[6,2],[6,6]];
        if (s === 13) return [[3,3],[3,9],[6,6],[9,3],[9,9]];
        if (s === 19) return [[3,3],[3,9],[3,15],[9,3],[9,9],[9,15],[15,3],[15,9],[15,15]];
        if (s === 7) return [[3,3]];
        if (s === 15) return [[3,3],[3,11],[7,7],[11,3],[11,11]];
        return [];
    }

    _drawLabels(ctx) {
        ctx.fillStyle = '#5a4a2a';
        // Cap the font so labels stay small enough to fit in the padding band.
        ctx.font = `${Math.max(9, Math.min(13, this.cellSize * 0.3))}px Inter, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        // Center labels within the padding band, clear of edge stones.
        const off = this.padding * 0.28;
        const letters = 'ABCDEFGHJKLMNOPQRST'; // Skip 'I' in Go
        for (let i = 0; i < this.boardSize; i++) {
            const p = this.padding + i * this.cellSize;
            // Top/bottom column labels
            ctx.fillText(letters[i], p, off);
            ctx.fillText(letters[i], p, this.logicalHeight - off);
            // Left/right row labels
            const rowNum = this.boardSize - i;
            ctx.fillText(rowNum, off, p);
            ctx.fillText(rowNum, this.logicalWidth - off, p);
        }
    }

    _drawLastMoveMarker(ctx, row, col) {
        const { x, y } = this._boardToCanvas(row, col);
        const color = this.grid[row][col];
        ctx.beginPath();
        ctx.arc(x, y, this.stoneRadius * 0.3, 0, Math.PI * 2);
        ctx.fillStyle = color === 1 ? '#fff' : '#333';
        ctx.fill();
    }

    _drawHoverPreview(ctx, row, col) {
        const { x, y } = this._boardToCanvas(row, col);
        ctx.beginPath();
        ctx.arc(x, y, this.stoneRadius, 0, Math.PI * 2);
        ctx.fillStyle = this.currentPlayer === 1 ? 'rgba(0,0,0,0.3)' : 'rgba(255,255,255,0.3)';
        ctx.fill();
    }

    _drawSuggestion(ctx, row, col) {
        const { x, y } = this._boardToCanvas(row, col);
        ctx.beginPath();
        ctx.arc(x, y, this.stoneRadius * 0.6, 0, Math.PI * 2);
        ctx.strokeStyle = '#4caf50';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = 'rgba(76, 175, 80, 0.2)';
        ctx.fill();
    }

    _drawOwnership(ctx) {
        if (!this.ownershipMap) return;
        for (let r = 0; r < this.boardSize; r++) {
            for (let c = 0; c < this.boardSize; c++) {
                const val = this.ownershipMap[r]?.[c] || 0;
                if (Math.abs(val) < 0.1) continue;

                const { x, y } = this._boardToCanvas(r, c);
                const size = this.cellSize * 0.4;

                if (val < 0) {
                    // Black territory
                    ctx.fillStyle = `rgba(30, 30, 30, ${Math.abs(val) * 0.5})`;
                } else {
                    // White territory
                    ctx.fillStyle = `rgba(255, 255, 255, ${Math.abs(val) * 0.4})`;
                }
                ctx.fillRect(x - size, y - size, size * 2, size * 2);
            }
        }
    }
}
