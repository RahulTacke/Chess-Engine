import numpy as np
import Chess

# ─────────────────────────────────────────────
# Piece-square tables (White's perspective; mirrored for Black)
# Values are centipawns (1 pawn = 100)
# ─────────────────────────────────────────────

PIECE_VALUES = {"P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 20000}

PST = {
    "P": [
         0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
         5,  5, 10, 25, 25, 10,  5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5, -5,-10,  0,  0,-10, -5,  5,
         5, 10, 10,-20,-20, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0,
    ],
    "N": [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    ],
    "B": [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    ],
    "R": [
         0,  0,  0,  0,  0,  0,  0,  0,
         5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
         0,  0,  0,  5,  5,  0,  0,  0,
    ],
    "Q": [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
         -5,  0,  5,  5,  5,  5,  0, -5,
          0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20,
    ],
    "K": [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
         20, 20,  0,  0,  0,  0, 20, 20,
         20, 30, 10,  0,  0, 10, 30, 20,
    ],
}


def _pst_index(file: int, rank: int, white: bool) -> int:
    """
    Convert (file, rank) board coords to a PST index.
    PST rows are from rank-8 down to rank-1 (standard orientation).
    White reads the table as-is; Black's table is mirrored vertically.
    """
    if white:
        row = 7 - rank  # rank 7 → row 0, rank 0 → row 7
    else:
        row = rank      # rank 0 → row 0 (mirror)
    return row * 8 + file


def evaluate(game: Chess) -> int:
    """
    Static board evaluation in centipawns from White's perspective.
    Positive = White is better; Negative = Black is better.
    Returns +100000 / -100000 as mate proxies (not used directly —
    the search detects no-moves and scores accordingly).
    """
    score = 0
    for (file, rank), square in np.ndenumerate(game.board):
        if square is None:
            continue
        piece, is_white = square
        material = PIECE_VALUES[piece]
        positional = PST[piece][_pst_index(file, rank, is_white)]
        if is_white:
            score += material + positional
        else:
            score -= material + positional
    return score


# ─────────────────────────────────────────────
# Alpha-Beta Search
# ─────────────────────────────────────────────

INF = float("inf")


def alpha_beta(game: Chess, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
    """
    Depth-limited alpha-beta pruning.

    Parameters
    ----------
    game        : Chess instance (will be mutated and restored in-place)
    depth       : plies remaining
    alpha       : best score the MAX player can guarantee so far
    beta        : best score the MIN player can guarantee so far
    maximizing  : True  → current player is trying to maximise (White)
                  False → current player is trying to minimise (Black)

    Returns
    -------
    Centipawn score from White's perspective.
    """
    moves = game.all_moves()

    # ── Terminal / leaf node ──────────────────────────────────────────
    if depth == 0 or not moves:
        if not moves:
            # No legal moves: checkmate or stalemate
            if Chess.in_check(game.board, game.white_move):
                # Checkmate — return a large score favouring the side that
                # delivered it.  Adding depth rewards faster mates.
                return -INF + (10 - depth) if maximizing else INF - (10 - depth)
            return 0  # Stalemate
        return evaluate(game)

    # Snapshot mutable state so moves can be undone
    board_snapshot    = game.board.copy()
    castle_snapshot   = [list(row) for row in game.castle_rights]
    ep_snapshot       = game.en_passant
    white_move_snap   = game.white_move
    king_snap         = list(game.king_locations)

    def restore():
        game.board          = board_snapshot.copy()
        game.castle_rights  = [list(row) for row in castle_snapshot]
        game.en_passant     = ep_snapshot
        game.white_move     = white_move_snap
        game.king_locations = list(king_snap)

    # ── MAX node (White to move) ──────────────────────────────────────
    if maximizing:
        value = -INF
        for start, dest, promo in moves:
            game.play_unchecked_move(start, dest, promo)
            value = max(value, alpha_beta(game, depth - 1, alpha, beta, False))
            restore()
            alpha = max(alpha, value)
            if beta <= alpha:
                break           # β cut-off
        return alpha

    # ── MIN node (Black to move) ──────────────────────────────────────
    else:
        value = INF
        for start, dest, promo in moves:
            game.play_unchecked_move(start, dest, promo)
            value = min(value, alpha_beta(game, depth - 1, alpha, beta, True))
            restore()
            beta = min(beta, value)
            if beta <= alpha:
                break           # α cut-off
        return beta


def best_move(game: Chess, depth: int = 3):
    """
    Return the best (start, dest, promotion) triple for the current player
    using alpha-beta search to the given depth.

    Returns None if the position has no legal moves (game over).
    """
    moves = game.all_moves()
    if not moves:
        return None

    maximizing = game.white_move   # White maximises, Black minimises
    best       = None
    alpha      = -INF
    beta       = INF

    board_snapshot    = game.board.copy()
    castle_snapshot   = [list(row) for row in game.castle_rights]
    ep_snapshot       = game.en_passant
    white_move_snap   = game.white_move
    king_snap         = list(game.king_locations)

    def restore():
        game.board          = board_snapshot.copy()
        game.castle_rights  = [list(row) for row in castle_snapshot]
        game.en_passant     = ep_snapshot
        game.white_move     = white_move_snap
        game.king_locations = list(king_snap)

    for start, dest, promo in moves:
        game.play_unchecked_move(start, dest, promo)
        score = alpha_beta(game, depth - 1, alpha, beta, not maximizing)
        restore()

        if maximizing:
            if score > alpha:
                alpha = score
                best  = (start, dest, promo)
        else:
            if score < beta:
                beta  = score
                best  = (start, dest, promo)

    return best


# ─────────────────────────────────────────────
# Quick demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    game = Chess()
    print(game)
    # Adjust depth search here
    move = best_move(game, depth=3)
    files = "ABCDEFGH"
    s, d, p = move
    print(f"\nBest move: {files[s[0]]}{s[1]+1} → {files[d[0]]}{d[1]+1}"
          + (f" (promote to {p})" if p else ""))