import os
import importlib.util
import numpy as np
import torch
import Chess

# ─────────────────────────────────────────────
# Trained CNN evaluator (Training/final_model.pt)
# Set USE_NN = False to fall back to the pure PST/material eval.
# ─────────────────────────────────────────────

USE_NN   = True
NN_BLEND = 1.0   # leaf score = NN_BLEND * CNN + (1 - NN_BLEND) * PST/material

_MODEL  = None
_DEVICE = None

def _load_model():
    global _MODEL, _DEVICE
    if _MODEL is None:
        _DEVICE = (
            torch.device("mps")  if torch.backends.mps.is_available() else
            torch.device("cuda") if torch.cuda.is_available()          else
            torch.device("cpu")
        )
        here = os.path.dirname(os.path.abspath(__file__))
        spec = importlib.util.spec_from_file_location(
            "training_eval", os.path.join(here, "Training", "Evaluation.py"))
        te = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(te)
        model = te.Evaluation()
        model.load_state_dict(torch.load(
            os.path.join(here, "Training", "final_model.pt"),
            map_location=_DEVICE))
        model.to(_DEVICE)
        model.eval()
        print(f"[AdversialSearch] inference device: {_DEVICE}")
        _MODEL = model
    return _MODEL

def nn_evaluate(game) -> float:
    """CNN leaf value in centipawns (White's perspective)."""
    model = _load_model()
    x = torch.from_numpy(game.tensor).unsqueeze(0).float().to(_DEVICE)
    with torch.no_grad():
        v = model(x).squeeze().clamp(-0.999, 0.999)
        return float(4.0 * torch.atanh(v) * 100.0)  # single .cpu() transfer via float()

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


def pst_evaluate(game: Chess) -> int:
    """
    Static board evaluation in centipawns from White's perspective.
    Positive = White is better; Negative = Black is better.
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


def evaluate(game: Chess) -> float:
    """
    Leaf evaluation in centipawns (White's perspective). Blends the trained
    CNN with the PST/material eval so saturated CNN values in clearly winning
    positions still convert. Falls back to PST alone when USE_NN is False.
    """
    if not USE_NN:
        return pst_evaluate(game)
    return NN_BLEND * nn_evaluate(game) + (1 - NN_BLEND) * pst_evaluate(game)


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
            if Chess.Chess.in_check(game.board, game.white_move, game.attack_directions,
                                    game.king_locations[0 if game.white_move else 1]):
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
    tensor_snap       = game.tensor.copy()

    def restore():
        game.board          = board_snapshot.copy()
        game.castle_rights  = [list(row) for row in castle_snapshot]
        game.en_passant     = ep_snapshot
        game.white_move     = white_move_snap
        game.king_locations = list(king_snap)
        game.tensor         = tensor_snap.copy()

    # ── Batched leaf evaluation at depth 1 ───────────────────────────
    if depth == 1 and USE_NN:
        batch_tensors = []
        for start, dest, promo in moves:
            game.play_unchecked_move(start, dest, promo)
            batch_tensors.append(game.tensor.copy())
            restore()

        model = _load_model()
        batch = torch.from_numpy(np.stack(batch_tensors)).float().to(_DEVICE)
        with torch.no_grad():
            scores_t = 4.0 * torch.atanh(model(batch).squeeze(1).clamp(-0.999, 0.999)) * 100.0
            if NN_BLEND >= 1.0:
                return float(scores_t.max() if maximizing else scores_t.min())
            nn_vals = scores_t.cpu().tolist()

        scores = []
        for j, (start, dest, promo) in enumerate(moves):
            game.play_unchecked_move(start, dest, promo)
            pst_cp = pst_evaluate(game)
            restore()
            scores.append(NN_BLEND * nn_vals[j] + (1 - NN_BLEND) * pst_cp)
        return max(scores) if maximizing else min(scores)

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


# ─────────────────────────────────────────────
# Iterative Deepening
# ─────────────────────────────────────────────

def best_move(game: Chess, max_depth: int = 4):
    """
    Return the best (start, dest, promotion) triple for the current player
    using iterative deepening alpha-beta search up to max_depth.

    The search proceeds depth-by-depth (1, 2, …, max_depth).  After each
    completed iteration the best move found is placed first in the move
    list for the next iteration, improving alpha-beta cut-offs and making
    each successive search significantly faster in practice.

    Returns None if the position has no legal moves (game over).
    """
    moves = game.all_moves()
    if not moves:
        return None

    maximizing = game.white_move   # White maximises, Black minimises

    board_snapshot    = game.board.copy()
    castle_snapshot   = [list(row) for row in game.castle_rights]
    ep_snapshot       = game.en_passant
    white_move_snap   = game.white_move
    king_snap         = list(game.king_locations)
    tensor_snap       = game.tensor.copy()

    def restore():
        game.board          = board_snapshot.copy()
        game.castle_rights  = [list(row) for row in castle_snapshot]
        game.en_passant     = ep_snapshot
        game.white_move     = white_move_snap
        game.king_locations = list(king_snap)
        game.tensor         = tensor_snap.copy()

    best       = moves[0]   # fallback: always have something to return
    prev_score = 0          # centre of the aspiration window (updated each iteration)
    DELTA      = 50         # initial half-width in centipawns

    for depth in range(1, max_depth + 1):
        ordered_moves = [best] + [m for m in moves if m != best]

        # ── Aspiration window ─────────────────────────────────────────
        # Start with a narrow window around the previous iteration's score.
        # Widen exponentially on fail-high / fail-low until the search fits.
        if depth == 1:
            alpha, beta = -INF, INF   # full window on first pass
        else:
            alpha = prev_score - DELTA
            beta  = prev_score + DELTA

        while True:
            current_best = ordered_moves[0]
            best_score   = -INF if maximizing else INF

            for start, dest, promo in ordered_moves:
                game.play_unchecked_move(start, dest, promo)
                score = alpha_beta(game, depth - 1, alpha, beta, not maximizing)
                restore()

                if maximizing:
                    if score > best_score:
                        best_score   = score
                        current_best = (start, dest, promo)
                    alpha = max(alpha, best_score)
                else:
                    if score < best_score:
                        best_score   = score
                        current_best = (start, dest, promo)
                    beta = min(beta, best_score)

            # Check whether the result fell outside the window
            if best_score <= prev_score - DELTA and depth > 1:
                DELTA *= 2          # fail-low: widen downward and re-search
                alpha  = prev_score - DELTA
            elif best_score >= prev_score + DELTA and depth > 1:
                DELTA *= 2          # fail-high: widen upward and re-search
                beta   = prev_score + DELTA
            else:
                break               # result is within window — accept it

        prev_score = best_score
        DELTA      = 50             # reset window for next depth
        best       = current_best

    return best


# ─────────────────────────────────────────────
# Quick demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    game = Chess()
    print(game)
    # Adjust max search depth here
    move = best_move(game, max_depth=4)
    files = "ABCDEFGH"
    s, d, p = move
    print(f"\nBest move: {files[s[0]]}{s[1]+1} → {files[d[0]]}{d[1]+1}"
          + (f" (promote to {p})" if p else ""))