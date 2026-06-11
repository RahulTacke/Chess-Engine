import cProfile
import pstats
import io
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Chess
import AdversialSearch

game = Chess.Chess()
print(game)
print()

# ── Timed best_move call ──────────────────────────────────────────────────────
MAX_DEPTH = 4

t0 = time.perf_counter()
with cProfile.Profile() as pr:
    move = AdversialSearch.best_move(game, max_depth=MAX_DEPTH)
elapsed = time.perf_counter() - t0

# ── Print result ──────────────────────────────────────────────────────────────
if move:
    start, dest, promo = move
    files = "ABCDEFGH"
    move_str = f"{files[start[0]]}{start[1]+1} → {files[dest[0]]}{dest[1]+1}"
    if promo:
        move_str += f" (promote to {promo})"
    print(f"Best move (depth {MAX_DEPTH}): {move_str}")
else:
    print("No legal moves.")
print(f"Time: {elapsed:.2f}s")
print()

# ── Profile stats ─────────────────────────────────────────────────────────────
s = io.StringIO()
stats = pstats.Stats(pr, stream=s)
stats.sort_stats('cumulative')
stats.print_stats(20)
print(s.getvalue())
