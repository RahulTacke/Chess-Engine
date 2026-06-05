import cProfile
import pstats
import io
import sys
sys.path.insert(0, '.')
import Chess
import AdversialSearch

game = Chess.Chess()

with cProfile.Profile() as pr:
    AdversialSearch.best_move(game, depth=5)

s = io.StringIO()
stats = pstats.Stats(pr, stream=s)
stats.sort_stats('cumulative')
stats.print_stats(20)
print(s.getvalue())
