import AdversialSearch as S
from Chess import *

def __main__():
    color = input("Choose color (B/W): ").capitalize() == "W"

    game = Chess()
    while not len(game.all_moves()) == 0:
        print(game)
        print("")
        if game.white_move == color:
            while True:
                try:
                    start = Chess.convert_coords(input("Choose starting square (e.g. C6): ").capitalize())
                    dest = Chess.convert_coords(input("Choose destination square (e.g. G5): ").capitalize())
                    promotion = input("Choose piece to promote to (press enter/return skip if not applicable): ")
                    if game.board[start] and Chess.ideally_reachable(game.board[start][0], start, dest[0] - start[0], dest[1] - start[1], game.white_move) and game.play_move(start, dest, promotion):
                        break
                    raise ValueError()
                except:
                    print("Invalid argument or move.")
        else:
            start, dest, promotion = S.best_move(game)
            game.play_unchecked_move(start, dest, promotion)
            print(Chess.convert_coords(start) + " to " + Chess.convert_coords(dest))
            if promotion:
                print("Promote to " + promotion)

__main__()