import AdversialSearch as S
from Chess import *

def __main__():
    color = input("Choose color (B/W): ").capitalize() == "W"

    game = Chess()

    print("Enter \'Quit\' to quit any time.")
    while not len(game.all_moves()) == 0:
        print(game)
        print("")
        if game.white_move == color:
            while True:
                try:
                    start = input("Choose starting square (e.g. C6): ").capitalize()
                    if start == "Quit":
                        return
                    dest = input("Choose destination square (e.g. G5): ").capitalize()
                    if dest == "Quit":
                        return
                    promotion = input("Choose piece to promote to (press enter/return skip if not applicable): ").capitalize()
                    if promotion == "Quit":
                        return
                    start = Chess.convert_coords(start)
                    dest = Chess.convert_coords(dest)
                    if game.board[start] and Chess.ideally_reachable(game.board[start][0], start, dest[0] - start[0], dest[1] - start[1], game.white_move) and game.play_move(start, dest, promotion):
                        break
                    raise ValueError()
                except:
                    print("Invalid argument or move.")
        else:
            start, dest, promotion = S.best_move(game, max_depth=4)
            game.play_unchecked_move(start, dest, promotion)
            print(Chess.convert_coords(start) + " to " + Chess.convert_coords(dest))
            if promotion:
                print("Promote to " + promotion)

__main__()