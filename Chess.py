import numpy as np

class Chess:
    def __init__(self, test_setup=None):
        self.board = np.full((8, 8), None, dtype=object) # 8 x 8 board, None for empty squares, (piece, color) tuple for filles squares where White = True, Black = False, and indexed by file-rank (C7 => [2, 6])
        self.white_move = True # Move tracker
        self.castle_rights = [[True, True], [True, True]] # Can players castle? ((White can castle short, White can castle long), (Black can castle short, Black can castle long))
        self.en_passant = False # If the other player just (on the last move) pushed a pawn two squares, this is equal to the file of that move, otherwise it is False
        if test_setup:
            self.board = test_setup[0]
            self.white_move = test_setup[1]
            self.castle_rights = test_setup[2]
            self.en_passant = test_setup[3]
        else:
            self.board[0, 0] = self.board[7, 0] = ("R", True)
            self.board[1, 0] = self.board[6, 0] = ("N", True)
            self.board[2, 0] = self.board[5, 0] = ("B", True)
            self.board[3, 0] = ("Q", True)
            self.board[4, 0] = ("K", True)
            self.board[:, 1] = [("P", True)] * 8
            self.board[0, 7] = self.board[7, 7] = ("R", False)
            self.board[1, 7] = self.board[6, 7] = ("N", False)
            self.board[2, 7] = self.board[5, 7] = ("B", False)
            self.board[3, 7] = ("Q", False)
            self.board[4, 7] = ("K", False)
            self.board[:, 6] = [("P", False)] * 8
        self.piece_movements = self._precompute_piece_movements()
        self.attack_directions = self._precompute_attack_directions()
        self.king_locations = [(4, 0), (4, 7)]
        if test_setup:
            for loc, square in np.ndenumerate(self.board):
                if square and square[0] == "K":
                    self.king_locations[int(not square[1])] = loc
    
    def __str__(self):
        out = "\x1b[31m"
        out += "White\'s Turn" if self.white_move else "Black\'s Turn"
        out += "\x1b[0m\n"
        for i in range(7, -1, -1):
            out += "\x1b[31m" + str(i + 1) + "\x1b[0m"
            for j in range(8):
                if not self.board[j, i]:
                    out += "  "
                elif self.board[j, i][1]:
                    out += f" \x1b[37m{self.board[j, i][0]}\x1b[0m"
                else:
                    out += f" \x1b[30m{self.board[j, i][0]}\x1b[0m"
            out += "\n"
        out += " "
        for i in range(8):
            out += " \x1b[31m" + "ABCDEFGH"[i] + "\x1b[0m"
        return out

    # Determines if a proposed move is possible on an ideal board (no pieces in the way, an opponent piece on the appropriate square if a pawn is taking diagonally, a rook if castling)
    # Assumes start and destination are valid board sqaures
    @staticmethod
    def ideally_reachable(piece, start, distX, distY, white):
        if distX == 0 and distY == 0:
            return False
        match piece:
            case "K":
                return (start == (4, 0 if white else 7) and abs(distX) == 2 and distY == 0) or (abs(distX) <= 1 and abs(distY) <= 1)
            case "Q":
                return abs(distX) - abs(distY) == 0 or distX == 0 or distY == 0
            case "R":
                return distX == 0 or distY == 0
            case "B":
                return abs(distX) - abs(distY) == 0
            case "N":
                return (abs(distX) == 2 and abs(distY) == 1) or (abs(distX) == 1 and abs(distY) == 2)
            case "P":
                if distX == 0:
                    if white:
                        return distY == 1 or (start[1] == 1 and distY == 2)
                    return distY == -1 or (start[1] == 6 and distY == -2)
                return abs(distX) == 1 and (distY == 1 if white else distY == -1)

    # Determines if the specified color is in check on a given board
    @staticmethod
    def in_check(board, color, attack_directions, kingLocation):
        for (file, rank) in attack_directions[kingLocation[0]][kingLocation[1]]:
            square = board[file, rank]
            if square and square[1] != color:
                match square[0]:
                    case "Q" | "R" | "B":
                        displacement = (kingLocation[0] - file, kingLocation[1] - rank)
                        if displacement not in [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]:
                            directions = []
                            match square[0]:
                                case "Q":
                                    directions += [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)]
                                case "R":
                                    directions += [(-1, 0), (1, 0), (0, -1), (0, 1)]
                                case "B":
                                    directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                            factor = max(abs(displacement[0]), abs(displacement[1]))
                            direction = (int(displacement[0]/factor), int(displacement[1]/factor))
                            if direction in directions:
                                for i in range(1, 8):
                                    x = file + direction[0] * i
                                    y = rank + direction[1] * i
                                    if x in [-1, 8] or y in [-1, 8]:
                                        break
                                    if board[x, y]:
                                        if board[x, y] == ("K", color):
                                            return True
                                        break
                    case "K" | "N" | "P":
                        targets = []
                        match square[0]:
                            case "K":
                                targets = [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)]
                            case "N":
                                targets = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
                            case "P":
                                targets = [(-1, -1), (1, -1)] if color else [(-1, 1), (1, 1)]
                        if (kingLocation[0] - file, kingLocation[1] - rank) in targets:
                            return True
        return False                    

    # Determines if a proposed move is legal
    # Assumes start and destination are valid board sqaures
    def legal_move(self, start, dest, promotion=False):
        piece = self.board[start][0]
        if promotion and (piece != "P" or promotion not in ["Q", "R", "B", "N"] or dest[1] not in [0, 7]):
            return False
        if piece == "P" and dest[1] in [0, 7] and not promotion:
            return False
        distX = dest[0] - start[0]
        distY = dest[1] - start[1]
        if self.white_move != self.board[start][1] or not Chess.ideally_reachable(piece, start, distX, distY, self.white_move):
            return False
        if piece == "P" and (distX == 0) == (bool(self.board[dest]) or (self.en_passant == dest[0] and (start[1] == 4 if self.white_move else start[1] == 3))):
            return False
        if self.board[dest] and self.board[dest][1] == self.white_move:
            return False
        if piece != "K" or abs(distX) != 2:
            if abs(distX) == abs(distY):
                multX = int(distX / abs(distX))
                multY = int(distY / abs(distY))
                for i in range(1, abs(distX)):
                    if self.board[start[0] + multX * i, start[1] + multY * i]:
                        return False
            if distX == 0:
                multY = int(distY / abs(distY))
                for i in range(1, abs(distY)):
                    if self.board[start[0], start[1] + multY * i]:
                        return False
            if distY == 0:
                multX = int(distX / abs(distX))
                for i in range(1, abs(distX)):
                    if self.board[start[0] + multX * i, start[1]]:
                        return False
            checkBoard = self.board.copy()
            if piece == "P" and distX != 0 and checkBoard[dest] == None:
                checkBoard[dest[0], start[1]] = None
            checkBoard[dest] = checkBoard[start]
            checkBoard[start] = None
            return not Chess.in_check(checkBoard, self.white_move, self.attack_directions, dest if piece == "K" else self.king_locations[0 if self.board[start][1] else 1])
        if not self.castle_rights[0 if self.white_move else 1][0 if distX == 2 else 1]:
            return False
        if distX == 2:
            for i in range(5, 7):
                if self.board[i, start[1]]:
                    return False
        else:
            for i in range(1, 4):
                if self.board[i, start[1]]:
                    return False
        if Chess.in_check(self.board, self.white_move, self.attack_directions, self.king_locations[0 if self.board[start][1] else 1]):
            return False
        checkBoard = self.board.copy()
        if distX == 2:
            checkBoard[5, start[1]] = checkBoard[start]
        else:
            checkBoard[3, start[1]] = checkBoard[start]
        checkBoard[start] = None
        if Chess.in_check(checkBoard, self.white_move, self.attack_directions, (5, start[1]) if distX == 2 else (3, start[1])):
            return False
        if distX == 2:
            checkBoard[6, start[1]] = self.board[start]
            checkBoard[5, start[1]] = checkBoard[7, start[1]]
            checkBoard[7, start[1]] = None
        else:
            checkBoard[2, start[1]] = self.board[start]
            checkBoard[3, start[1]] = checkBoard[0, start[1]]
            checkBoard[0, start[1]] = None
        return not Chess.in_check(checkBoard, self.white_move, self.attack_directions, dest)
    
    # Generates a list of all legal moves that the current player can play as (start, dest, promotion) triples
    def all_moves(self):
        moves = []
        for start, square in np.ndenumerate(self.board):
            if square and square[1] == self.white_move:
                if square[0] == "P":
                    i = 5 if self.white_move else 6
                else:
                    i = ["K", "Q", "R", "B", "N"].index(square[0])
                for dest in self.piece_movements[start[0]][start[1]][i]:
                    if square[0] == "P" and dest[1] in [0, 7]:
                        for promotion in ["Q", "R", "B", "N"]:
                            if self.legal_move(start, dest, promotion):
                                moves.append((start, dest, promotion))
                    else:
                        if self.legal_move(start, dest):
                            moves.append((start, dest, False))
        return moves
    
    # Plays a move if it is legal, returning true if successful
    # Updates all tracking attributes as necessary
    def play_move(self, start, dest, promotion=False):
        if self.legal_move(start, dest, promotion):
            self.play_unchecked_move(start, dest, promotion)
            return True
        return False
    
    # Plays a move without checking if it is legal
    # Updates all tracking attributes as necessary
    def play_unchecked_move(self, start, dest, promotion=False):
        self.en_passant = False
        if self.board[start][0] == "P":
            if start[0] - dest[0] != 0 and self.board[dest] == None:
                self.board[dest[0], start[1]] = None
            if abs(start[1] - dest[1]) == 2:
                self.en_passant = start[0]
        if self.board[start][0] == "K":
            self.castle_rights[0 if self.white_move else 1] = [False, False]
            if dest[0] - start[0] == 2:
                self.board[5, start[1]] = self.board[7, start[1]]
                self.board[7, start[1]] = None
            if dest[0] - start[0] == -2:
                self.board[3, start[1]] = self.board[0, start[1]]
                self.board[0, start[1]] = None
            self.king_locations[0 if self.board[start][1] else 1] = dest
        if self.board[start][0] == "R":
            if start[0] in [0, 7]:
                self.castle_rights[0 if self.white_move else 1][0 if start[0] == 7 else 1] = False
        if self.board[dest] and self.board[dest][0] == "R":
            if dest[0] in [0, 7]:
                self.castle_rights[1 if self.white_move else 0][0 if dest[0] == 7 else 1] = False
        self.board[dest] = self.board[start]
        self.board[start] = None
        if promotion:
            self.board[dest] = (promotion, self.board[dest][1])
        self.white_move = not self.white_move

    def _precompute_piece_movements(self):
        piece_movements = [[[[] for i in range(7)] for j in range(8)] for k in range(8)]

        for file, rank in np.ndindex(self.board.shape):
            piece_movements[file][rank][0] = [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)]
            if file == 4 and rank in [0, 7]:
                piece_movements[file][rank][0].append((-2, 0))
                piece_movements[file][rank][0].append((2, 0))
            for i in range(1, 8):
                for j in [1, 2]:
                    piece_movements[file][rank][j].append((-i, 0))
                    piece_movements[file][rank][j].append((0, -i))
                    piece_movements[file][rank][j].append((0, i))
                    piece_movements[file][rank][j].append((i, 0))
                for j in [1, 3]:
                    piece_movements[file][rank][j].append((-i, -i))
                    piece_movements[file][rank][j].append((-i, i))
                    piece_movements[file][rank][j].append((i, -i))
                    piece_movements[file][rank][j].append((i, i))
            piece_movements[file][rank][4] = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
            if rank > 0:
                piece_movements[file][rank][5] = [(-1, 1), (0, 1), (1, 1)]
            if rank == 1:
                piece_movements[file][rank][5].append((0, 2))
            if rank < 7:
                piece_movements[file][rank][6] = [(-1, -1), (0, -1), (1, -1)]
            if rank == 6:
                piece_movements[file][rank][6].append((0, -2))
        for file, rank in np.ndindex(self.board.shape):
            for i in range(7):
                remList = []
                for j in range(len(piece_movements[file][rank][i])):
                    dest = (file + piece_movements[file][rank][i][j][0], rank + piece_movements[file][rank][i][j][1])
                    if dest[0] < 0 or dest[0] > 7 or dest[1] < 0 or dest[1] > 7:
                        remList.append(piece_movements[file][rank][i][j])
                    else:
                        piece_movements[file][rank][i][j] = dest
                for element in remList:
                    piece_movements[file][rank][i].remove(element)
        
        return piece_movements
    
    def _precompute_attack_directions(self):
        attack_directions = [[[] for j in range(8)] for k in range(8)]

        for file, rank in np.ndindex(self.board.shape):
            for loc in [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]:
                x = file + loc[0]
                y = rank + loc[1]
                if not (x < 0 or x > 7 or y < 0 or y > 7):
                    attack_directions[file][rank].append((x, y))
            for i in range(0, 8):
                if i != file:
                    attack_directions[file][rank].append((i, rank))
                if i != rank:
                    attack_directions[file][rank].append((file, i))
                if i != 0:
                    for dir in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        x = file + i * dir[0]
                        y = rank + i * dir[1]
                        if not (x < 0 or x > 7 or y < 0 or y > 7):
                            attack_directions[file][rank].append((x, y))
        return attack_directions