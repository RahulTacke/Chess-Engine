import sys
import os
import re
import math
import numpy as np
import chess
import zstandard as zstd

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.join(SCRIPT_DIR, '..', 'Training')

PIECE_TO_PLANE = {
    chess.KING: 0,
    chess.QUEEN: 1,
    chess.ROOK: 2,
    chess.BISHOP: 3,
    chess.KNIGHT: 4,
    chess.PAWN: 5,
}

def boardToTensor(board):
    tensor = np.zeros((8, 8, 8), dtype=np.float32)
    for piece_type in chess.PIECE_TYPES:
        plane = PIECE_TO_PLANE[piece_type]
        for square in board.pieces(piece_type, chess.WHITE):
            tensor[plane, chess.square_file(square), chess.square_rank(square)] = 1.0
        for square in board.pieces(piece_type, chess.BLACK):
            tensor[plane, chess.square_file(square), chess.square_rank(square)] = -1.0
    if board.has_kingside_castling_rights(chess.WHITE):  tensor[6, 7, 0] =  1.0
    if board.has_queenside_castling_rights(chess.WHITE): tensor[6, 0, 0] =  1.0
    if board.has_kingside_castling_rights(chess.BLACK):  tensor[6, 7, 7] = -1.0
    if board.has_queenside_castling_rights(chess.BLACK): tensor[6, 0, 7] = -1.0
    if board.ep_square is not None:
        f = chess.square_file(board.ep_square)
        tensor[7, f, :] = 1.0 if board.turn == chess.WHITE else -1.0
    return tensor

boards = []
evals = []
chunkIndex = 0
CHUNK_SIZE = 5000000 ##############<> CHUNKSIZE <>############
currentHeaders = {}
currentMoves = []

def saveChunk():
    global boards, evals, chunkIndex
    path = os.path.join(TRAINING_DIR, f"dataset_{chunkIndex}.npz")
    np.savez(path, boards=np.array(boards), evals=np.array(evals))
    print(f"Saved chunk {chunkIndex} with {len(evals)} positions to {path}")
    boards = []
    evals = []
    chunkIndex += 1

def processGame(currentHeaders, currentMoves):
    if currentHeaders.get("Termination") == "Time forfeit": return
    if "BOT" in currentHeaders.get("WhiteTitle", ""): return
    if "BOT" in currentHeaders.get("BlackTitle", ""): return
    if "Variant" in currentHeaders: return

    board = chess.Board()
    gameBoards = []
    gameEvals = []

    movetext = " ".join(currentMoves)
    moveJustPlayed = False
    for token in re.finditer(r'\{([^}]*)\}|(\S+)', movetext):
        comment, word = token.group(1), token.group(2)

        if comment is not None:
            if moveJustPlayed:
                m = re.search(r'\[%eval ([^\]]+)\]', comment)
                if m:
                    evalStr = m.group(1)
                    if evalStr.startswith("-#"):
                        evalVal = -1.0
                    elif evalStr.startswith("#"):
                        evalVal = 1.0
                    else:
                        evalVal = math.tanh(float(evalStr) / 4.0)
                    gameBoards.append(boardToTensor(board))
                    gameEvals.append(evalVal)
            moveJustPlayed = False
            continue

        if re.fullmatch(r'\d+\.+', word) or word in ("1-0", "0-1", "1/2-1/2", "*"):
            continue
        san = word.rstrip("?!")
        if not san or san.startswith("$"):
            continue
        try:
            board.push_san(san)
        except ValueError:
            break
        moveJustPlayed = True

    if gameEvals:
        boards.extend(gameBoards)
        evals.extend(gameEvals)

with open(os.path.join(SCRIPT_DIR, "lichessdata.pgn.zst"), "rb") as f:
    dctx = zstd.ZstdDecompressor()
    stream = dctx.stream_reader(f)
    buffer = b""
    gamesProcessed = 0

    while True:
        chunk = stream.read(65536)
        if not chunk:
            break
        buffer += chunk
        lines = buffer.split(b"\n")
        buffer = lines[-1]

        for line in lines[:-1]:
            line = line.decode("utf-8", errors="ignore").strip()

            if line.startswith("["):
                key = re.match(r'\[(\w+)', line).group(1)
                value = re.search(r'"(.+)"', line).group(1)
                currentHeaders[key] = value

            elif line == "":
                processGame(currentHeaders, currentMoves)
                gamesProcessed += 1
                if gamesProcessed % 10000 == 0:
                    print(f"Games processed: {gamesProcessed}, positions: {len(evals) + chunkIndex * CHUNK_SIZE}")
                if len(evals) >= CHUNK_SIZE:
                    saveChunk()
                currentHeaders = {}
                currentMoves = []

            else:
                currentMoves.append(line)

if evals:
    saveChunk()

print(f"Done. {chunkIndex} chunks saved.")