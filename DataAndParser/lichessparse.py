import sys
import os
import re
import numpy as np
import chess
import zstandard as zstd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PIECE_MAP = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}

def boardToTensor(board):
    tensor = np.zeros((8, 8, 12), dtype=np.int8)
    for piece_type in chess.PIECE_TYPES:
        for color in chess.COLORS:
            plane = PIECE_MAP[(piece_type, color)]
            for square in board.pieces(piece_type, color):
                tensor[chess.square_file(square), chess.square_rank(square), plane] = 1
    return tensor

boards = []
evals = []
chunkIndex = 0
CHUNK_SIZE = 5000000 ##############<> CHUNKSIZE <>############
currentHeaders = {}
currentMoves = []

def saveChunk():
    global boards, evals, chunkIndex
    path = os.path.join(SCRIPT_DIR, f"dataset_{chunkIndex}.npz")
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

    for moveLine in currentMoves:
        evalMatches = re.findall(r'\[%eval ([^\]]+)\]', moveLine)

        clean = re.sub(r'\{[^}]*\}', '', moveLine)
        clean = re.sub(r'\d+\.+', '', clean)
        clean = re.sub(r'[?!]+', '', clean)
        sanMoves = clean.split()

        for i, san in enumerate(sanMoves):
            isWhite = board.turn == chess.WHITE
            try:
                board.push_san(san)
            except:
                break
            if i < len(evalMatches):
                evalStr = evalMatches[i]
                if evalStr.startswith("-#"):
                    evalVal = -10.0
                elif evalStr.startswith("#"):
                    evalVal = 10.0
                else:
                    evalVal = float(evalStr)
                evalVal = max(-10.0, min(10.0, evalVal))
                if not isWhite:
                    evalVal = -evalVal
                gameBoards.append(boardToTensor(board))
                gameEvals.append(evalVal)

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