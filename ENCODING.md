# CNN Input Encoding

The model expects a tensor of shape `(batch, 8, 8, 8)` in PyTorch channel-first format `(batch, C, H, W)`, where `C` is the channel (plane), `H` is the file, and `W` is the rank. A single position is encoded as `(8, 8, 8)` indexed `[channel, file, rank]` — no transpose is needed before batching.

Board coordinates are `[file, rank]` where file 0–7 = a–h and rank 0–7 = 1–8.

## Sign Convention

Positive values (+1) represent white features. Negative values (−1) represent black features. Zero means absent or empty.

---

## Channels 0–5: Pieces

One channel per piece type. Each square contains:
- `+1` if a white piece of that type occupies the square
- `−1` if a black piece of that type occupies the square
- `0` if the square is empty or occupied by a different piece type

| Channel | Piece |
|---------|-------|
| 0 | King |
| 1 | Queen |
| 2 | Rook |
| 3 | Bishop |
| 4 | Knight |
| 5 | Pawn |

---

## Channel 6: Castling Rights

A single channel encoding all four castling rights. All other squares are 0.

| Square | Coordinates `[file, rank]` | Value | Meaning |
|--------|----------------------------|-------|---------|
| h1 | (7, 0) | +1 | White can castle kingside |
| a1 | (0, 0) | +1 | White can castle queenside |
| h8 | (7, 7) | −1 | Black can castle kingside |
| a8 | (0, 7) | −1 | Black can castle queenside |

---

## Channel 7: En Passant

If en passant is available, the entire file (column) of the target pawn is marked. All other squares are 0.

The sign reflects whose turn it is (en passant is always capturable by the current player):
- `+1` across the file if it is white's turn
- `−1` across the file if it is black's turn
