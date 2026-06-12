# Chess AI with CNN Evaluation

A fully playable command-line chess engine that pits a human player against an AI opponent. The AI combines a trained Convolutional Neural Network (CNN) position evaluator with classic alpha-beta adversarial search to select moves.

---

## 1. Overview

The core challenge this project addresses is **chess position evaluation**: given a board state, how good is it for the side to move? Traditional engines use hand-crafted heuristics (material counts, piece-square tables). This project trains a CNN on real chess games to learn evaluation automatically, then plugs that model into a search algorithm to play full games.

The pipeline has three stages:

1. **Data parsing** — A PGN file of Lichess games is parsed into board tensors and evaluation labels, saved as chunked `.npz` files.
2. **Model training** — A 5-layer CNN is trained on those tensors to predict a scalar position score in the range `[−1, +1]` (negative = Black winning, positive = White winning).
3. **Gameplay** — The trained model is loaded as a leaf evaluator inside an iterative-deepening alpha-beta search. A human plays against the AI from the command line.

The board is encoded as an `(8, 8, 8)` tensor (8 channels × 8 files × 8 ranks). Channels encode piece locations by type, castling rights, and en passant availability using a `+1 / −1 / 0` sign convention for White/Black/empty. See [`ENCODING.md`](ENCODING.md) for the full specification.

---

## 2. Environment Setup

### Prerequisites

- Python **3.10+**
- `pip` or `conda`
- A CUDA-capable GPU is strongly recommended for training (CPU inference works fine for gameplay)

### Option A — pip (virtual environment)

```bash
python -m venv chess-env
source chess-env/bin/activate        # Windows: chess-env\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy
```

> If you do not have a CUDA GPU, install the CPU-only build instead:
> ```bash
> pip install torch torchvision
> ```

### Option B — conda

```bash
conda create -n chess python=3.11 numpy -y
conda activate chess
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Verify the install

```python
import torch, numpy
print(torch.__version__, numpy.__version__)
print("CUDA available:", torch.cuda.is_available())
```

---

## 3. Reproducing the Results

### Step 1 — Obtain training data

Download a Lichess game database (`.pgn.zst`) from [https://database.lichess.org](https://database.lichess.org) and place it at:

```
DataAndParser/lichessdata.pgn.zst
```

### Step 2 — Parse the PGN into training tensors

```bash
python3 DataAndParser/lichessparse.py
```

This reads the compressed PGN and writes chunked `.npz` dataset files into `DataAndParser/`. Each file contains board tensors and their corresponding evaluation labels. On a large PGN this can run for several hours; you can stop it once enough chunks have been written (`ls -lh DataAndParser/`).

**On an HPC cluster (Slurm):**
```bash
sbatch parse.sbatch
```

### Step 3 — Train the CNN

```bash
python3 Training/final_train.py
```

This loads all `dataset_*.npz` files, trains the `Evaluation` CNN, and saves the final weights to `Training/final_model.pt`.

**On an HPC cluster (Slurm):**
```bash
sbatch train.sbatch
```
The job requests 1 GPU, 4 CPUs, and 64 GB RAM for up to 8 hours (adjust as needed in `train.sbatch`).

### Step 4 — Play against the AI

```bash
python3 Main.py
```

You will be prompted to choose a color. Enter moves in algebraic notation (e.g. `E2` → `E4`). The AI searches to depth 4 by default. Type `Quit` at any prompt to exit.

**To adjust search depth**, edit the `max_depth` argument in `Main.py`:

```python
start, dest, promotion = S.best_move(game, max_depth=4)
```

**To disable the CNN** and use the hand-crafted piece-square table evaluator instead, set in `AdversialSearch.py`:

```python
USE_NN = False
```

---

## 4. Code Organization

```
.
├── Main.py                    # Entry point — human vs. AI game loop
├── Chess.py                   # Chess game engine (board, rules, move generation)
├── AdversialSearch.py         # Alpha-beta search, iterative deepening, evaluation blend
├── ENCODING.md                # Specification for the (8,8,8) board tensor format
│
├── Training/
│   ├── Evaluation.py          # CNN architecture definition (5 conv + 4 FC layers)
│   ├── final_train.py         # Training script (loads .npz data, trains, saves model)
│   └── final_model.pt         # Saved model weights (produced by training)
│
├── DataAndParser/
│   ├── lichessparse.py        # PGN parser → board tensors + labels → dataset_*.npz
│   └── dataset_*.npz          # Chunked training data (git-ignored)
│
├── parse.sbatch               # Slurm job script for data parsing
└── train.sbatch               # Slurm job script for model training
```

### Module descriptions

**`Chess.py`** — Implements the full chess ruleset: piece movement, legal move generation, check detection, castling, en passant, promotion, and an incrementally maintained `(8, 8, 8)` NumPy tensor that mirrors the board state for fast CNN inference.

**`AdversialSearch.py`** — Contains the AI logic. `best_move()` runs iterative-deepening alpha-beta search with aspiration windows and move ordering. Leaf nodes are evaluated by blending the CNN score with a traditional piece-square table score (controlled by `NN_BLEND`). At depth 1, all child positions are batched and evaluated in a single GPU forward pass for efficiency.

**`Training/Evaluation.py`** — Defines the `Evaluation` CNN: five `Conv2d` layers (8→16→32→64→128→128 channels, 3×3 kernels, same padding) followed by global average pooling and four fully connected layers (128→128→128→64→1). Activations are `softsign` throughout, with a final `tanh` to bound output to `[−1, +1]`.

**`DataAndParser/lichessparse.py`** — Reads a `.pgn.zst` Lichess database, converts each position to the tensor encoding described in `ENCODING.md`, pairs it with the game's evaluation, and flushes chunks to `.npz` files.

**`Main.py`** — The game loop. Handles user input (with validation), delegates AI moves to `AdversialSearch.best_move()`, and prints the board after each turn.
