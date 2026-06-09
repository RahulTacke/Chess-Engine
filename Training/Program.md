# Chess CNN Experiment Program

## Goal
Minimize `val_loss` (MSE) on the chess position evaluation task. The model predicts
tanh-normalized Stockfish evaluations in (-1, 1) from an 8-channel board tensor.

## What can be changed
Modify `Evaluation.py` and/or the hyperparameters at the top of `train.py`.

### Architecture (Evaluation.py)
- Conv channel widths (currently 8→16→32→64→128→128)
- Number of conv layers (currently 5)
- FC layer sizes (currently 128→128→64→1)
- Number of FC layers
- Activation function (currently softsign throughout — must remain symmetric: f(-x) = -f(x))
- Dropout rate (currently nn.Dropout() = 0.5)
- Batch normalization after conv layers

### Hyperparameters (train.py)
- Learning rate (currently 1e-3)
- Batch size (currently 512)
- Optimizer: Adam vs AdamW; weight decay
- LR scheduler (e.g. ReduceLROnPlateau, CosineAnnealingLR)

## Constraints
- `in_channels` must stay 8 — do not change the input encoding
- The final output must pass through `torch.tanh` — targets are in (-1, 1)
- Activation functions must be symmetric (f(-x) = -f(x)) — the input uses ±1 encoding
  where +1 = white piece and -1 = black piece. Valid options: softsign, tanh, hardtanh
- Keep the model under ~1M parameters — this runs on a MacBook with MPS
- SUBSET = 200_000 for experiments; do not change this during the search

## What not to change
- Data loading logic
- Val/train split
- Loss function (MSELoss)
- Checkpoint saving logic

## Baseline
Established on the first run. All experiments are judged against the baseline val_loss.
A change is kept only if it strictly improves val_loss.
