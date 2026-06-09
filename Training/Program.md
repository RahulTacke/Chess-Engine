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

## Scope
Only files inside the `Training/` folder may be modified. Do not touch any files
outside this folder (Chess.py, AdversialSearch.py, DataAndParser/, etc.).
Do not run `git push` or any command that writes to the remote repository.
Git commits and all results stay local only.
Do not modify `Program.md`.
All output files (checkpoints, logs, etc.) must be written inside `Training/` — do not
use paths that resolve outside this folder.
Do not run destructive git operations: `git reset --hard`, `git clean`, or
`git checkout` on any file outside `Training/`.
Do not install, upgrade, or remove any Python packages (`pip`, `conda`, etc.).
Do not change `SUBSET` away from `200_000` or set `EPOCHS` above `10` — experiments
must complete in under 10 minutes.
Do not use `../` or any path that resolves outside `Training/`, including in
`DATA_DIR`, checkpoint paths, or any other file reference.
Do not execute any script outside `Training/` — do not call `lichessparse.py`,
`Chess.py`, `AdversialSearch.py`, or any other file in the parent directories.
Do not delete, overwrite, or modify any dataset files (`dataset_*.npz`) inside
`Training/` or any file in `DataAndParser/`.
Do not modify anything inside the `.git/` directory, including config and hooks.

## Baseline
Established on the first run. All experiments are judged against the baseline val_loss.
A change is kept only if it strictly improves val_loss.
