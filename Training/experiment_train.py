import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Evaluation import Evaluation

# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS       = 10
BATCH_SIZE   = 512
LR           = 3e-3
VAL_SPLIT    = 0.1
SUBSET       = 500_000   # positions to sample per experiment (None = use all)
SUBSET_SEED  = 42        # fixed seed for reproducible subset sampling
DATA_DIR     = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT   = os.path.join(os.path.dirname(__file__), 'best_model.pt')
# ─────────────────────────────────────────────────────────────────────────────

device = (
    torch.device('mps')  if torch.backends.mps.is_available() else
    torch.device('cuda') if torch.cuda.is_available()         else
    torch.device('cpu')
)
print(f"Device: {device}")

# ── Load data ─────────────────────────────────────────────────────────────────
HOLDOUT = 'dataset_1.npz'  # held out for final post-experiment testing
all_chunks = sorted(glob.glob(os.path.join(DATA_DIR, 'dataset_*.npz')))
chunks = [p for p in all_chunks if os.path.basename(p) != HOLDOUT]
assert chunks, f"No usable dataset_*.npz files found in {DATA_DIR}"

all_boards, all_evals = [], []
for path in chunks:
    data = np.load(path)
    all_boards.append(data['boards'])
    all_evals.append(data['evals'])
    print(f"Loaded {path}  ({len(data['evals'])} positions)")

boards = np.concatenate(all_boards, axis=0)
evals  = np.concatenate(all_evals,  axis=0)

if SUBSET is not None and SUBSET < len(evals):
    rng    = np.random.default_rng(SUBSET_SEED)
    idx    = rng.choice(len(evals), SUBSET, replace=False)
    boards = boards[idx]
    evals  = evals[idx]

print(f"Total positions: {len(evals)}")

boards_t = torch.from_numpy(boards).float()
evals_t  = torch.from_numpy(evals).float().unsqueeze(1)

dataset  = TensorDataset(boards_t, evals_t)
val_n    = int(len(dataset) * VAL_SPLIT)
train_n  = len(dataset) - val_n
train_ds, val_ds = random_split(dataset, [train_n, val_n],
                                generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=False)

# ── Model ─────────────────────────────────────────────────────────────────────
model     = Evaluation().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
criterion = nn.MSELoss()

# ── Train ─────────────────────────────────────────────────────────────────────
best_val = float('inf')

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(x)
    train_loss /= train_n

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            val_loss += criterion(model(x), y).item() * len(x)
    val_loss /= val_n

    scheduler.step()
    print(f"Epoch {epoch:3d}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  lr={optimizer.param_groups[0]['lr']:.2e}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), CHECKPOINT)
        print(f"           --> saved checkpoint (val_loss={best_val:.6f})")

print(f"\nDone. Best val_loss: {best_val:.6f}")
