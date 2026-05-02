# Texel Tuner

A standalone tuner that fits AIChessbot's hand-crafted evaluation weights to a
labeled position dataset by minimizing sigmoid(K·eval) − label MSE.

## What it tunes

Everything in the `EvalWeights` struct (`src/evaluate.h`):

- piece values (MG / EG)
- six 64-square piece-square tables (MG / EG)
- pawn structure: passed-pawn bonuses, connected, doubled, isolated
- rook on open / semi-open file
- king-shelter slots (close, far, missing, behind)
- mobility (MG / EG)
- bishop pair (MG / EG)
- tempo

Total: 817 free parameters.

## How to run

1. Fetch the dataset (Bitbucket, ~40 MB compressed):

   ```
   curl -L https://bitbucket.org/zurichess/tuner/downloads/tuner.7z -o tuner.7z
   python -c "import py7zr; py7zr.SevenZipFile('tuner.7z').extract(targets=['quiet-labeled.epd'])"
   ```

   Place `quiet-labeled.epd` next to `tuner.cpp`. The dataset ships 725 000
   labeled positions in EPD format (`<fen> c9 "1-0|0-1|1/2-1/2";`).

2. Build:

   ```
   make tuner
   ```

3. Run from the `chess_engine/` directory:

   ```
   ./tuner/tuner.exe --epd tuner/quiet-labeled.epd --iters 10000 --sample 725000 --lr 1.0
   ```

The tuner will:

- load + extract sparse features from every position (one pass)
- sanity-check that feature-eval matches `evaluate_hce()` exactly (200/200)
- solve K via golden-section search
- run Adam gradient descent over all 817 parameters
- write the tuned literals back into `src/evaluate.cpp` between the markers
  `// === TEXEL-TUNED-WEIGHTS-BEGIN ===` and `// === TEXEL-TUNED-WEIGHTS-END ===`

After tuning completes, rebuild the engine to pick up the new weights:

```
make
```

All progress output goes to **stderr**; UCI traffic is reserved for stdout.

## Original dataset attribution

`quiet-labeled.epd` is from the zurichess project by Alexandru Moșoi
(`https://bitbucket.org/zurichess/tuner`). See `LICENSE` for the BSD-2-Clause
terms. The dataset itself is **not** committed to this repository — fetch it
separately as documented above.
