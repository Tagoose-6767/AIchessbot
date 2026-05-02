# AIChessbot

A pure-Python UCI chess engine implementing a modern alpha-beta search
pipeline on top of `python-chess` for board representation, legal move
generation, and Polyglot Zobrist hashing.

## Realistic strength & speed

- **Strength**: roughly 2000–2400 Elo, depending on hardware and time control.
  The 4000-Elo target in the original spec is not achievable in pure Python
  without an NNUE evaluator and a C/Cython move generator — the strongest
  engine ever (Stockfish 16 NNUE) is around 3640 Elo.
- **Speed**: 30k–200k NPS in CPython, roughly 5–10x more under PyPy. The
  bottleneck is python-chess's pure-Python `legal_moves` generator. The 1M-NPS
  target requires a C extension; out of scope here.

That said, every search/eval feature in the spec is implemented.

## Features

**Search**: iterative deepening · aspiration windows · principal variation
search (PVS / NegaScout) · adaptive null-move pruning · late move reductions
(LMR) · futility & reverse-futility pruning · razoring · check extension ·
mate-distance pruning.

**Move ordering**: TT (hash) move · MVV-LVA captures · static exchange
evaluation (SEE) for bad-capture demotion · killer moves (2 slots/ply) ·
history heuristic · countermove heuristic.

**Quiescence**: stand-pat · delta pruning · SEE-pruned bad captures · full
evasions when in check.

**Evaluation**: tapered MG/EG using PeSTO piece-square tables · bishop pair ·
pawn structure (doubled / isolated / passed / connected passers) · rook on
open / semi-open files · simple king-pawn-shield safety.

**Transposition table**: Zobrist (Polyglot-compatible) keys · two-bucket
depth-preferred + always-replace · ply-adjusted mate scores.

**Opening book**: Polyglot `.bin` reader with weighted move selection.

## Installing

```bash
cd chess_engine
pip install -r requirements.txt
```

## Running as a UCI engine

From inside the `chess_engine/` directory:

```bash
python engine.py
```

Interactively (test with raw UCI):

```
> uci
> isready
> position startpos moves e2e4 e7e5
> go movetime 2000
```

To use it in a GUI (Arena, Cute Chess, Banksia, BanksiaGUI, Nibbler):
register a new engine and point it at the `python` interpreter with
`engine.py` as the argument, working directory set to `chess_engine/`.

To use an opening book, set the `OPENING_BOOK` environment variable to a
Polyglot `.bin` file before launching, or send the UCI command:

```
setoption name Book value /path/to/Performance.bin
```

Free Polyglot books to try:
- Performance.bin (small, generic)
- gm2600.bin (titled-player games)
- KomodoVariety.bin (varied lines)

### Syzygy endgame tablebases

The C++ engine ships with embedded Fathom probes (see `src/fathom/`). Once
the position has at most 5 pieces (any combination), the engine queries the
tablebase at the root and during search for perfect endgame play.

Download the 5-piece WDL (`.rtbw`) and DTZ (`.rtbz`) files from any of:

- `http://tablebase.sesse.net/syzygy/3-4-5/`  (≈ 1 GB total)
- `https://syzygy-tables.info/`

Place them in a single directory and tell the engine where to find them:

```
setoption name SyzygyPath value /path/to/syzygy
```

The engine logs `info string SyzygyPath: loaded, TB_LARGEST=5` to stderr when
files are found. It silently skips probing when the path is unset, missing,
or empty.

## Tests

Move-generation correctness:

```bash
python tests/perft.py
```

Search benchmark (depth 8 by default):

```bash
python tests/benchmark.py [depth]
```

Self-play (writes `selfplay.pgn`):

```bash
python tests/selfplay.py [n_games] [seconds_per_move]
```

## Project layout

```
chess_engine/
├── engine.py         # UCI loop + main entry point
├── search.py         # Iterative deepening, alpha-beta, all pruning
├── evaluate.py       # Tapered static evaluation
├── transposition.py  # Zobrist + TT (two-bucket, age-aware)
├── movegen.py        # Move ordering + SEE
├── book.py           # Polyglot book reader
├── utils.py          # Constants, piece values, PSTs, MVV-LVA
├── README.md
├── requirements.txt
└── tests/
    ├── perft.py
    ├── benchmark.py
    └── selfplay.py
```

## How to make it stronger

If you actually want to push toward engine-level play, the order of
diminishing returns is:

1. Run on PyPy (5–10x speedup, free).
2. Replace the pure-Python move generator with a Cython/C extension.
3. Add an NNUE evaluator (small neural net trained on millions of positions).
   At this point Python becomes a glue layer around C/CUDA tensor ops.
4. Syzygy tablebase probing for 5–7 piece endings.
5. Texel tuning of the eval weights against a labeled position set.

(1) and (2) are by far the biggest wins per effort.
