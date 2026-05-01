# AIChessbot — C++ engine

UCI chess engine in C++17 with magic-bitboard move generation, full
alpha-beta search pipeline (PVS, NMP, LMR, futility, razoring), tapered
PeSTO evaluation, Zobrist transposition table, and the standard UCI
protocol.

## Realistic strength

This is a hand-written engine of moderate scope (~2,500 LOC), no NNUE.
Expected playing strength is roughly **2400–2800 Elo** with mature search
and the PeSTO eval. The 3600-Elo target in the original spec is the realm
of NNUE-equipped engines (Stockfish, Lc0); reaching it requires a neural
evaluator and several engineer-years of tuning.

## Building

### MinGW / Linux / macOS (gcc/clang)

```
cd chess_engine
make            # produces chess_engine.exe (Windows) or chess_engine
```

### CMake (any platform)

```
cd chess_engine
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Windows MSVC

```
cmake -B build -G "Visual Studio 17 2022"
cmake --build build --config Release
```

## Running

UCI shell:

```
./chess_engine
> uci
> isready
> position startpos moves e2e4 e7e5
> go movetime 1000
```

Or load it in a UCI GUI (Arena, Cute Chess, Banksia) by pointing the
GUI at the `chess_engine` binary.

## Built-in commands

Beyond standard UCI:

- `perft N` — count leaf nodes at depth N from the current position.
- `divide N` — perft broken down per first move.
- `bench [depth]` — search 10 standard positions to the given depth and
  report total nodes / NPS. Without arguments, depth = 13.
- `d` / `print` — print the board.
- `eval` — print the static evaluation.

`./chess_engine bench [depth]` runs bench non-interactively and exits.

## What's implemented

**Search** — iterative deepening, aspiration windows, PVS / NegaScout,
adaptive null-move pruning, late move reductions, futility pruning,
reverse futility, razoring, check extension, mate-distance pruning,
quiescence with delta pruning and SEE-pruned bad captures.

**Move ordering** — TT move, MVV-LVA captures with SEE-based bad-capture
demotion, killers (2 slots/ply), history heuristic, queen promotion
prioritization. Countermove heuristic infrastructure is present but not
currently fed (kept simple in this initial release).

**Evaluation** — tapered PeSTO MG/EG PSTs, bishop pair, pawn structure
(doubled / isolated / passed / connected passers), rook on open and
semi-open files, king pawn shield, simple mobility, tempo bonus.

**Board** — bitboards (12 piece BBs + occupancy), magic-bitboard sliding
attacks (init-time random magic search, plain layout), full make/unmake
with state stack, Zobrist incremental key.

**TT** — depth-preferred replacement with aging, ply-adjusted mate
scores, configurable size via `setoption name Hash`.

## Limitations

- **Polyglot opening book** is stubbed. A real reader needs the 781-entry
  Polyglot Random64 table; see TODO in `src/book.cpp`. The Python sister
  reader (`chess_engine/book.py`) is fully functional in the meantime.
- **Single-threaded.** Multi-threaded search (Lazy SMP) is the single
  largest realistic strength upgrade after this baseline.
- **No NNUE.** Hand-tuned PSTs only.
- **No tablebase probing.**

## Layout

```
src/
├── types.h         enums, Move encoding, constants, popcount/lsb
├── board.h/cpp     bitboard position, FEN, make/unmake, attackers, SEE
├── movegen.h/cpp   magic init, attack lookups, pseudo-legal generation, perft
├── evaluate.h/cpp  tapered eval
├── tt.h/cpp        Zobrist, transposition table
├── search.h/cpp    iterative deepening + alpha-beta
├── book.h/cpp      polyglot stub
└── main.cpp        UCI loop, perft / bench / divide commands
```
