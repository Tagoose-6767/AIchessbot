"""Perft (move-generation correctness) test suite.

Compares move counts against known reference values for the standard
positions. Since we delegate move generation to python-chess, these passing
mainly validates that we haven't broken anything in our wrapping of it.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chess  # noqa: E402

from engine import perft  # noqa: E402


SUITE = [
    # (FEN, depth, expected nodes)
    (chess.STARTING_FEN, 1, 20),
    (chess.STARTING_FEN, 2, 400),
    (chess.STARTING_FEN, 3, 8902),
    (chess.STARTING_FEN, 4, 197281),
    (chess.STARTING_FEN, 5, 4865609),
    # "Kiwipete" — popular tactical perft position.
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
     4, 4085603),
    # Position 3 (endgame, lots of edge-case checks).
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674624),
]


def main():
    failed = 0
    for fen, depth, expected in SUITE:
        b = chess.Board(fen)
        t0 = time.time()
        n = perft(b, depth)
        dt = time.time() - t0
        ok = n == expected
        status = "OK" if ok else "FAIL"
        short = fen if len(fen) <= 40 else fen[:37] + "..."
        print(f"perft({depth}) {short:42s}  got {n:>10d}  expected {expected:>10d}  "
              f"{status}  [{dt:.2f}s]")
        if not ok:
            failed += 1
    sys.exit(failed)


if __name__ == "__main__":
    main()
