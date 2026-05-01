"""Benchmark: measure NPS on a fixed middlegame position.

Note on the spec target of 1M NPS: that's not achievable in pure Python with
python-chess (typical range is 30k–200k NPS). For higher NPS, run under PyPy.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chess  # noqa: E402

from search import Search  # noqa: E402
from transposition import TranspositionTable  # noqa: E402

# Quiet-ish middlegame from a real game; commonly used for engine benchmarking.
FEN = "r1bq1rk1/pp2bppp/2n2n2/2pp4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 1"


def main():
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    board = chess.Board(FEN)
    s = Search(TranspositionTable(128))

    print(f"Position: {FEN}")
    print(f"Searching to depth {depth}...")

    last = {}

    def info(depth, seldepth, score, nodes, nps, ms, pv):
        last["depth"] = depth
        last["score"] = score
        last["nodes"] = nodes
        last["nps"] = nps
        last["ms"] = ms
        last["pv"] = pv
        print(f"  depth {depth:>2}  seldepth {seldepth:>2}  score {score:>6}  "
              f"nodes {nodes:>8}  nps {nps:>7}  time {ms}ms")

    t0 = time.time()
    bm, score = s.go(board, max_depth=depth, time_limit=None, info_cb=info)
    dt = time.time() - t0

    print()
    print(f"Best move: {bm.uci() if bm else None}  score: {score}")
    print(f"Total: {s.nodes} nodes in {dt:.2f}s  ({int(s.nodes / max(dt, 1e-6))} NPS)")
    print(f"Quiescence nodes: {s.q_nodes} ({100 * s.q_nodes / max(s.nodes, 1):.1f}%)")


if __name__ == "__main__":
    main()
