"""Engine vs engine self-play, output PGN.

Usage:
    python selfplay.py [n_games] [seconds_per_move]
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chess  # noqa: E402
import chess.pgn  # noqa: E402

from search import Search  # noqa: E402
from transposition import TranspositionTable  # noqa: E402


def play_one(white, black, time_per_move=1.0):
    board = chess.Board()
    while not board.is_game_over(claim_draw=True):
        engine = white if board.turn == chess.WHITE else black
        bm, _ = engine.go(board, max_depth=64, time_limit=time_per_move,
                          info_cb=lambda *a, **k: None)
        if bm is None:
            break
        board.push(bm)
    return board


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    tpm = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    games = []
    for i in range(n_games):
        # Fresh TTs per side so neither has stale data.
        w = Search(TranspositionTable(64))
        b = Search(TranspositionTable(64))
        print(f"Game {i + 1}/{n_games}...", file=sys.stderr)
        t0 = time.time()
        board = play_one(w, b, tpm)
        result = board.result(claim_draw=True)
        game = chess.pgn.Game.from_board(board)
        game.headers["Event"] = "AIChessbot self-play"
        game.headers["White"] = "AIChessbot"
        game.headers["Black"] = "AIChessbot"
        game.headers["Result"] = result
        games.append(game)
        print(f"  {result} in {time.time() - t0:.1f}s, "
              f"{len(board.move_stack)} moves", file=sys.stderr)

    out = "selfplay.pgn"
    with open(out, "w") as f:
        for g in games:
            print(g, file=f, end="\n\n")
    print(f"Wrote {n_games} games to {out}")


if __name__ == "__main__":
    main()
