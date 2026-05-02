"""Smoke test: drive the engine through a 150-ply game via UCI and confirm
it still answers `go depth 6` at the end. Pre-fix, the state_stack overflow
made this hang or crash around move ~80. Generates a deterministic legal
sequence using python-chess and uses python-chess's UCI engine wrapper so we
wait for `bestmove` properly instead of racing it with `quit`.
"""
from __future__ import annotations
import sys, random, chess, chess.engine, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENGINE = ROOT / "chess_engine.exe"
PLIES = 400  # ~200 moves; well past anything CuteChess will reach
SEARCH_DEPTH = 6
MOVE_TIMEOUT_S = 30.0


def build_long_sequence(plies: int) -> tuple[list[str], chess.Board]:
    rng = random.Random(42)
    board = chess.Board()
    moves: list[str] = []
    for _ in range(plies):
        legal = list(board.legal_moves)
        if not legal or board.is_game_over(claim_draw=False):
            break
        # Prefer non-captures and non-promotions to keep the game long, and
        # skip moves that would terminate the game on the very next ply so the
        # final position still has legal replies for the engine to choose from.
        non_term = []
        for m in legal:
            board.push(m)
            terminal_after = board.is_game_over(claim_draw=False) or not list(board.legal_moves)
            board.pop()
            if not terminal_after:
                non_term.append(m)
        candidates = non_term or legal
        quiet = [m for m in candidates
                 if not board.is_capture(m) and m.promotion is None]
        chosen = rng.choice(quiet) if quiet else rng.choice(candidates)
        moves.append(chosen.uci())
        board.push(chosen)
    return moves, board


def main() -> int:
    moves, final_board = build_long_sequence(PLIES)
    print(f"generated {len(moves)} plies of legal play")
    print(f"final FEN: {final_board.fen()}")
    print(f"final-position legal moves: {len(list(final_board.legal_moves))}")

    engine = chess.engine.SimpleEngine.popen_uci(
        str(ENGINE), timeout=MOVE_TIMEOUT_S
    )
    try:
        result = engine.play(
            final_board,
            chess.engine.Limit(depth=SEARCH_DEPTH),
        )
    finally:
        engine.quit()

    if result.move is None:
        print("FAIL: engine returned no move")
        return 1
    if result.move not in final_board.legal_moves:
        print(f"FAIL: engine returned illegal move {result.move.uci()}")
        return 1
    print(f"PASS: engine answered bestmove {result.move.uci()} after {len(moves)} plies")
    return 0


if __name__ == "__main__":
    sys.exit(main())
