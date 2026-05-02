"""Reproduce the long-game disconnect by giving the engine very small remaining
time, emulating what happens late in a Cute Chess game when the side has
spent most of its clock. Sends a 130-ply position and a `go wtime 5 btime 5`,
then waits 3 seconds for a bestmove. Pre-fix the engine returns NO bestmove
because the time-budget computation underflows and the search runs forever.
"""
from __future__ import annotations
import pathlib, random, subprocess, sys, threading, time
import chess

ROOT   = pathlib.Path(__file__).resolve().parent.parent
ENGINE = ROOT / "chess_engine.exe"


def build_long_sequence(plies: int) -> tuple[list[str], chess.Board]:
    rng = random.Random(7)
    board = chess.Board()
    moves: list[str] = []
    for _ in range(plies):
        legal = list(board.legal_moves)
        if not legal or board.is_game_over(claim_draw=False):
            break
        non_term = []
        for m in legal:
            board.push(m)
            term = board.is_game_over(claim_draw=False) or not list(board.legal_moves)
            board.pop()
            if not term:
                non_term.append(m)
        cands = non_term or legal
        quiet = [m for m in cands if not board.is_capture(m) and m.promotion is None]
        chosen = rng.choice(quiet) if quiet else rng.choice(cands)
        moves.append(chosen.uci())
        board.push(chosen)
    return moves, board


def run(commands: list[str], wait_s: float) -> tuple[list[str], list[str]]:
    proc = subprocess.Popen(
        [str(ENGINE)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, cwd=str(ROOT),
    )
    out: list[str] = []
    err: list[str] = []

    def reader(stream, sink):
        for line in stream:
            sink.append(line.rstrip("\n"))

    th_o = threading.Thread(target=reader, args=(proc.stdout, out), daemon=True)
    th_e = threading.Thread(target=reader, args=(proc.stderr, err), daemon=True)
    th_o.start(); th_e.start()

    assert proc.stdin
    for c in commands:
        proc.stdin.write(c + "\n")
        proc.stdin.flush()

    deadline = time.time() + wait_s
    while time.time() < deadline:
        if any(l.startswith("bestmove") for l in out):
            break
        time.sleep(0.05)
    proc.stdin.write("quit\n"); proc.stdin.flush(); proc.stdin.close()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill(); proc.wait(timeout=2)
    return out, err


def main() -> int:
    moves, fb = build_long_sequence(130)
    print(f"generated {len(moves)} plies; final FEN: {fb.fen()}")
    print(f"final-position legal moves: {len(list(fb.legal_moves))}")

    cmds = [
        "uci",
        "isready",
        "position startpos moves " + " ".join(moves),
        "go wtime 5 btime 5 winc 0 binc 0",
    ]
    out, err = run(cmds, wait_s=3.0)
    bm = next((l for l in out if l.startswith("bestmove")), None)
    if bm is None:
        print("REPRO: no bestmove within 3s -- engine hangs (would disconnect in CuteChess)")
        print("--- last 20 stderr lines ---")
        for l in err[-20:]: print(l)
        return 1
    print(f"OK: engine emitted {bm}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
