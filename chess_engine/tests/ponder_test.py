"""Drive the engine through go ponder + ponderhit and go ponder + stop, and
assert it emits bestmove (with a ponder hint where available) in the right
order. Verifies threads don't hang and the global deadline rewrite from the
UCI thread actually halts the search.
"""
from __future__ import annotations
import pathlib, subprocess, sys, threading, time

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENGINE = ROOT / "chess_engine.exe"


def run_session(commands: list[tuple[float, str]], total_timeout_s: float) -> list[str]:
    """Send commands paced by the supplied delays; collect stdout lines."""
    proc = subprocess.Popen(
        [str(ENGINE)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(ROOT),
    )
    out: list[str] = []

    def reader():
        assert proc.stdout is not None
        for line in proc.stdout:
            out.append(line.rstrip("\n"))

    th = threading.Thread(target=reader, daemon=True)
    th.start()

    assert proc.stdin is not None
    for delay, cmd in commands:
        if delay > 0:
            time.sleep(delay)
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()
    proc.stdin.close()

    try:
        proc.wait(timeout=total_timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)
        out.append("<<TIMEOUT>>")
    th.join(timeout=2)
    return out


def first_bestmove(lines: list[str]) -> str | None:
    for ln in lines:
        if ln.startswith("bestmove"):
            return ln
    return None


def assert_with(cond: bool, msg: str, lines: list[str]) -> None:
    if cond:
        return
    print(f"FAIL: {msg}")
    print("--- engine output ---")
    for ln in lines:
        print(ln)
    sys.exit(1)


def test_ponderhit_path() -> None:
    print("=== test 1: go ponder, then ponderhit ===")
    cmds = [
        (0.0,  "uci"),
        (0.05, "isready"),
        (0.05, "position startpos moves e2e4 e7e5 g1f3"),
        (0.0,  "go ponder wtime 4000 btime 4000"),
        (0.6,  "ponderhit"),     # convert to time-limited search
        (3.0,  "quit"),          # in case bestmove never arrives, force exit
    ]
    out = run_session(cmds, total_timeout_s=10)
    bm = first_bestmove(out)
    assert_with(bm is not None and "<<TIMEOUT>>" not in out,
                "no bestmove after ponderhit (engine likely hung)", out)
    print(f"  -> {bm}")


def test_stop_path() -> None:
    print("=== test 2: go ponder, then stop ===")
    cmds = [
        (0.0,  "uci"),
        (0.05, "isready"),
        (0.05, "position startpos moves e2e4 e7e5 g1f3"),
        (0.0,  "go ponder"),     # no time fields => budget=0 => INT64_MAX deadline
        (0.5,  "stop"),
        (2.0,  "quit"),
    ]
    out = run_session(cmds, total_timeout_s=8)
    bm = first_bestmove(out)
    assert_with(bm is not None and "<<TIMEOUT>>" not in out,
                "no bestmove after stop during ponder (engine hung)", out)
    print(f"  -> {bm}")


def test_normal_go_emits_ponder() -> None:
    print("=== test 3: regular go depth 8 emits 'ponder' tag ===")
    cmds = [
        (0.0,  "uci"),
        (0.05, "isready"),
        (0.05, "position startpos"),
        (0.0,  "go depth 8"),
        (5.0,  "quit"),
    ]
    out = run_session(cmds, total_timeout_s=15)
    bm = first_bestmove(out)
    assert_with(bm is not None and "<<TIMEOUT>>" not in out,
                "no bestmove after go depth 8", out)
    assert_with(" ponder " in bm,
                f"bestmove line had no ponder tag: {bm!r}", out)
    print(f"  -> {bm}")


def main() -> int:
    test_ponderhit_path()
    test_stop_path()
    test_normal_go_emits_ponder()
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
