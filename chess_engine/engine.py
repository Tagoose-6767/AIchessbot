"""UCI loop. All UCI traffic goes to stdout; debug/log lines go to stderr.

Run from inside this directory:
    python engine.py

Or load it in a UCI GUI (Arena, Cute Chess, Banksia) by pointing the GUI at
the same `python engine.py` invocation as the engine binary.
"""

import os
import sys
import threading

import chess

from book import OpeningBook
from search import Search
from transposition import TranspositionTable
from utils import score_to_uci


class UCIEngine:
    NAME = "AIChessbot 0.1"
    AUTHOR = "AIChessbot"

    def __init__(self):
        self.board = chess.Board()
        self.tt = TranspositionTable(128)
        self.search = Search(self.tt)
        self.book = OpeningBook(os.environ.get("OPENING_BOOK"))
        self.search_thread = None

    # --- Main loop ------------------------------------------------------------
    def loop(self):
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                self.handle(line)
            except SystemExit:
                raise
            except Exception as e:
                # Never let an exception kill the engine; report to stderr only.
                print(f"info string error: {e}", file=sys.stderr, flush=True)

    def handle(self, line):
        parts = line.split()
        cmd = parts[0]
        if cmd == "uci":
            self._send(f"id name {self.NAME}")
            self._send(f"id author {self.AUTHOR}")
            self._send("option name Hash type spin default 128 min 1 max 4096")
            self._send("option name Book type string default ")
            self._send("uciok")
        elif cmd == "isready":
            self._wait_search()
            self._send("readyok")
        elif cmd == "ucinewgame":
            self._wait_search()
            self.tt.clear()
            self.board = chess.Board()
        elif cmd == "setoption":
            self._set_option(parts)
        elif cmd == "position":
            self._wait_search()
            self._handle_position(parts)
        elif cmd == "go":
            self._handle_go(parts)
        elif cmd == "stop":
            self.search.stop_flag = True
            self._wait_search()
        elif cmd == "quit":
            self.search.stop_flag = True
            self._wait_search()
            if self.book:
                self.book.close()
            sys.exit(0)
        elif cmd == "perft":
            self._handle_perft(parts)
        elif cmd == "eval":
            from evaluate import evaluate
            self._send(f"info string eval {evaluate(self.board)}")
        # Unknown commands are silently ignored, per UCI spec.

    # --- Options --------------------------------------------------------------
    def _set_option(self, parts):
        try:
            ni = parts.index("name")
            vi = parts.index("value")
        except ValueError:
            return
        name = " ".join(parts[ni + 1:vi])
        value = " ".join(parts[vi + 1:])
        if name.lower() == "hash":
            try:
                mb = int(value)
            except ValueError:
                return
            self.tt = TranspositionTable(mb)
            self.search.tt = self.tt
        elif name.lower() == "book":
            if self.book:
                self.book.close()
            self.book = OpeningBook(value if value else None)

    # --- position -------------------------------------------------------------
    def _handle_position(self, parts):
        if "startpos" in parts:
            self.board = chess.Board()
            i = parts.index("startpos") + 1
        elif "fen" in parts:
            i = parts.index("fen") + 1
            fen_parts = []
            while i < len(parts) and parts[i] != "moves":
                fen_parts.append(parts[i])
                i += 1
            self.board = chess.Board(" ".join(fen_parts))
        else:
            return
        if i < len(parts) and parts[i] == "moves":
            for m in parts[i + 1:]:
                try:
                    self.board.push_uci(m)
                except ValueError:
                    print(f"info string illegal move in position: {m}",
                          file=sys.stderr, flush=True)
                    break

    # --- go -------------------------------------------------------------------
    def _handle_go(self, parts):
        movetime = None
        depth = 64
        wtime = btime = winc = binc = None
        movestogo = 30
        infinite = False

        i = 1
        while i < len(parts):
            tok = parts[i]
            if tok == "movetime" and i + 1 < len(parts):
                movetime = int(parts[i + 1]) / 1000.0; i += 2
            elif tok == "depth" and i + 1 < len(parts):
                depth = int(parts[i + 1]); i += 2
            elif tok == "wtime" and i + 1 < len(parts):
                wtime = int(parts[i + 1]) / 1000.0; i += 2
            elif tok == "btime" and i + 1 < len(parts):
                btime = int(parts[i + 1]) / 1000.0; i += 2
            elif tok == "winc" and i + 1 < len(parts):
                winc = int(parts[i + 1]) / 1000.0; i += 2
            elif tok == "binc" and i + 1 < len(parts):
                binc = int(parts[i + 1]) / 1000.0; i += 2
            elif tok == "movestogo" and i + 1 < len(parts):
                movestogo = max(1, int(parts[i + 1])); i += 2
            elif tok == "infinite":
                infinite = True; i += 1
            else:
                i += 1

        if infinite:
            time_limit = None
        elif movetime is not None:
            time_limit = movetime
        else:
            our_time = wtime if self.board.turn == chess.WHITE else btime
            our_inc = (winc if self.board.turn == chess.WHITE else binc) or 0
            if our_time is not None:
                time_limit = max(0.05, our_time / movestogo + our_inc * 0.7)
                # Hard cap at 1/4 of remaining clock for safety.
                time_limit = min(time_limit, our_time / 4)
            else:
                time_limit = None

        # Try the opening book first.
        if self.book is not None:
            bm = self.book.find_move(self.board)
            if bm is not None:
                self._send(f"bestmove {bm.uci()}")
                return

        # Run search in a thread so we can accept "stop" mid-flight.
        self._wait_search()
        self.search.stop_flag = False
        self.search_thread = threading.Thread(
            target=self._run_search, args=(depth, time_limit), daemon=True)
        self.search_thread.start()

    def _wait_search(self):
        t = self.search_thread
        if t is not None and t.is_alive():
            t.join()

    def _run_search(self, depth, time_limit):
        bm, _ = self.search.go(self.board, max_depth=depth,
                               time_limit=time_limit, info_cb=self._info)
        if bm is None:
            legal = list(self.board.legal_moves)
            bm = legal[0] if legal else None
        if bm is not None:
            self._send(f"bestmove {bm.uci()}")
        else:
            self._send("bestmove 0000")

    # --- info / send ----------------------------------------------------------
    def _info(self, depth, seldepth, score, nodes, nps, ms, pv):
        pv_str = " ".join(m.uci() for m in pv) if pv else ""
        self._send(
            f"info depth {depth} seldepth {seldepth} "
            f"score {score_to_uci(score)} nodes {nodes} nps {nps} "
            f"time {ms} pv {pv_str}".rstrip()
        )

    # --- debug perft ----------------------------------------------------------
    def _handle_perft(self, parts):
        depth = int(parts[1]) if len(parts) > 1 else 4
        n = perft(self.board, depth)
        self._send(f"info string perft({depth}) = {n}")

    def _send(self, s):
        print(s, flush=True)


def perft(board, depth):
    if depth == 0:
        return 1
    if depth == 1:
        return sum(1 for _ in board.legal_moves)
    n = 0
    for m in board.legal_moves:
        board.push(m)
        n += perft(board, depth - 1)
        board.pop()
    return n


def main():
    UCIEngine().loop()


if __name__ == "__main__":
    main()
