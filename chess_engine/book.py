"""Polyglot opening book reader (thin wrapper over python-chess).

Polyglot books store positions by Zobrist hash and a list of moves with
weights. We pick a move proportionally to its weight so games don't all
collapse onto the same line.
"""

import os
import random

import chess
import chess.polyglot


class OpeningBook:
    def __init__(self, path=None):
        self.path = path
        self.reader = None
        if path and os.path.exists(path):
            try:
                self.reader = chess.polyglot.open_reader(path)
            except Exception:
                self.reader = None

    def find_move(self, board, weighted=True):
        if self.reader is None:
            return None
        try:
            entries = list(self.reader.find_all(board))
        except Exception:
            return None
        if not entries:
            return None
        if not weighted:
            return entries[0].move
        total = sum(e.weight for e in entries)
        if total <= 0:
            return entries[0].move
        pick = random.randint(1, total)
        running = 0
        for e in entries:
            running += e.weight
            if running >= pick:
                return e.move
        return entries[0].move

    def close(self):
        if self.reader is not None:
            self.reader.close()
            self.reader = None
