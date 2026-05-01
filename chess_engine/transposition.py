"""Transposition table with Zobrist keying.

We reuse python-chess's polyglot Zobrist hash rather than maintain incremental
hashing ourselves. That's a fair tradeoff in pure Python where move-gen
already dominates the cost.

The TT uses a two-bucket scheme per index: a "depth-preferred" slot that
keeps the deepest important result, plus an "always-replace" fallback so that
shallow probes during search never evict our best stored work.

Mate scores are stored relative to root (depth-from-mate) by adjusting in
score_to_tt / score_from_tt, so that a TT hit from a different ply doesn't
report misleading "mate in N".
"""

import chess
import chess.polyglot

from utils import MATE_IN_MAX

# Bound flags for stored scores.
EXACT = 0   # alpha < score < beta (PV node)
LOWER = 1   # beta cutoff happened — score is a lower bound
UPPER = 2   # failed low — score is an upper bound


class TTEntry:
    __slots__ = ("key", "depth", "score", "flag", "move", "age")

    def __init__(self, key=0, depth=0, score=0, flag=EXACT, move=None, age=0):
        self.key = key
        self.depth = depth
        self.score = score
        self.flag = flag
        self.move = move
        self.age = age


class TranspositionTable:
    def __init__(self, mb_size: int = 128):
        # In Python a TT entry is far heavier than a C struct — the advertised
        # MB size is treated as a hint. We pick a power-of-two entry count for
        # cheap masking.
        approx_bytes_per_pair = 200
        n_entries = max(2, (mb_size * 1024 * 1024) // approx_bytes_per_pair)
        n = 1
        while (n << 1) <= n_entries:
            n <<= 1
        self.size = n
        self.mask = n - 1
        self.deep = [None] * n
        self.always = [None] * n
        self.age = 0

    def new_search(self):
        self.age = (self.age + 1) & 0xFF

    def probe(self, key: int):
        idx = key & self.mask
        e = self.deep[idx]
        if e is not None and e.key == key:
            return e
        e = self.always[idx]
        if e is not None and e.key == key:
            return e
        return None

    def store(self, key, depth, score, flag, move):
        idx = key & self.mask
        deep = self.deep[idx]
        # Depth-preferred replacement: a deeper or fresher entry wins; otherwise
        # bounce to the always-replace slot so we don't lose the deep hit.
        if deep is None or deep.age != self.age or depth >= deep.depth:
            self.deep[idx] = TTEntry(key, depth, score, flag, move, self.age)
        else:
            self.always[idx] = TTEntry(key, depth, score, flag, move, self.age)

    def clear(self):
        self.deep = [None] * self.size
        self.always = [None] * self.size


def zobrist(board: chess.Board) -> int:
    return chess.polyglot.zobrist_hash(board)


def score_to_tt(score, ply):
    """Adjust a mate score so the stored value is independent of ply."""
    if score >= MATE_IN_MAX:
        return score + ply
    if score <= -MATE_IN_MAX:
        return score - ply
    return score


def score_from_tt(score, ply):
    """Inverse of score_to_tt: re-anchor a stored mate score to current ply."""
    if score >= MATE_IN_MAX:
        return score - ply
    if score <= -MATE_IN_MAX:
        return score + ply
    return score
