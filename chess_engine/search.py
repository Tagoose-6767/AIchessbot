"""Iterative-deepening alpha-beta search with PVS and modern reductions.

Architecture (negamax — every score is from the side-to-move's perspective):

  iterative_deepening
    aspiration windows around previous score
    negamax
      mate-distance pruning
      check extension (depth + 1)
      transposition table probe (cutoff for non-PV nodes)
      reverse-futility / static null-move pruning
      razoring (drop to qsearch at very low depth)
      null-move pruning (skip a turn; if still ≥ beta, prune)
      move ordering (TT, captures w/ MVV-LVA + SEE, killers, history, countermove)
      futility pruning of late quiet moves at low depth
      late-move reductions (LMR) on quiet, late, non-checking moves
      principal variation search (PVS / NegaScout) — null window first, re-search
      transposition table store
    quiescence search at horizon
      stand-pat
      delta pruning
      SEE-based bad-capture pruning
"""

import math
import time

import chess

from evaluate import evaluate
from movegen import order_moves, order_captures, see
from transposition import (
    EXACT, LOWER, UPPER,
    TranspositionTable, zobrist, score_to_tt, score_from_tt,
)
from utils import (
    INF, MATE_SCORE, MATE_IN_MAX, MAX_PLY,
    PIECE_VALUES_SIMPLE,
)


# --- Tunables -----------------------------------------------------------------
NULL_MOVE_BASE_R = 2
RAZOR_MARGIN = [0, 250, 400]
FUTILITY_MARGIN = [0, 150, 300, 450]
LMR_MIN_DEPTH = 3
LMR_MIN_MOVE_INDEX = 4
ASPIRATION_DELTA = 25
DELTA_MARGIN = 200  # qsearch delta-pruning slack


class Search:
    def __init__(self, tt=None):
        self.tt = tt or TranspositionTable(128)
        self.nodes = 0
        self.q_nodes = 0
        self.start_time = 0.0
        self.time_limit = None
        self.stop_flag = False
        self.seldepth = 0

        self.killers = [[None, None] for _ in range(MAX_PLY)]
        self.history = {}      # (color, from, to) -> score
        self.countermove = {}  # (prev_from, prev_to) -> move

        self.pv_table = [[None] * MAX_PLY for _ in range(MAX_PLY)]
        self.pv_length = [0] * MAX_PLY

    # --- Time / stop ----------------------------------------------------------
    def stop_check(self):
        if self.stop_flag:
            return True
        if self.time_limit is not None and (time.time() - self.start_time) > self.time_limit:
            self.stop_flag = True
            return True
        return False

    # --- Iterative deepening driver ------------------------------------------
    def go(self, board, max_depth=64, time_limit=None, info_cb=None):
        self.start_time = time.time()
        self.time_limit = time_limit
        self.stop_flag = False
        self.nodes = 0
        self.q_nodes = 0
        self.tt.new_search()
        self.killers = [[None, None] for _ in range(MAX_PLY)]
        self.history = {}
        self.countermove = {}

        best_move = None
        score = 0

        for depth in range(1, max_depth + 1):
            self.seldepth = 0

            # Aspiration windows: assume next score is close to last; widen on fail.
            if depth >= 4 and not _is_mate(score):
                alpha = score - ASPIRATION_DELTA
                beta = score + ASPIRATION_DELTA
            else:
                alpha, beta = -INF, INF

            delta = ASPIRATION_DELTA
            v = score
            while True:
                self.pv_length = [0] * MAX_PLY
                v = self._negamax(board, depth, 0, alpha, beta, True)
                if self.stop_flag:
                    break
                if v <= alpha:
                    alpha = max(-INF, alpha - delta)
                    delta *= 2
                elif v >= beta:
                    beta = min(INF, beta + delta)
                    delta *= 2
                else:
                    break

            # Don't accept partial results from a stopped iteration.
            if self.stop_flag and depth > 1:
                break
            score = v

            pv = [m for m in self.pv_table[0][:self.pv_length[0]] if m is not None]
            if pv:
                best_move = pv[0]

            elapsed = max(1e-6, time.time() - self.start_time)
            nps = int(self.nodes / elapsed)
            if info_cb is not None:
                info_cb(depth, self.seldepth, score, self.nodes, nps,
                        int(elapsed * 1000), pv)

            # Stop early if we've found a forced mate.
            if _is_mate(score):
                break

        return best_move, score

    # --- Main negamax ---------------------------------------------------------
    def _negamax(self, board, depth, ply, alpha, beta, allow_null):
        # Default: no PV from this node yet.
        self.pv_length[ply] = 0

        # Periodic stop check.
        if (self.nodes & 2047) == 0 and self.stop_check():
            return 0
        if ply >= MAX_PLY - 1:
            return evaluate(board)

        # Mate-distance pruning: shrink the window if a mate has been seen.
        alpha = max(alpha, -MATE_SCORE + ply)
        beta = min(beta, MATE_SCORE - ply - 1)
        if alpha >= beta:
            return alpha

        in_check = board.is_check()
        # Check extension: never drop into qsearch while in check.
        if in_check:
            depth += 1

        if depth <= 0:
            return self._quiescence(board, ply, alpha, beta)

        self.nodes += 1
        if ply > self.seldepth:
            self.seldepth = ply

        # Draw detection.
        if ply > 0 and (board.is_repetition(2)
                        or board.halfmove_clock >= 100
                        or board.is_insufficient_material()):
            return 0

        is_pv = (beta - alpha) > 1

        # --- Transposition table probe ---
        key = zobrist(board)
        tt_move = None
        tt_entry = self.tt.probe(key)
        if tt_entry is not None:
            tt_move = tt_entry.move
            if tt_entry.depth >= depth and ply > 0 and not is_pv:
                s = score_from_tt(tt_entry.score, ply)
                if tt_entry.flag == EXACT:
                    return s
                if tt_entry.flag == LOWER and s >= beta:
                    return s
                if tt_entry.flag == UPPER and s <= alpha:
                    return s

        static_eval = evaluate(board) if not in_check else 0

        # --- Pre-move pruning (skip when in check or in a PV node) ---
        if not in_check and not is_pv:
            # Razoring: at very low depth, if static eval is well below alpha,
            # the only way to escape is a tactic — drop straight to qsearch.
            if depth <= 2 and static_eval + RAZOR_MARGIN[depth] <= alpha:
                q = self._quiescence(board, ply, alpha, beta)
                if q <= alpha:
                    return q

            # Reverse futility / static null-move: if static eval is way above
            # beta we expect any sensible move to keep us above too.
            if depth <= 3 and static_eval - FUTILITY_MARGIN[depth] >= beta and abs(beta) < MATE_IN_MAX:
                return static_eval

            # Null-move pruning: pass the move and see if opponent can still
            # not bring us below beta. Skipped in zugzwang-prone endings, hence
            # the non-pawn-material guard.
            if (allow_null and depth >= 3
                    and static_eval >= beta
                    and self._has_non_pawn_material(board)):
                R = NULL_MOVE_BASE_R + (depth // 6) + 1  # adaptive R = 3..
                board.push(chess.Move.null())
                v = -self._negamax(board, depth - 1 - R, ply + 1,
                                   -beta, -beta + 1, False)
                board.pop()
                if self.stop_flag:
                    return 0
                if v >= beta and abs(v) < MATE_IN_MAX:
                    return v

        # --- Move generation & ordering ---
        moves = list(board.legal_moves)
        if not moves:
            return -MATE_SCORE + ply if in_check else 0

        prev_move = board.peek() if board.move_stack else None
        prev_key = (prev_move.from_square, prev_move.to_square) if prev_move else None
        ordered = order_moves(board, moves, tt_move,
                              self.killers[ply], self.history,
                              self.countermove.get(prev_key))

        # Frontier futility: skip late quiet moves at low depth that almost
        # certainly can't raise alpha.
        do_futility = (not in_check and not is_pv and depth <= 3
                       and static_eval + FUTILITY_MARGIN[depth] <= alpha
                       and abs(alpha) < MATE_IN_MAX)

        best_score = -INF
        best_move = None
        flag = UPPER
        moves_searched = 0

        for move in ordered:
            is_capture = board.is_capture(move)
            gives_check = board.gives_check(move)
            is_quiet = (not is_capture and move.promotion is None
                        and not gives_check)

            if do_futility and is_quiet and moves_searched > 0:
                continue

            board.push(move)
            new_depth = depth - 1

            # Late move reduction: trust ordering for late quiet moves and
            # search them shallower with a null window. If they surprise us,
            # re-search at full depth.
            reduction = 0
            if (depth >= LMR_MIN_DEPTH
                    and moves_searched >= LMR_MIN_MOVE_INDEX
                    and is_quiet and not in_check):
                reduction = 1 + int(math.log(depth) * math.log(moves_searched + 1) / 2)
                reduction = min(reduction, new_depth - 1)
                if reduction < 0:
                    reduction = 0

            if moves_searched == 0:
                # First move: full window, full depth (PV).
                v = -self._negamax(board, new_depth, ply + 1, -beta, -alpha, True)
            else:
                # PVS: null-window probe, possibly reduced.
                v = -self._negamax(board, new_depth - reduction, ply + 1,
                                   -alpha - 1, -alpha, True)
                if v > alpha and reduction > 0:
                    v = -self._negamax(board, new_depth, ply + 1,
                                       -alpha - 1, -alpha, True)
                if v > alpha and v < beta:
                    # PV re-search at full window.
                    v = -self._negamax(board, new_depth, ply + 1,
                                       -beta, -alpha, True)

            board.pop()
            if self.stop_flag:
                return 0

            moves_searched += 1
            if v > best_score:
                best_score = v
                best_move = move
                if v > alpha:
                    alpha = v
                    flag = EXACT
                    # Update triangular PV table.
                    self.pv_table[ply][0] = move
                    child_len = self.pv_length[ply + 1]
                    for i in range(child_len):
                        self.pv_table[ply][i + 1] = self.pv_table[ply + 1][i]
                    self.pv_length[ply] = child_len + 1

                    if v >= beta:
                        flag = LOWER
                        # Reward this move for future ordering.
                        if is_quiet:
                            kl = self.killers[ply]
                            if kl[0] != move:
                                kl[1] = kl[0]
                                kl[0] = move
                            color = board.turn  # after pop, side to move = mover
                            hkey = (color, move.from_square, move.to_square)
                            self.history[hkey] = self.history.get(hkey, 0) + depth * depth
                            if prev_key is not None:
                                self.countermove[prev_key] = move
                        break

        if not self.stop_flag:
            self.tt.store(key, depth,
                          score_to_tt(best_score, ply),
                          flag, best_move)
        return best_score

    # --- Quiescence -----------------------------------------------------------
    def _quiescence(self, board, ply, alpha, beta):
        if (self.nodes & 2047) == 0 and self.stop_check():
            return 0
        self.nodes += 1
        self.q_nodes += 1
        if ply > self.seldepth:
            self.seldepth = ply

        in_check = board.is_check()
        if in_check:
            stand_pat = -INF
        else:
            stand_pat = evaluate(board)
            if stand_pat >= beta:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat

        if ply >= MAX_PLY - 1:
            return stand_pat if stand_pat != -INF else evaluate(board)

        # In check: search all evasions; otherwise only captures/promotions.
        if in_check:
            move_iter = board.legal_moves
        else:
            move_iter = order_captures(board)

        any_legal = False
        for move in move_iter:
            any_legal = True
            if not in_check:
                # SEE-based pruning: throw out captures that lose material.
                if see(board, move) < 0:
                    continue
                # Delta pruning: even capturing this victim (plus promo bonus)
                # plus a slack margin can't possibly raise alpha — skip.
                if board.is_en_passant(move):
                    vval = PIECE_VALUES_SIMPLE[chess.PAWN]
                else:
                    victim = board.piece_at(move.to_square)
                    vval = PIECE_VALUES_SIMPLE[victim.piece_type] if victim else 0
                if move.promotion:
                    vval += PIECE_VALUES_SIMPLE.get(move.promotion, 0) - PIECE_VALUES_SIMPLE[chess.PAWN]
                if stand_pat + vval + DELTA_MARGIN < alpha:
                    continue

            board.push(move)
            v = -self._quiescence(board, ply + 1, -beta, -alpha)
            board.pop()
            if self.stop_flag:
                return 0
            if v >= beta:
                return v
            if v > alpha:
                alpha = v

        if in_check and not any_legal:
            return -MATE_SCORE + ply
        return alpha

    # --- Helpers --------------------------------------------------------------
    @staticmethod
    def _has_non_pawn_material(board):
        c = board.turn
        own = board.occupied_co[c]
        return bool((board.knights | board.bishops | board.rooks | board.queens) & own)


def _is_mate(score):
    return abs(score) >= MATE_IN_MAX
