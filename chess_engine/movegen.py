"""Move ordering and Static Exchange Evaluation (SEE).

Good move ordering is the single biggest factor in alpha-beta efficiency:
fail-high on the first move and you prune the rest of the subtree. Order:
  1. TT (hash) move
  2. Winning/equal captures by MVV-LVA, with bad captures (SEE < 0) demoted
  3. Queen promotions
  4. Killer moves (2 slots per ply)
  5. Countermove (refutation of opponent's last move)
  6. History heuristic (depth^2 weighted) for remaining quiets
"""

import chess

from utils import MVV_LVA, PIECE_VALUES_SIMPLE


def order_moves(board, moves, tt_move, killers, history, counter):
    scored = []
    for m in moves:
        scored.append((_score_move(board, m, tt_move, killers, history, counter), m))
    scored.sort(key=lambda x: -x[0])
    return [m for _, m in scored]


def _score_move(board, move, tt_move, killers, history, counter):
    if tt_move is not None and move == tt_move:
        return 1_000_000
    captured = _captured_piece_type(board, move)
    if captured is not None:
        attacker = board.piece_type_at(move.from_square)
        s = 800_000 + MVV_LVA[captured][attacker]
        # Bad captures (losing material per SEE) are demoted below killers/quiet
        # history but still ahead of nothing — sometimes they're tactically
        # forced (e.g. desperado).
        if see(board, move) < 0:
            s -= 600_000
        return s
    if move.promotion == chess.QUEEN:
        return 900_000
    if move.promotion:
        return 750_000
    if killers[0] is not None and move == killers[0]:
        return 700_000
    if killers[1] is not None and move == killers[1]:
        return 690_000
    if counter is not None and move == counter:
        return 600_000
    color = board.turn
    return history.get((color, move.from_square, move.to_square), 0)


def _captured_piece_type(board, move):
    if board.is_en_passant(move):
        return chess.PAWN
    return board.piece_type_at(move.to_square)


def order_captures(board):
    """Yield captures (and queen promotions) in MVV-LVA order for qsearch."""
    caps = []
    for m in board.generate_legal_captures():
        victim = _captured_piece_type(board, m) or chess.PAWN
        attacker = board.piece_type_at(m.from_square)
        s = MVV_LVA[victim][attacker]
        if m.promotion:
            s += PIECE_VALUES_SIMPLE.get(m.promotion, 0)
        caps.append((s, m))
    # Also include non-capturing queen promotions (still tactical).
    for m in board.legal_moves:
        if m.promotion == chess.QUEEN and not board.is_capture(m):
            caps.append((PIECE_VALUES_SIMPLE[chess.QUEEN], m))
    caps.sort(key=lambda x: -x[0])
    for _, m in caps:
        yield m


# --- Static Exchange Evaluation -----------------------------------------------
# Standard "swap" algorithm: simulate alternating recaptures by the least
# valuable attacker on each side, then unwind the gain stack with a min/max
# choice (each side may stop capturing if it would lose material).

def see(board, move):
    to_sq = move.to_square
    from_sq = move.from_square
    side = board.turn

    if board.is_en_passant(move):
        gain0 = PIECE_VALUES_SIMPLE[chess.PAWN]
    else:
        target = board.piece_at(to_sq)
        gain0 = PIECE_VALUES_SIMPLE[target.piece_type] if target else 0

    attacker_piece = board.piece_type_at(from_sq)
    occ = board.occupied & ~chess.BB_SQUARES[from_sq]
    if board.is_en_passant(move):
        ep_sq = to_sq + (-8 if side == chess.WHITE else 8)
        occ &= ~chess.BB_SQUARES[ep_sq]

    side = not side
    gain = [gain0]
    d = 0

    while True:
        d += 1
        gain.append(PIECE_VALUES_SIMPLE[attacker_piece] - gain[d - 1])
        # Speculative pruning: even if the recapture chain continues
        # optimally, we already know it's losing here — bail out.
        if max(-gain[d - 1], gain[d]) < 0:
            break
        attacker_sq, attacker_piece = _least_valuable_attacker(board, to_sq, side, occ)
        if attacker_sq is None:
            break
        occ &= ~chess.BB_SQUARES[attacker_sq]
        side = not side

    while d > 0:
        gain[d - 1] = -max(-gain[d - 1], gain[d])
        d -= 1
    return gain[0]


def _least_valuable_attacker(board, sq, side, occ):
    # Filter by our custom occupancy too — `attackers_mask` ANDs with the
    # board's actual color mask, so without this a piece we already "moved"
    # could still be returned as an attacker.
    attackers = board.attackers_mask(side, sq, occ) & occ
    if not attackers:
        return None, None
    for piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP,
                       chess.ROOK, chess.QUEEN, chess.KING):
        bb = attackers & board.pieces_mask(piece_type, side)
        if bb:
            s = (bb & -bb).bit_length() - 1
            return s, piece_type
    return None, None
