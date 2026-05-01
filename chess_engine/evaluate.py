"""Tapered static evaluation.

We compute midgame and endgame scores in parallel, then linearly interpolate
using a phase number derived from non-pawn material. Score is returned from
the side-to-move's perspective (negamax convention).

Terms:
  - material + PST (tapered, PeSTO values)
  - bishop pair
  - pawn structure: doubled, isolated, passed (rank-bonus)
  - rook on open / semi-open file
  - simple king pawn-shield
"""

import chess

from utils import (
    PIECE_VALUES_MG, PIECE_VALUES_EG, PST_MG, PST_EG,
    PHASE_WEIGHTS, TOTAL_PHASE, MATE_SCORE,
)


# --- Precomputed masks --------------------------------------------------------
# For each (color, square): bitboard of squares "in front of and on adjacent
# files of" the pawn on `square`. Empty intersection with enemy pawns ⇒ passed.
_PASSED_MASKS = [[0] * 64, [0] * 64]
for sq in range(64):
    f = chess.square_file(sq)
    r = chess.square_rank(sq)
    files_mask = chess.BB_FILES[f]
    if f > 0:
        files_mask |= chess.BB_FILES[f - 1]
    if f < 7:
        files_mask |= chess.BB_FILES[f + 1]
    ahead_w = 0
    for rr in range(r + 1, 8):
        ahead_w |= chess.BB_RANKS[rr]
    _PASSED_MASKS[chess.WHITE][sq] = files_mask & ahead_w
    ahead_b = 0
    for rr in range(0, r):
        ahead_b |= chess.BB_RANKS[rr]
    _PASSED_MASKS[chess.BLACK][sq] = files_mask & ahead_b

_PASSED_RANK_BONUS_MG = [0, 5, 10, 20, 35, 60, 100, 0]
_PASSED_RANK_BONUS_EG = [0, 10, 20, 35, 60, 100, 150, 0]


def evaluate(board: chess.Board) -> int:
    """Centipawn score from the side-to-move's perspective."""
    if board.is_checkmate():
        return -MATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    mg_w = mg_b = eg_w = eg_b = 0
    phase = 0

    for piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP,
                       chess.ROOK, chess.QUEEN, chess.KING):
        pst_mg = PST_MG[piece_type]
        pst_eg = PST_EG[piece_type]
        pmg = PIECE_VALUES_MG[piece_type]
        peg = PIECE_VALUES_EG[piece_type]
        pw = PHASE_WEIGHTS[piece_type]

        bb = board.pieces_mask(piece_type, chess.WHITE)
        while bb:
            sq = (bb & -bb).bit_length() - 1
            bb &= bb - 1
            mg_w += pmg + pst_mg[sq]
            eg_w += peg + pst_eg[sq]
            phase += pw

        bb = board.pieces_mask(piece_type, chess.BLACK)
        while bb:
            sq = (bb & -bb).bit_length() - 1
            bb &= bb - 1
            # Black mirrors via XOR 56 (vertical flip).
            mg_b += pmg + pst_mg[sq ^ 56]
            eg_b += peg + pst_eg[sq ^ 56]
            phase += pw

    # Bishop pair (PeSTO already partly captures this in PSTs but the bonus
    # is well-known to be worth a separate explicit term).
    if chess.popcount(board.pieces_mask(chess.BISHOP, chess.WHITE)) >= 2:
        mg_w += 30
        eg_w += 50
    if chess.popcount(board.pieces_mask(chess.BISHOP, chess.BLACK)) >= 2:
        mg_b += 30
        eg_b += 50

    # Pawn structure.
    pmg, peg = _pawn_structure(board)
    mg_w += pmg[0]; mg_b += pmg[1]
    eg_w += peg[0]; eg_b += peg[1]

    # Rooks on open / semi-open files.
    rmg, reg = _rook_files(board)
    mg_w += rmg[0]; mg_b += rmg[1]
    eg_w += reg[0]; eg_b += reg[1]

    # King safety (midgame only — in the endgame an active king is good).
    mg_w += _king_safety(board, chess.WHITE)
    mg_b += _king_safety(board, chess.BLACK)

    mg_score = mg_w - mg_b
    eg_score = eg_w - eg_b

    if phase > TOTAL_PHASE:
        phase = TOTAL_PHASE
    score = (mg_score * phase + eg_score * (TOTAL_PHASE - phase)) // TOTAL_PHASE

    return score if board.turn == chess.WHITE else -score


def _pawn_structure(board):
    mg = [0, 0]
    eg = [0, 0]
    for color in (chess.WHITE, chess.BLACK):
        own = board.pieces_mask(chess.PAWN, color)
        opp = board.pieces_mask(chess.PAWN, not color)
        files = [chess.popcount(own & chess.BB_FILES[f]) for f in range(8)]

        bb = own
        while bb:
            sq = (bb & -bb).bit_length() - 1
            bb &= bb - 1
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            rel_rank = r if color == chess.WHITE else 7 - r

            # Doubled: more than one own pawn on the file (we deliberately
            # double-count a doubled pair; the magnitude is calibrated for it).
            if files[f] > 1:
                mg[color] -= 10
                eg[color] -= 20

            # Isolated: no own pawns on adjacent files.
            isolated = True
            if f > 0 and files[f - 1] > 0:
                isolated = False
            if f < 7 and files[f + 1] > 0:
                isolated = False
            if isolated:
                mg[color] -= 15
                eg[color] -= 10

            # Passed: nothing in the "front cone" can stop or contest us.
            if not (_PASSED_MASKS[color][sq] & opp):
                mg[color] += _PASSED_RANK_BONUS_MG[rel_rank]
                eg[color] += _PASSED_RANK_BONUS_EG[rel_rank]
                # Connected passer bonus: another own pawn on an adjacent file
                # within 1 rank.
                adj = 0
                if f > 0:
                    adj |= chess.BB_FILES[f - 1]
                if f < 7:
                    adj |= chess.BB_FILES[f + 1]
                near_ranks = chess.BB_RANKS[r]
                if r > 0:
                    near_ranks |= chess.BB_RANKS[r - 1]
                if r < 7:
                    near_ranks |= chess.BB_RANKS[r + 1]
                if own & adj & near_ranks:
                    mg[color] += 10
                    eg[color] += 20
    return mg, eg


def _rook_files(board):
    mg = [0, 0]
    eg = [0, 0]
    for color in (chess.WHITE, chess.BLACK):
        own_pawns = board.pieces_mask(chess.PAWN, color)
        opp_pawns = board.pieces_mask(chess.PAWN, not color)
        bb = board.pieces_mask(chess.ROOK, color)
        while bb:
            sq = (bb & -bb).bit_length() - 1
            bb &= bb - 1
            file_bb = chess.BB_FILES[chess.square_file(sq)]
            if not (file_bb & own_pawns):
                if not (file_bb & opp_pawns):
                    mg[color] += 25  # fully open file
                    eg[color] += 15
                else:
                    mg[color] += 12  # semi-open
                    eg[color] += 7
    return mg, eg


def _king_safety(board, color):
    """Pawn shield: penalise missing/distant pawns on the three king files."""
    king_sq = board.king(color)
    if king_sq is None:
        return 0
    f = chess.square_file(king_sq)
    own_pawns = board.pieces_mask(chess.PAWN, color)
    score = 0
    for ff in range(max(0, f - 1), min(7, f + 1) + 1):
        file_bb = chess.BB_FILES[ff] & own_pawns
        if not file_bb:
            score -= 25
            continue
        if color == chess.WHITE:
            # Closest white pawn on this file is the lowest set bit.
            pawn_sq = (file_bb & -file_bb).bit_length() - 1
            rel = chess.square_rank(pawn_sq) - chess.square_rank(king_sq)
        else:
            # Closest black pawn is the highest set bit.
            pawn_sq = file_bb.bit_length() - 1
            rel = chess.square_rank(king_sq) - chess.square_rank(pawn_sq)
        if rel == 1:
            score += 5
        elif rel == 2:
            score += 2
        elif rel <= 0:
            score -= 15
    return score
