// evaluate.h — tapered static evaluation.
#pragma once

#include "types.h"
class Board;

// Score is centipawns from the side-to-move's perspective.
int evaluate(const Board& b);
