#include "perft.h"
#include "movegen.h"
#include <iostream>
#include <iomanip>

namespace Perft {

// ============================================================================
// Internal recursive implementation
// ============================================================================

static void perft_recursive(Position& pos, int depth, Result& result) {
    MoveList list;
    MoveGen::generate_moves(pos, list);

    for (int i = 0; i < list.size(); i++) {
        Move m = list[i];

        UndoInfo undo = pos.make_move(m);

        // Skip if move leaves our king in check (pseudo-legal filter)
        if (pos.is_attacked_by(pos.king_square(flip(pos.side_to_move())),
                               pos.side_to_move())) {
            pos.unmake_move(m, undo);
            continue;
        }

        if (depth == 1) {
            // Count this leaf node and its properties
            result.nodes++;

            if (undo.captured != NO_PIECE || m.is_en_passant())
                result.captures++;

            if (m.is_en_passant())
                result.ep++;

            if (m.is_castling())
                result.castles++;

            if (m.is_promotion())
                result.promotions++;

            if (pos.in_check())
                result.checks++;
        } else {
            perft_recursive(pos, depth - 1, result);
        }

        pos.unmake_move(m, undo);
    }
}

// ============================================================================
// Public Interface
// ============================================================================

Result perft(Position& pos, int depth) {
    Result result;

    if (depth == 0) {
        result.nodes = 1;
        return result;
    }

    perft_recursive(pos, depth, result);
    return result;
}

void divide(Position& pos, int depth) {
    MoveList list;
    MoveGen::generate_moves(pos, list);

    ui64 total = 0;

    std::cout << "\nDivide (depth " << depth << "):\n";

    for (int i = 0; i < list.size(); i++) {
        Move m = list[i];

        UndoInfo undo = pos.make_move(m);

        // Skip illegal moves
        if (pos.is_attacked_by(pos.king_square(flip(pos.side_to_move())),
                               pos.side_to_move())) {
            pos.unmake_move(m, undo);
            continue;
        }

        ui64 nodes = 0;

        if (depth == 1) {
            nodes = 1;
        } else {
            Result sub;
            perft_recursive(pos, depth - 1, sub);
            nodes = sub.nodes;
        }

        pos.unmake_move(m, undo);

        std::cout << m.to_string() << ": " << nodes << "\n";
        total += nodes;
    }

    std::cout << "\nTotal: " << total << "\n";
}

} // namespace Perft
