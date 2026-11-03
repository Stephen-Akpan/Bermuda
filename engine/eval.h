#pragma once

#include "position.h"
#include <string>

namespace Eval
{

struct Parameters {
    double material = 1.0;
    double piece_square = 1.0;
    double pawn_structure = 1.0;
    double passed_pawns = 1.0;
    double mobility = 1.0;
    double activity = 1.0;
    double threats = 1.0;
    double king_safety = 1.0;
    double tempo = 1.0;
    double pawn_mg = 1.0;
    double knight_mg = 1.0;
    double bishop_mg = 1.0;
    double rook_mg = 1.0;
    double queen_mg = 1.0;
    double pawn_eg = 1.0;
    double knight_eg = 1.0;
    double bishop_eg = 1.0;
    double rook_eg = 1.0;
    double queen_eg = 1.0;
};

// ============================================================================
// Evaluation
// ============================================================================

// Evaluate the position with a tapered handcrafted evaluation.
// Returns centipawns from the side-to-move's perspective.
Score evaluate(const Position& pos);
Score evaluate(const Position& pos, const Parameters& parameters);

bool load_parameters(const std::string& path, std::string& error);
bool load_parameters(const std::string& path, Parameters& parameters,
                     std::string& error);
bool save_parameters(const std::string& path, const Parameters& parameters,
                     std::string& error);
void reset_parameters();
Parameters get_parameters();

} // namespace Eval
