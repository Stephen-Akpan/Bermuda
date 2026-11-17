#include "graph_encoder.h"

#include "bitboard.h"

namespace GNN {
namespace {

Bitboard attacks_from(PieceType type, Square square, Colour colour, Bitboard occupied) {
    switch (type) {
    case PAWN:   return BB::pawn_attacks(BB::square_bb(square), colour);
    case KNIGHT: return BB::knight_attack(square);
    case BISHOP: return BB::bishop_attack(square, occupied);
    case ROOK:   return BB::rook_attack(square, occupied);
    case QUEEN:  return BB::queen_attack(square, occupied);
    case KING:   return BB::king_attack(square);
    default:     return 0;
    }
}

} // namespace

void GraphEncoder::encode(const Position& position, GraphFeatures& output) const {
    output = GraphFeatures{};

    for (ui8 squareIndex = 0; squareIndex < SQUARE_COUNT; ++squareIndex) {
        const Square square = static_cast<Square>(squareIndex);
        const Piece piece = position.piece_on(square);
        if (piece == NO_PIECE) {
            output.squares[squareIndex][static_cast<ui8>(SquareFeature::EMPTY)] = 1.0f;
            continue;
        }

        const Colour colour = colour_of(piece);
        const PieceType type = type_of(piece);
        const ui8 identityFeature = static_cast<ui8>(SquareFeature::WHITE_PAWN)
            + (colour == BLACK ? 6 : 0) + static_cast<ui8>(type);
        output.squares[squareIndex][identityFeature] = 1.0f;

        if (type == KING) {
            if (colour == WHITE)
                output.white_king_square = square;
            else
                output.black_king_square = square;
        }
    }

    output.global[static_cast<ui8>(GlobalFeature::SIDE_TO_MOVE)] =
        position.side_to_move() == WHITE ? 1.0f : -1.0f;
    output.global[static_cast<ui8>(GlobalFeature::WHITE_KINGSIDE_CASTLING)] =
        static_cast<f32>(position.can_castle(WHITE_OO));
    output.global[static_cast<ui8>(GlobalFeature::WHITE_QUEENSIDE_CASTLING)] =
        static_cast<f32>(position.can_castle(WHITE_OOO));
    output.global[static_cast<ui8>(GlobalFeature::BLACK_KINGSIDE_CASTLING)] =
        static_cast<f32>(position.can_castle(BLACK_OO));
    output.global[static_cast<ui8>(GlobalFeature::BLACK_QUEENSIDE_CASTLING)] =
        static_cast<f32>(position.can_castle(BLACK_OOO));

    const Square epSquare = position.ep_square();
    if (epSquare != NO_SQUARE)
        output.squares[epSquare][static_cast<ui8>(SquareFeature::EN_PASSANT_TARGET)] = 1.0f;

    const Bitboard occupied = position.all_pieces();
    for (ui8 sourceIndex = 0; sourceIndex < SQUARE_COUNT; ++sourceIndex) {
        const Square source = static_cast<Square>(sourceIndex);
        const Piece piece = position.piece_on(source);
        if (piece == NO_PIECE)
            continue;

        const Colour colour = colour_of(piece);
        const Bitboard sourceBit = BB::square_bb(source);
        Bitboard attacks = attacks_from(type_of(piece), source, colour, occupied);
        const ui8 controlFeature = static_cast<ui8>(colour == WHITE
            ? SquareFeature::WHITE_CONTROL_COUNT
            : SquareFeature::BLACK_CONTROL_COUNT);

        while (attacks) {
            const Square target = static_cast<Square>(pop_lsb(attacks));
            output.squares[target][controlFeature] += 1.0f / 8.0f;

            const Piece targetPiece = position.piece_on(target);
            if (targetPiece == NO_PIECE)
                continue;

            if (colour_of(targetPiece) == colour)
                output.defence_sources[target] |= sourceBit;
            else
                output.attack_sources[target] |= sourceBit;
        }
    }
}

} // namespace GNN