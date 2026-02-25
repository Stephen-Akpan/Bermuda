#include "catch_amalgamated.hpp"
#include "position.h"
#include "bitboard.h"

#include <iostream>

// Helper to initialize bitboard tables before tests that need them
inline void init_bb() {
  static bool initialized = false;
  if (!initialized) {
    BB::init();
    initialized = true;
  }
}

// ============================================================================
// Helpers
// ============================================================================

static Position startpos() {
    init_bb();
    Position pos;
    pos.set_startpos();
    return pos;
}

static Position from_fen(const std::string& fen) {
    init_bb();
    return Position::from_fen(fen);
}

// ===========================================================================
// Debugging
// ===========================================================================
// TEST_CASE("debug: pawn attack raw bits", "[debug]") {
//     init_bb();

//     // white pawn on e2
//     Bitboard e2_pawn = BB::square_bb(e2);
//     Bitboard attacks = BB::pawn_attacks(e2_pawn, WHITE);

//     std::cout << "White pawn on e2 attacks:\n";
//     BB::print(attacks);

//     // Black pawn on e7
//     Bitboard e7_pawn = BB::square_bb(e7);
//     Bitboard battacks = BB::pawn_attacks(e7_pawn, BLACK);

//     std::cout << "Black pawn on e2 attacks:\n";
//     BB::print(battacks);
// }

// TEST_CASE("debug: is_attacked_by internals", "[debug]") {
//     init_bb();
//     Position pos = from_fen("8/8/8/8/8/8/4P3/4K2K w - - 0 1");

//     // probe d1
//     Bitboard occ = pos.all_pieces();
//     std::cout << "occ: \n";
//     BB::print(occ);

//     std::cout << "white pawns:\n";
//     BB::print(pos.pieces(WHITE, PAWN));

//     // Reverse attack on d1 as black
//     Bitboard rev = BB::pawn_attacks(BB::square_bb(d1), BLACK);
//     std::cout << "pawn_attacks(d1, BLACK) : \n";
//     BB::print(rev);
//     std::cout << "overlap with white pawns: " << (bool)(rev & pos.pieces(WHITE, PAWN)) << "\n";

//     // Bishop attack from b1
//     Bitboard bish = BB::bishop_attack(b1, occ);
//     std::cout << "bishop_attack(b1, occ): \n";
//     BB::print(bish);
//     std::cout << "overlap with white bishops: " << (bool)(bish & pos.pieces(WHITE, BISHOP)) << "\n";

//     // For comparison, bishop from e4
//     Position pos2 = from_fen("8/8/8/8/4B3/8/8/4K2K w - - 0 1");
//     Bitboard occ2 = pos2.all_pieces();
//     std::cout << "bishop_attack(e4, occ2):\n";
//     BB::print(BB::bishop_attack(e4, occ2));
//     std::cout << "bishop_attack(b1, occ2):\n";
//     BB::print(BB::bishop_attack(b1, occ2));
// }

// ============================================================================
// FEN Parsing & Round-trip
// ============================================================================

TEST_CASE("FEN round-trip: starting position", "[fen]") {
    const std::string fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    Position pos = from_fen(fen);
    REQUIRE(pos.to_fen() == fen);
}

TEST_CASE("FEN round-trip: mid-game position", "[fen]") {
    const std::string fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4";
    Position pos = from_fen(fen);
    REQUIRE(pos.to_fen() == fen);
}

TEST_CASE("FEN round-trip: en passant square", "[fen]") {
    const std::string fen = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3";
    Position pos = from_fen(fen);
    REQUIRE(pos.to_fen() == fen);
}

TEST_CASE("FEN round-trip: no castling rights", "[fen]") {
    const std::string fen = "r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1";
    Position pos = from_fen(fen);
    REQUIRE(pos.to_fen() == fen);
}

TEST_CASE("FEN round-trip: partial castling rights", "[fen]") {
    const std::string fen = "r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1";
    Position pos = from_fen(fen);
    REQUIRE(pos.to_fen() == fen);
}

TEST_CASE("FEN round-trip: black to move", "[fen]") {
    const std::string fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
    Position pos = from_fen(fen);
    REQUIRE(pos.to_fen() == fen);
}

TEST_CASE("FEN round-trip: endgame KQvK", "[fen]") {
    const std::string fen = "8/8/4k3/8/8/3K4/8/7Q w - - 0 1";
    Position pos = from_fen(fen);
    REQUIRE(pos.to_fen() == fen);
}

TEST_CASE("FEN round-trip: high halfmove clock", "[fen]") {
    const std::string fen = "8/8/4k3/8/8/3K4/8/7Q w - - 47 92";
    Position pos = from_fen(fen);
    REQUIRE(pos.to_fen() == fen);
}

TEST_CASE("FEN parsing: invalid piece character throws", "[fen]") {
    init_bb();
    REQUIRE_THROWS_AS(
        Position::from_fen("rnbqkXnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        std::invalid_argument
    );
}

TEST_CASE("FEN parsing: invalid side to move throws", "[fen]") {
    init_bb();
    REQUIRE_THROWS_AS(
        Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1"),
        std::invalid_argument
    );
}

TEST_CASE("FEN parsing: invalid en passant square throws", "[fen]") {
    init_bb();
    REQUIRE_THROWS_AS(
        Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq e4 0 1"),
        std::invalid_argument
    );
}

// ============================================================================
// Starting Position Sanity
// ============================================================================

TEST_CASE("Starting position: side to move is white", "[startpos]") {
    Position pos = startpos();
    REQUIRE(pos.side_to_move() == WHITE);
}

TEST_CASE("Starting position: all castling rights", "[startpos]") {
    Position pos = startpos();
    REQUIRE(pos.can_castle(WHITE_OO));
    REQUIRE(pos.can_castle(WHITE_OOO));
    REQUIRE(pos.can_castle(BLACK_OO));
    REQUIRE(pos.can_castle(BLACK_OOO));
}

TEST_CASE("Starting position: no en passant square", "[startpos]") {
    Position pos = startpos();
    REQUIRE(pos.ep_square() == NO_SQUARE);
}

TEST_CASE("Starting position: halfmove clock is zero", "[startpos]") {
    Position pos = startpos();
    REQUIRE(pos.halfmove() == 0);
}

TEST_CASE("Starting position: fullmove number is one", "[startpos]") {
    Position pos = startpos();
    REQUIRE(pos.fullmove() == 1);
}

TEST_CASE("Starting position: correct piece counts", "[startpos]") {
    Position pos = startpos();

    REQUIRE(popcount(pos.pieces(WHITE, PAWN))   == 8);
    REQUIRE(popcount(pos.pieces(BLACK, PAWN))   == 8);
    REQUIRE(popcount(pos.pieces(WHITE, KNIGHT)) == 2);
    REQUIRE(popcount(pos.pieces(BLACK, KNIGHT)) == 2);
    REQUIRE(popcount(pos.pieces(WHITE, BISHOP)) == 2);
    REQUIRE(popcount(pos.pieces(BLACK, BISHOP)) == 2);
    REQUIRE(popcount(pos.pieces(WHITE, ROOK))   == 2);
    REQUIRE(popcount(pos.pieces(BLACK, ROOK))   == 2);
    REQUIRE(popcount(pos.pieces(WHITE, QUEEN))  == 1);
    REQUIRE(popcount(pos.pieces(BLACK, QUEEN))  == 1);
    REQUIRE(popcount(pos.pieces(WHITE, KING))   == 1);
    REQUIRE(popcount(pos.pieces(BLACK, KING))   == 1);
}

TEST_CASE("Starting position: total occupancy is 32 squares", "[startpos]") {
    Position pos = startpos();
    REQUIRE(popcount(pos.all_pieces()) == 32);
}

TEST_CASE("Starting position: white occupies ranks 1 and 2", "[startpos]") {
    Position pos = startpos();
    REQUIRE(Bitboard(pos.pieces(WHITE) & (BB::RANK_1 | BB::RANK_2)) == pos.pieces(WHITE));
}

TEST_CASE("Starting position: black occupies ranks 7 and 8", "[startpos]") {
    Position pos = startpos();
    REQUIRE((pos.pieces(BLACK) & (BB::RANK_7 | BB::RANK_8)) == pos.pieces(BLACK));
}

// ============================================================================
// Piece Placement on Specific Squares
// ============================================================================

TEST_CASE("piece_on: correct pieces on starting squares", "[piece_on]") {
    Position pos = startpos();

    REQUIRE(pos.piece_on(e1) == WHITE_KING);
    REQUIRE(pos.piece_on(d1) == WHITE_QUEEN);
    REQUIRE(pos.piece_on(a1) == WHITE_ROOK);
    REQUIRE(pos.piece_on(h1) == WHITE_ROOK);
    REQUIRE(pos.piece_on(b1) == WHITE_KNIGHT);
    REQUIRE(pos.piece_on(g1) == WHITE_KNIGHT);
    REQUIRE(pos.piece_on(c1) == WHITE_BISHOP);
    REQUIRE(pos.piece_on(f1) == WHITE_BISHOP);

    REQUIRE(pos.piece_on(e8) == BLACK_KING);
    REQUIRE(pos.piece_on(d8) == BLACK_QUEEN);
    REQUIRE(pos.piece_on(a8) == BLACK_ROOK);
    REQUIRE(pos.piece_on(h8) == BLACK_ROOK);
    REQUIRE(pos.piece_on(b8) == BLACK_KNIGHT);
    REQUIRE(pos.piece_on(g8) == BLACK_KNIGHT);
    REQUIRE(pos.piece_on(c8) == BLACK_BISHOP);
    REQUIRE(pos.piece_on(f8) == BLACK_BISHOP);
}

TEST_CASE("piece_on: empty squares return NO_PIECE", "[piece_on]") {
    Position pos = startpos();

    // Ranks 3-6 should all be empty in the starting position
    for (int r = 2; r <= 5; r++) {
        for (int f = 0; f < 8; f++) {
            Square sq = make_square((ui8)f, (ui8)r);
            REQUIRE(pos.piece_on(sq) == NO_PIECE);
            REQUIRE(pos.empty(sq));
        }
    }
}

TEST_CASE("piece_on: all pawns on correct ranks", "[piece_on]") {
    Position pos = startpos();

    for (int f = 0; f < 8; f++) {
        REQUIRE(pos.piece_on(make_square((ui8)f, 1)) == WHITE_PAWN);
        REQUIRE(pos.piece_on(make_square((ui8)f, 6)) == BLACK_PAWN);
    }
}

// ============================================================================
// King Square
// ============================================================================

TEST_CASE("king_square: correct in starting position", "[king]") {
    Position pos = startpos();
    REQUIRE(pos.king_square(WHITE) == e1);
    REQUIRE(pos.king_square(BLACK) == e8);
}

TEST_CASE("king_square: correct after custom FEN", "[king]") {
    Position pos = from_fen("8/8/4k3/8/8/3K4/8/7Q w - - 0 1");
    REQUIRE(pos.king_square(WHITE) == d3);
    REQUIRE(pos.king_square(BLACK) == e6);
}

// ============================================================================
// Castling Rights
// ============================================================================

TEST_CASE("castling: only white kingside", "[castling]") {
    Position pos = from_fen("r3k2r/8/8/8/8/8/8/R3K2R w K - 0 1");
    REQUIRE( pos.can_castle(WHITE_OO));
    REQUIRE(!pos.can_castle(WHITE_OOO));
    REQUIRE(!pos.can_castle(BLACK_OO));
    REQUIRE(!pos.can_castle(BLACK_OOO));
}

TEST_CASE("castling: only black queenside", "[castling]") {
    Position pos = from_fen("r3k2r/8/8/8/8/8/8/R3K2R w q - 0 1");
    REQUIRE(!pos.can_castle(WHITE_OO));
    REQUIRE(!pos.can_castle(WHITE_OOO));
    REQUIRE(!pos.can_castle(BLACK_OO));
    REQUIRE( pos.can_castle(BLACK_OOO));
}

TEST_CASE("castling: no rights at all", "[castling]") {
    Position pos = from_fen("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1");
    REQUIRE(!pos.can_castle(WHITE_OO));
    REQUIRE(!pos.can_castle(WHITE_OOO));
    REQUIRE(!pos.can_castle(BLACK_OO));
    REQUIRE(!pos.can_castle(BLACK_OOO));
}

// ============================================================================
// En Passant
// ============================================================================

TEST_CASE("en passant: square parsed correctly for white", "[ep]") {
    // Black just pushed d7-d5, en passant square is d6
    Position pos = from_fen("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3");
    REQUIRE(pos.ep_square() == d6);
}

TEST_CASE("en passant: square parsed correctly for black", "[ep]") {
    // White just pushed e2-e4, en passant square is e3
    Position pos = from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    REQUIRE(pos.ep_square() == e3);
}

TEST_CASE("en passant: no square when dash", "[ep]") {
    Position pos = startpos();
    REQUIRE(pos.ep_square() == NO_SQUARE);
}

// ============================================================================
// Attack Detection
// ============================================================================

TEST_CASE("is_attacked_by: starting position back rank squares not attacked", "[attacks]") {
    Position pos = startpos();

    // White's back rank is not attacked by black in the starting position
    for (int f = 0; f < 8; f++) {
        Square sq = make_square((ui8)f, 0);
        REQUIRE_FALSE(pos.is_attacked_by(sq, BLACK));
    }
}

TEST_CASE("is_attacked_by: white pawns attack rank 3 in starting position", "[attacks]") {
    Position pos = startpos();

    // Every square on rank 3 (index 2) is attacked by a white pawn
    // (each pawn on rank 2 attacks two squares on rank 3)
    Bitboard rank3_attacked = 0;
    for (int f = 0; f < 8; f++) {
        Square sq = make_square((ui8)f, 2);
        if (pos.is_attacked_by(sq, WHITE))
            rank3_attacked |= BB::square_bb(sq);
    }
    // b3 through g3 are attacked (a3 and h3 are only attacked by one pawn each)
    // but all of rank 3 should be attacked since pawns on a2 and h2 cover b3/g3
    REQUIRE(rank3_attacked == BB::RANK_3);
}

TEST_CASE("is_attacked_by: knight attacks", "[attacks]") {
    // Knight on e4, everything else empty except kings
    Position pos = from_fen("8/8/8/8/4N3/8/8/4K2k w - - 0 1");

    // e4 knight attacks: d6, f6, c5, g5, c3, g3, d2, f2
    REQUIRE(pos.is_attacked_by(d6, WHITE));
    REQUIRE(pos.is_attacked_by(f6, WHITE));
    REQUIRE(pos.is_attacked_by(c5, WHITE));
    REQUIRE(pos.is_attacked_by(g5, WHITE));
    REQUIRE(pos.is_attacked_by(c3, WHITE));
    REQUIRE(pos.is_attacked_by(g3, WHITE));
    REQUIRE(pos.is_attacked_by(d2, WHITE));
    REQUIRE(pos.is_attacked_by(f2, WHITE));

    // Squares not attacked by the knight
    REQUIRE_FALSE(pos.is_attacked_by(e5, WHITE));
    REQUIRE_FALSE(pos.is_attacked_by(e6, WHITE));
    REQUIRE_FALSE(pos.is_attacked_by(a1, WHITE));
}

TEST_CASE("is_attacked_by: bishop attacks along diagonal", "[attacks]") {
    Position pos = from_fen("8/8/8/8/4B3/8/8/4K2k w - - 0 1");

    // Bishop on e4 should attack a8 along the NW diagonal
    REQUIRE(pos.is_attacked_by(a8, WHITE));
    REQUIRE(pos.is_attacked_by(h7, WHITE));
    REQUIRE(pos.is_attacked_by(h1, WHITE));
    REQUIRE(pos.is_attacked_by(b1, WHITE));

    // Should not attack orthogonally
    REQUIRE_FALSE(pos.is_attacked_by(e8, WHITE));
    REQUIRE_FALSE(pos.is_attacked_by(a4, WHITE));
}

TEST_CASE("is_attacked_by: rook attacks orthogonally", "[attacks]") {
    Position pos = from_fen("8/8/8/8/4R3/8/8/4K2k w - - 0 1");

    REQUIRE(pos.is_attacked_by(e8, WHITE));
    REQUIRE(pos.is_attacked_by(e1, WHITE));
    REQUIRE(pos.is_attacked_by(a4, WHITE));
    REQUIRE(pos.is_attacked_by(h4, WHITE));

    // Should not attack diagonally
    REQUIRE_FALSE(pos.is_attacked_by(d5, WHITE));
    REQUIRE_FALSE(pos.is_attacked_by(f3, WHITE));
}

TEST_CASE("is_attacked_by: queen attacks both diagonally and orthogonally", "[attacks]") {
    Position pos = from_fen("8/8/8/8/4Q3/8/8/4K2k w - - 0 1");

    REQUIRE(pos.is_attacked_by(e8, WHITE));  // North (rook-like)
    REQUIRE(pos.is_attacked_by(a4, WHITE));  // West  (rook-like)
    REQUIRE(pos.is_attacked_by(a8, WHITE));  // NW    (bishop-like)
    REQUIRE(pos.is_attacked_by(h7, WHITE));  // NE    (bishop-like)
    REQUIRE(pos.is_attacked_by(h1, WHITE));  // SE    (bishop-like)
    REQUIRE(pos.is_attacked_by(b1, WHITE));  // SW    (bishop-like)
}

TEST_CASE("is_attacked_by: sliding piece blocked by own piece", "[attacks]") {
    // Rook on a1, own pawn on a4 — a5 through a8 should not be attacked
    Position pos = from_fen("8/8/8/8/P7/8/8/R3K2k w - - 0 1");

    REQUIRE(pos.is_attacked_by(a2, WHITE));
    REQUIRE(pos.is_attacked_by(a3, WHITE));
    REQUIRE(pos.is_attacked_by(a4, WHITE));  // The pawn itself is attacked (by the rook behind it)
    REQUIRE_FALSE(pos.is_attacked_by(a5, WHITE));
    REQUIRE_FALSE(pos.is_attacked_by(a8, WHITE));
}

TEST_CASE("is_attacked_by: sliding piece blocked by enemy piece", "[attacks]") {
    // Rook on a1, black pawn on a4 — a4 is attacked but a5 onward is not
    Position pos = from_fen("8/8/8/8/p7/8/8/R3K2k w - - 0 1");

    REQUIRE(pos.is_attacked_by(a4, WHITE));
    REQUIRE_FALSE(pos.is_attacked_by(a5, WHITE));
}

TEST_CASE("is_attacked_by: pawn attacks correct direction for white", "[attacks]") {
    Position pos = from_fen("8/8/8/8/8/8/4P3/8 w - - 0 1 ");

    // White pawn on e2 attacks d3 and f3, not d1/f1 (backwards)
    REQUIRE(pos.is_attacked_by(d3, WHITE));
    REQUIRE(pos.is_attacked_by(f3, WHITE));
    REQUIRE_FALSE(pos.is_attacked_by(d1, WHITE));
    REQUIRE_FALSE(pos.is_attacked_by(f1, WHITE));
    REQUIRE_FALSE(pos.is_attacked_by(e3, WHITE));  // Directly ahead is not an attack
}

TEST_CASE("is_attacked_by: pawn attacks correct direction for black", "[attacks]") {
    Position pos = from_fen("8/4p3/8/8/8/8/8/8 b - - 0 1 ");

    // Black pawn on e7 attacks d6 and f6, not d8/f8 (backwards)
    REQUIRE(pos.is_attacked_by(d6, BLACK));
    REQUIRE(pos.is_attacked_by(f6, BLACK));
    REQUIRE_FALSE(pos.is_attacked_by(e8, BLACK));
    REQUIRE_FALSE(pos.is_attacked_by(d8, BLACK));
    REQUIRE_FALSE(pos.is_attacked_by(f8, BLACK));
    REQUIRE_FALSE(pos.is_attacked_by(e6, BLACK));
}

TEST_CASE("is_attacked_by: king attacks", "[attacks]") {
    Position pos = from_fen("8/8/8/8/4K3/8/8/7k w - - 0 1");

    // King on e4 attacks all 8 adjacent squares
    REQUIRE(pos.is_attacked_by(d5, WHITE));
    REQUIRE(pos.is_attacked_by(e5, WHITE));
    REQUIRE(pos.is_attacked_by(f5, WHITE));
    REQUIRE(pos.is_attacked_by(d4, WHITE));
    REQUIRE(pos.is_attacked_by(f4, WHITE));
    REQUIRE(pos.is_attacked_by(d3, WHITE));
    REQUIRE(pos.is_attacked_by(e3, WHITE));
    REQUIRE(pos.is_attacked_by(f3, WHITE));

    // Not further away
    REQUIRE_FALSE(pos.is_attacked_by(c6, WHITE));
    REQUIRE_FALSE(pos.is_attacked_by(e6, WHITE));
}

// ============================================================================
// Check Detection
// ============================================================================

TEST_CASE("in_check: starting position is not in check", "[check]") {
    Position pos = startpos();
    REQUIRE_FALSE(pos.in_check());
}

TEST_CASE("in_check: king in check from rook", "[check]") {
    // Black rook on e8 gives check to white king on e1
    Position pos = from_fen("4r3/8/8/8/8/8/8/4K2k w - - 0 1");
    REQUIRE(pos.in_check());
}

TEST_CASE("in_check: king in check from knight", "[check]") {
    // Black knight on f3 gives check to white king on e1
    Position pos = from_fen("8/8/8/8/8/5n2/8/4K2k w - - 0 1");
    REQUIRE(pos.in_check());
}

TEST_CASE("in_check: king in check from bishop", "[check]") {
    Position pos = from_fen("8/8/8/8/8/2b5/8/4K2k w - - 0 1");
    REQUIRE(pos.in_check());
}

TEST_CASE("in_check: king in check from pawn", "[check]") {
    // Black pawn on d2 gives check to white king on e1
    Position pos = from_fen("8/8/8/8/8/8/3p4/4K2k w - - 0 1");
    REQUIRE(pos.in_check());
}

TEST_CASE("in_check: check is blocked by intervening piece", "[check]") {
    // Black rook on e8, white pawn on e4 blocks — white king on e1 not in check
    Position pos = from_fen("4r3/8/8/8/4P3/8/8/4K2k w - - 0 1");
    REQUIRE_FALSE(pos.in_check());
}

TEST_CASE("in_check: black king in check", "[check]") {
    // White queen on h5 gives check to black king on e8
    Position pos = from_fen("4k3/8/8/7Q/8/8/8/4K3 b - - 0 1");
    REQUIRE(pos.in_check());
}
