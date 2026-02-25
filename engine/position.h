#pragma once

#include "types.h"
#include "bitboard.h"
#include "move.h"

#include <stdexcept>

// =============================================================
// Undo Information
// =============================================================
struct UndoInfo {
    Piece captured;
    Square ep_sq;
    ui8 castling_rights;
    ui8 halfmove_clock;
};

class Position {
public:
    Position() { clear(); }

    // Parse FEN string and return a position
    static Position from_fen(const std::string& fen);

    std::string to_fen() const;

    // Set to standard chess starting position
    void set_startpos();

    // ------------------------------------------------------
    // Piece queries
    // ------------------------------------------------------

    Bitboard pieces(Colour c, PieceType pt) const {
        return piece_bb[c][pt];
    }

    Bitboard pieces(PieceType pt) const {
        return piece_bb[WHITE][pt] | piece_bb[BLACK][pt];
    }

    Bitboard pieces(Colour c) const {
        return occupancy[c];
    }

    // Returns a bitboard of all occupied squares
    Bitboard all_pieces() const {
        return occupancy[WHITE] | occupancy[BLACK];
    }

    // Piece on a given square(NO_PIECE if empty)
    Piece piece_on(Square sq) const {
        return board[sq];
    }

    bool empty(Square sq) const {
        return board[sq] == NO_PIECE;
    }

    Square king_square(Colour c) const {
        return (Square)lsb(piece_bb[c][KING]);
    }

    // -------------------------------------------------
    // State Queries
    // -------------------------------------------------

    Colour side_to_move() const { return stm; }
    Square ep_square() const { return ep_sq; }
    ui8 castling_rights() const { return castling_right; }
    ui8 halfmove() const { return halfmove_clock; }
    ui8 fullmove() const { return fullmove_number; }

    bool can_castle(CastlingRight cr) const {
        return castling_right & cr;
    }

    bool is_attacked_by(Square sq, Colour c) const;

    bool is_attacked_by_pawn(Square sq, Colour c) const;
    bool is_attacked_by_knight(Square sq, Colour c) const;
    bool is_attacked_by_bishop(Square sq, Colour c) const;
    bool is_attacked_by_rook(Square sq, Colour c) const;
    bool is_attacked_by_queen(Square sq, Colour c) const;
    bool is_attacked_by_king(Square sq, Colour c) const;

    bool in_check() const {
        return is_attacked_by(king_square(stm), flip(stm));
    }

    // Make/Unmake move on the position
    UndoInfo make_move(Move m);
    void unmake_move(Move m, const UndoInfo& undo);

    // ---------------------------------------------------
    // Debugging
    // ---------------------------------------------------
    void print() const; // Prints ASCII board to stdout
private:
    Bitboard piece_bb[2][6];
    Bitboard occupancy[2];
    Piece board[64];

    Colour stm; //  Side to move
    Square ep_sq; // En Passant square
    ui8 castling_right;
    ui8 halfmove_clock;
    ui8 fullmove_number;

    void clear();
    void put_piece(Piece p, Square sq);
    void remove_piece(Square sq);
    void move_piece(Square from, Square to);
};
