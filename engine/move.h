#pragma once

#include "types.h"
#include <string>

// ============================================================================
// Move
// ============================================================================

struct Move {
    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    // Default: null move (value of 0)
    constexpr Move() : data(0) {}

    // Normal move
    static constexpr Move make(Square from, Square to) {
        return Move(((ui16)NORMAL << 14) | ((ui16)to << 6) | (ui16)from);
    }

    // Promotion move
    static constexpr Move make_promo(Square from, Square to, PromoPiece pp) {
        return Move(((ui16)PROMOTION << 14) | ((ui16)pp << 12) | ((ui16)to << 6) | (ui16)from);
    }

    // En passant move
    static constexpr Move make_ep(Square from, Square to) {
        return Move(((ui16)EN_PASSANT << 14) | ((ui16)to << 6) | (ui16)from);
    }

    // Castling move
    static constexpr Move make_castling(Square from, Square to) {
        return Move(((ui16)CASTLING << 14) | ((ui16)to << 6) | (ui16)from);
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    constexpr Square    from()       const { return (Square)  (data & 0x3F); }
    constexpr Square    to()         const { return (Square)  ((data >> 6)  & 0x3F); }
    constexpr PromoPiece promo()     const { return (PromoPiece)((data >> 12) & 0x3); }
    constexpr MoveFlag  flag()       const { return (MoveFlag)((data >> 14) & 0x3); }

    constexpr bool is_promotion()   const { return flag() == PROMOTION; }
    constexpr bool is_en_passant()  const { return flag() == EN_PASSANT; }
    constexpr bool is_castling()    const { return flag() == CASTLING; }
    constexpr bool is_normal()      const { return flag() == NORMAL; }

    // Null move sentinel check
    constexpr bool is_null()        const { return data == 0; }

    // -------------------------------------------------------------------------
    // Equality
    // -------------------------------------------------------------------------

    constexpr bool operator==(Move other) const { return data == other.data; }
    constexpr bool operator!=(Move other) const { return data != other.data; }

    // -------------------------------------------------------------------------
    // String representation (e.g. "e2e4", "e7e8q" for promotion)
    // -------------------------------------------------------------------------

    std::string to_string() const {
        if (is_null()) return "0000";

        std::string s;
        s += (char)('a' + file(from()));
        s += (char)('1' + rank(from()));
        s += (char)('a' + file(to()));
        s += (char)('1' + rank(to()));

        if (is_promotion()) {
            switch (promo()) {
                case PROMO_KNIGHT: s += 'n'; break;
                case PROMO_BISHOP: s += 'b'; break;
                case PROMO_ROOK:   s += 'r'; break;
                case PROMO_QUEEN:  s += 'q'; break;
            }
        }

        return s;
    }

    // Raw data access (for hashing, storing in TT etc.)
    constexpr ui16 raw() const { return data; }

private:
    explicit constexpr Move(ui16 d) : data(d) {}
    ui16 data;
};

// Null move constant
constexpr Move NULL_MOVE = Move();

// ============================================================================
// MoveList
// ============================================================================
// Fixed-size stack-allocated move list. 256 entries is safely above the
// maximum number of legal moves in any chess position (~218 is the known max).

struct MoveList {
    static constexpr i32 MAX_MOVES = 256;

    // -------------------------------------------------------------------------
    // Mutation
    // -------------------------------------------------------------------------

    void push(Move m) {
        moves[count++] = m;
    }

    // Convenience pushers
    void push(Square from, Square to) {
        push(Move::make(from, to));
    }

    void push_promo(Square from, Square to) {
        // Always generate all four promotion types
        push(Move::make_promo(from, to, PROMO_QUEEN));
        push(Move::make_promo(from, to, PROMO_ROOK));
        push(Move::make_promo(from, to, PROMO_BISHOP));
        push(Move::make_promo(from, to, PROMO_KNIGHT));
    }

    void push_ep(Square from, Square to) {
        push(Move::make_ep(from, to));
    }

    void push_castling(Square from, Square to) {
        push(Move::make_castling(from, to));
    }

    void clear() { count = 0; }

    // -------------------------------------------------------------------------
    // Access
    // -------------------------------------------------------------------------

    i32  size()                    const { return count; }
    bool empty()                   const { return count == 0; }

    Move&       operator[](i32 i)        { return moves[i]; }
    const Move& operator[](i32 i) const  { return moves[i]; }

    Move*       begin()       { return moves; }
    Move*       end()         { return moves + count; }
    const Move* begin() const { return moves; }
    const Move* end()   const { return moves + count; }

private:
    Move moves[MAX_MOVES];
    i32  count = 0;
};
