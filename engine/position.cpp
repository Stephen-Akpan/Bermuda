#include "position.h"
#include <iostream>
#include <sstream>
#include <cctype>
#include <stdexcept>

static constexpr ui8 castling_rights_mask[64] = {
    ~WHITE_OOO, 15, 15, 15, ~(WHITE_OO | WHITE_OOO), 15, 15, ~WHITE_OO, // rank 1
    15, 15, 15, 15, 15, 15, 15, 15, // rank 2
    15, 15, 15, 15, 15, 15, 15, 15, // rank 3
    15, 15, 15, 15, 15, 15, 15, 15, // rank 4
    15, 15, 15, 15, 15, 15, 15, 15, // rank 5
    15, 15, 15, 15, 15, 15, 15, 15, // rank 6
    15, 15, 15, 15, 15, 15, 15, 15, // rank 7
    ~BLACK_OOO, 15, 15, 15, ~(BLACK_OO | BLACK_OOO), 15, 15, ~BLACK_OO
};

static bool ep_is_capturable(const Position& pos, Square ep_sq) {
    if (ep_sq == NO_SQUARE) return false;
    Colour them = flip(pos.side_to_move());
    Bitboard ep_attackers = BB::pawn_attacks(BB::square_bb(ep_sq), them) & pos.pieces(pos.side_to_move(), PAWN);
    return ep_attackers != 0;
}

// ============================================================================
// Internal Helpers
// ============================================================================

void Position::clear() {
    for (i32 c = 0; c < 2; c++) {
        occupancy[c] = 0;
        for (i32 pt = 0; pt < 6; pt++)
            piece_bb[c][pt] = 0;
    }
    for (i32 sq = 0; sq < 64; sq++)
        board[sq] = NO_PIECE;

    stm             = WHITE;
    ep_sq           = NO_SQUARE;
    castling_right = NO_CASTLING;
    halfmove_clock  = 0;
    fullmove_number = 1;
    hash_key = 0;
}

void Position::put_piece(Piece p, Square sq) {
    Colour    c  = colour_of(p);
    PieceType pt = type_of(p);

    board[sq]       = p;
    piece_bb[c][pt] |= BB::square_bb(sq);
    occupancy[c]    |= BB::square_bb(sq);

    #ifndef NDEBUG
    Bitboard recomputed_white = 0;
    Bitboard recomputed_black = 0;

    for (i32 sq = 0; sq < 64; sq++) {
        Piece p = board[sq];
        if (p == NO_PIECE) continue;
        if (colour_of(p) == WHITE)
            recomputed_white |= BB::square_bb(Square(sq));
        else
            recomputed_black |= BB::square_bb(Square(sq));
    }
    assert(recomputed_white == occupancy[WHITE]);
    assert(recomputed_black == occupancy[BLACK]);
    #endif
}

void Position::remove_piece(Square sq) {
    Piece p = board[sq];
    if (p == NO_PIECE) return;

    Colour    c  = colour_of(p);
    PieceType pt = type_of(p);

    board[sq]       = NO_PIECE;
    piece_bb[c][pt] &= ~BB::square_bb(sq);
    occupancy[c]    &= ~BB::square_bb(sq);

    #ifndef NDEBUG
    Bitboard recomputed_white = 0;
    Bitboard recomputed_black = 0;

    for (i32 sq = 0; sq < 64; sq++) {
        Piece p = board[sq];
        if (p == NO_PIECE) continue;
        if (colour_of(p) == WHITE)
            recomputed_white |= BB::square_bb(Square(sq));
        else
            recomputed_black |= BB::square_bb(Square(sq));
    }
    assert(recomputed_white == occupancy[WHITE]);
    assert(recomputed_black == occupancy[BLACK]);
    #endif
}

void Position::move_piece(Square from, Square to) {
    Piece p = board[from];
    Colour c = colour_of(p);
    PieceType pt = type_of(p);

    board[to] = p;
    board[from] = NO_PIECE;
    piece_bb[c][pt] ^= BB::square_bb(from) | BB::square_bb(to);
    occupancy[c] ^= BB::square_bb(from) | BB::square_bb(to);
}

// ============================================================================
// FEN Parsing
// ============================================================================

Position Position::from_fen(const std::string& fen) {
    Position pos;
    pos.clear();

    std::istringstream ss(fen);
    std::string token;
 
    // Field 1: Piece placement
 
    ss >> token;

    i32 rank = 7;  // FEN starts from rank 8
    i32 file = 0;

    for (char c : token) {
        if (c == '/') {
            rank--;
            file = 0;
        } else if (std::isdigit(c)) {
            file += (c - '0');
        } else {
            Colour col = std::isupper(c) ? WHITE : BLACK;
            char lower = std::tolower(c);

            PieceType pt;
            switch (lower) {
                case 'p': pt = PAWN;   break;
                case 'n': pt = KNIGHT; break;
                case 'b': pt = BISHOP; break;
                case 'r': pt = ROOK;   break;
                case 'q': pt = QUEEN;  break;
                case 'k': pt = KING;   break;
                default:
                    throw std::invalid_argument(
                        std::string("Invalid FEN piece character: ") + c);
            }

            Square sq = make_square((ui8)file, (ui8)rank);
            pos.put_piece(make_piece(col, pt), sq);
            file++;
        }
    }

    // ------------------------------------------------------------------------
    // Rank validation
    // ------------------------------------------------------------------------
    if (rank != 0) {
        throw std::invalid_argument("Invalid FEN: expected 8 ranks, got " + std::to_string(8 - rank));
    }
 
    // Field 2: Side to move
 
    ss >> token;
    if (token == "w")      pos.stm = WHITE;
    else if (token == "b") pos.stm = BLACK;
    else throw std::invalid_argument("Invalid FEN side to move: " + token);

    // Field 3: Castling rights
 
    ss >> token;
    pos.castling_right = NO_CASTLING;

    if (token != "-") {
        for (char c : token) {
            switch (c) {
                case 'K': pos.castling_right |= WHITE_OO;  break;
                case 'Q': pos.castling_right |= WHITE_OOO; break;
                case 'k': pos.castling_right |= BLACK_OO;  break;
                case 'q': pos.castling_right |= BLACK_OOO; break;
                default:
                    throw std::invalid_argument(
                        std::string("Invalid FEN castling character: ") + c);
            }
        }
    }
 
    // Field 4: En passant square
 
    ss >> token;
    if (token == "-") {
        pos.ep_sq = NO_SQUARE;
    } else {
        if (token.size() < 2
            || token[0] < 'a' || token[0] > 'h'
            || (token[1] != '3' && token[1] != '6')) {
            throw std::invalid_argument("Invalid FEN en passant square: " + token);
        }
        ui8 ep_file = token[0] - 'a';
        ui8 ep_rank = token[1] - '1';
        pos.ep_sq = make_square(ep_file, ep_rank);
    }

    // Field 5: Halfmove clock
 
    if (ss >> token) {
        pos.halfmove_clock = (ui8)std::stoi(token);
    }
 
    // Field 6: Fullmove number
 
    if (ss >> token) {
        pos.fullmove_number = (ui16)std::stoi(token);
    }

    // Field 7: Hash
    pos.hash_key = Zobrist::compute(pos.piece_bb, pos.stm, pos.castling_right, pos.ep_sq);

    return pos;
}

// ============================================================================
// FEN Serialization
// ============================================================================

std::string Position::to_fen() const {
    std::ostringstream ss;
 
    // Field 1: Piece placement
 
    for (i32 r = 7; r >= 0; r--) {
        i32 empty_count = 0;

        for (i32 f = 0; f < 8; f++) {
            Square sq = make_square((ui8)f, (ui8)r);
            Piece  p  = board[sq];

            if (p == NO_PIECE) {
                empty_count++;
            } else {
                if (empty_count > 0) {
                    ss << empty_count;
                    empty_count = 0;
                }

                PieceType pt = type_of(p);
                Colour    c  = colour_of(p);

                char piece_char;
                switch (pt) {
                    case PAWN:   piece_char = 'p'; break;
                    case KNIGHT: piece_char = 'n'; break;
                    case BISHOP: piece_char = 'b'; break;
                    case ROOK:   piece_char = 'r'; break;
                    case QUEEN:  piece_char = 'q'; break;
                    case KING:   piece_char = 'k'; break;
                    default:     piece_char = '?'; break;
                }

                if (c == WHITE) piece_char = std::toupper(piece_char);
                ss << piece_char;
            }
        }

        if (empty_count > 0) ss << empty_count;
        if (r > 0) ss << '/';
    }

    // Field 2: Side to move
 
    ss << ' ' << (stm == WHITE ? 'w' : 'b');
 
    // Field 3: Castling rights
 
    ss << ' ';
    if (castling_right == NO_CASTLING) {
        ss << '-';
    } else {
        if (castling_right & WHITE_OO)  ss << 'K';
        if (castling_right & WHITE_OOO) ss << 'Q';
        if (castling_right & BLACK_OO)  ss << 'k';
        if (castling_right & BLACK_OOO) ss << 'q';
    }
 
    // Field 4: En passant square
 
    ss << ' ';
    if (ep_sq == NO_SQUARE) {
        ss << '-';
    } else {
        ss << (char)('a' + file(ep_sq));
        ss << (char)('1' + rank(ep_sq));
    }

    // Fields 5 & 6: Halfmove clock and fullmove number
 
    ss << ' ' << (i32)halfmove_clock;
    ss << ' ' << (i32)fullmove_number;

    return ss.str();
}

// ============================================================================
// Starting Position
// ============================================================================

void Position::set_startpos() {
    *this = from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

// ============================================================================
// Attack Queries
// ============================================================================

bool Position::is_attacked_by(Square sq, Colour c) const {
    return is_attacked_by_pawn(sq, c) || is_attacked_by_knight(sq, c) 
        || is_attacked_by_bishop(sq, c) || is_attacked_by_rook(sq, c)
        || is_attacked_by_queen(sq, c) || is_attacked_by_king(sq, c);
}

bool Position::is_attacked_by_pawn(Square sq, Colour c) const
{
    bool attacks = BB::pawn_attacks(BB::square_bb(sq), flip(c)) & piece_bb[c][PAWN];
    if (attacks == true)  { 
        return true; 
    }  else { return false; }
}

bool Position::is_attacked_by_knight(Square sq, Colour c) const
{
    bool attacks = BB::knight_attack(sq) & piece_bb[c][KNIGHT];
    if (attacks == true)  { 
        return true; 
    }  else { return false; }
}

bool Position::is_attacked_by_bishop(Square sq, Colour c) const {
    Bitboard occ = all_pieces();
    bool attacks = BB::bishop_attack(sq, occ) & piece_bb[c][BISHOP];
    if (attacks == true)  { 
        return true; 
    }  else { return false; }
}

bool Position::is_attacked_by_rook(Square sq, Colour c) const {
    Bitboard occ = all_pieces();
    bool attacks = BB::rook_attack(sq, occ) & piece_bb[c][ROOK];
    if (attacks == true)  { 
        return true; 
    }  else { return false; }
}

bool Position::is_attacked_by_queen(Square sq, Colour c) const {
     Bitboard occ = all_pieces();
    bool attacks = BB::queen_attack(sq, occ) & piece_bb[c][QUEEN];
    if (attacks == true)  { 
        return true; 
    }  else { return false; }
}

bool Position::is_attacked_by_king(Square sq, Colour c) const {
    bool attacks = BB::king_attack(sq) & piece_bb[c][KING];
    if (attacks == true)  { 
        return true; 
    }  else { return false; }
}

// ============================================================================
// Make Move
// ============================================================================
UndoInfo Position::make_move(Move m) {
    // Save unrecoverable state
    UndoInfo undo;
    undo.ep_sq = ep_sq;
    undo.castling_rights = castling_right;
    undo.halfmove_clock = halfmove_clock;
    undo.captured = NO_PIECE;
    undo.hash_key = hash_key;

    const Square from = m.from();
    const Square to = m.to();
    const Piece piece = board[from];
    const Colour us = stm;
    const Colour them = flip(stm);

    // Update hash
    hash_key ^= Zobrist::piece_keys[piece][from];
    hash_key ^= Zobrist::castling_keys[castling_right];
    if (ep_sq != NO_SQUARE) {
        Bitboard attackers = BB::pawn_attacks(BB::square_bb(ep_sq), them) & piece_bb[us][PAWN];
        if (attackers)
            hash_key ^= Zobrist::ep_keys[file(ep_sq)];
    }

    // Halfmove clock: reset on pawn move or capture, else increment
    if (type_of(piece) == PAWN || board[to] != NO_PIECE) {
        halfmove_clock = 0;
    } else {
        halfmove_clock++;
    }

    switch (m.flag()) {
        case NORMAL: {
            // Save captured piece if any
            undo.captured = board[to];
            if (undo.captured != NO_PIECE) {
                remove_piece(to);
                hash_key ^= Zobrist::piece_keys[undo.captured][to];
            }
            move_piece(from, to);
            hash_key ^= Zobrist::piece_keys[piece][to];
            break;
        }
        case PROMOTION: {
            undo.captured = board[to];
            if (undo.captured != NO_PIECE) {
                remove_piece(to);
                hash_key ^= Zobrist::piece_keys[undo.captured][to];
            }
            
            // Remove the pawn
            remove_piece(from);

            PieceType promo_type;
            switch (m.promo())
            {
            case PROMO_KNIGHT: promo_type = KNIGHT; break;
            case PROMO_BISHOP: promo_type = BISHOP; break;
            case PROMO_ROOK: promo_type = ROOK; break;
            case PROMO_QUEEN: promo_type = QUEEN; break;
            default: promo_type = QUEEN; break;
            }

            Piece promo_piece = make_piece(us, promo_type);
            put_piece(make_piece(us, promo_type), to);
            hash_key ^= Zobrist::piece_keys[promo_piece][to];
            break;
        }
        case EN_PASSANT: {
            // captured pawn is not on 'to' but one rank behind it
            Square cap_sq = (us == WHITE) ? Square(to - 8) : Square(to + 8);
            undo.captured = board[cap_sq];
            hash_key ^= Zobrist::piece_keys[undo.captured][cap_sq];
            remove_piece(cap_sq);
            move_piece(from, to);
            hash_key ^= Zobrist::piece_keys[piece][to];
            break;
        }
        case CASTLING: {
            // King moves from 'from' to 'to'
            // Determine rook squares based on destination
            Square rook_from, rook_to;
            if (to == g1) { rook_from = h1; rook_to = f1; } // White kingside
            else if (to == c1) { rook_from = a1; rook_to = d1; } // White kingside
            else if (to == g8) { rook_from = h8; rook_to = f8; } // Black kingside
            else { rook_from = a8; rook_to = d8; } // Black queenside

            Piece rook = board[rook_from];
            hash_key ^= Zobrist::piece_keys[rook][rook_from];
            hash_key ^= Zobrist::piece_keys[rook][rook_to];

            move_piece(from, to);
            move_piece(rook_from, rook_to);
            hash_key ^= Zobrist::piece_keys[piece][to];
            break;
        }
    }

    // Update en passant square
    ep_sq = NO_SQUARE;

    if (type_of(piece) == PAWN) {
        // Double pawn push - set en pessant square
        i32 diff = (i32)to - (i32)from;
        if (diff == 16 || diff == -16) {
            Square new_ep = Square(from + diff / 2);
            Bitboard attackers = BB::pawn_attacks(BB::square_bb(new_ep), us) & piece_bb[them][PAWN];
            if (attackers) {
                ep_sq = new_ep;
                hash_key ^= Zobrist::ep_keys[file(new_ep)];
            }
            else {
                ep_sq = new_ep;
            }
        }
    }

    // Update castling rights
    castling_right &= castling_rights_mask[from] & castling_rights_mask[to];
    hash_key ^= Zobrist::castling_keys[castling_right];

    hash_key ^= Zobrist::side_key;

    // Update fullmove number
    if (us == BLACK)
        fullmove_number++;
    
    stm = them;

    #ifndef NDEBUG
    Bitboard recomputed_white = 0;
    Bitboard recomputed_black = 0;

    for (i32 sq = 0; sq < 64; sq++) {
        Piece p = board[sq];
        if (p == NO_PIECE) continue;
        if (colour_of(p) == WHITE)
            recomputed_white |= BB::square_bb(Square(sq));
        else
            recomputed_black |= BB::square_bb(Square(sq));
    }
    assert(recomputed_white == occupancy[WHITE]);
    assert(recomputed_black == occupancy[BLACK]);
    #endif

    return undo;
}

void Position::unmake_move(Move m, const UndoInfo &undo) {
    const Square from = m.from();
    const Square to = m.to();
    const Colour us = flip(stm);
    const Colour them = flip(us);

    // restore state
    ep_sq = undo.ep_sq;
    castling_right = undo.castling_rights;
    halfmove_clock = undo.halfmove_clock;
    stm = us;
    hash_key = undo.hash_key;

    if (them == BLACK)
        fullmove_number--;
    
    // Reverse the move flag
    switch (m.flag())
    {
    case NORMAL: {
        move_piece(to, from);

        if (undo.captured != NO_PIECE)
            put_piece(undo.captured, to);
        
        break;
    }
    case PROMOTION: {
        // Remove promoted piece
        remove_piece(to);
        put_piece(make_piece(us, PAWN), from);

        if (undo.captured != NO_PIECE)
            put_piece(undo.captured, to);

        break;
    }
    case EN_PASSANT: {
        move_piece(to, from);

        // Restorre captured pawn
        Square cap_sq = (us == WHITE) ? Square(to - 8) : Square(to + 8);
        put_piece(undo.captured, cap_sq);
        break;
    }
    case CASTLING: {
        Square rook_from, rook_to;
        if (to == g1) { rook_from = h1; rook_to = f1; } // White kingside
        else if (to == c1) { rook_from = a1; rook_to = d1; } // White kingside
        else if (to == g8) { rook_from = h8; rook_to = f8; } // Black kingside
        else { rook_from = a8; rook_to = d8; } // Black queenside

        move_piece(to, from); // King back
        move_piece(rook_to, rook_from); // Rook back
        break;
    }
    default:
        break;
    }

    #ifndef NDEBUG
    Bitboard recomputed_white = 0;
    Bitboard recomputed_black = 0;

    for (i32 sq = 0; sq < 64; sq++) {
        Piece p = board[sq];
        if (p == NO_PIECE) continue;
        if (colour_of(p) == WHITE)
            recomputed_white |= BB::square_bb(Square(sq));
        else
            recomputed_black |= BB::square_bb(Square(sq));
    }
    assert(recomputed_white == occupancy[WHITE]);
    assert(recomputed_black == occupancy[BLACK]);
    #endif
}

NullMoveUndo Position::make_null_move() {
    NullMoveUndo undo;
    undo.ep_sq = ep_sq;
    undo.hash_key = hash_key;
    undo.halfmove_clock = halfmove_clock;

    if (ep_sq != NO_SQUARE) {
        Bitboard attackers = BB::pawn_attacks(BB::square_bb(ep_sq), stm) & piece_bb[flip(stm)][PAWN];
        if (attackers) {
            hash_key ^= Zobrist::ep_keys[file(ep_sq)];
        }
    }
    hash_key ^= Zobrist::side_key;
    stm = flip(stm);
    ep_sq = NO_SQUARE;
    halfmove_clock++;

    return undo;
}

void Position::unmake_null_move(const NullMoveUndo& undo) {
    ep_sq = undo.ep_sq;
    hash_key = undo.hash_key;
    halfmove_clock = undo.halfmove_clock;
    stm = flip(stm);
}

// ============================================================================
// Debugging
// ============================================================================

void Position::print() const {
    static const char* piece_chars = "PNBRQKpnbrqk.";

    std::cout << "\n";
    for (i32 r = 7; r >= 0; r--) {
        std::cout << (r + 1) << "  ";
        for (i32 f = 0; f < 8; f++) {
            Square sq = make_square((ui8)f, (ui8)r);
            Piece  p  = board[sq];

            if (p == NO_PIECE) {
                std::cout << ".";
            } else {
                // White pieces: uppercase (0-5), black pieces: lowercase (8-13 → offset to 6-11)
                i32 idx = (colour_of(p) == WHITE) ? type_of(p) : type_of(p) + 6;
                std::cout << piece_chars[idx];
            }
            std::cout << ' ';
        }
        std::cout << '\n';
    }
    std::cout << "   a b c d e f g h\n\n";
    std::cout << "FEN: " << to_fen() << '\n';
    std::cout << "Side to move: " << (stm == WHITE ? "White" : "Black") << '\n';

    std::cout << "Castling: ";
    if (castling_right == NO_CASTLING) std::cout << "-";
    else {
        if (castling_right & WHITE_OO)  std::cout << "K";
        if (castling_right & WHITE_OOO) std::cout << "Q";
        if (castling_right & BLACK_OO)  std::cout << "k";
        if (castling_right & BLACK_OOO) std::cout << "q";
    }
    std::cout << '\n';

    std::cout << "En passant: ";
    if (ep_sq == NO_SQUARE) std::cout << "-";
    else std::cout << (char)('a' + file(ep_sq)) << (char)('1' + rank(ep_sq));
    std::cout << '\n';

    std::cout << "Halfmove clock: " << (i32)halfmove_clock << '\n';
    std::cout << "Fullmove number: " << (i32)fullmove_number << '\n';
}
