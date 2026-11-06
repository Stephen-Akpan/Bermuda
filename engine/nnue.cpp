#include "nnue.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>

namespace NNUE {
namespace {

constexpr ui32 INPUT_COUNT = 768;
constexpr char FILE_MAGIC[8] = { 'B', 'M', 'N', 'N', 'U', 'E', '1', '\0' };

struct Network {
    ui32 hidden_size = 0;
    f32 eval_scale = 0.0f;
    std::vector<f32> feature_weights;
    std::vector<f32> feature_bias;
    std::vector<f32> output_weights;
    f32 output_bias = 0.0f;
    ui64 generation = 0;

    bool loaded() const { return hidden_size != 0; }
};

Network network;
ui64 next_generation = 1;

bool read_u32_le(std::istream& input, ui32& value) {
    ui8 bytes[4];
    if (!input.read(reinterpret_cast<char*>(bytes), sizeof(bytes)))
        return false;
    value = static_cast<ui32>(bytes[0])
          | (static_cast<ui32>(bytes[1]) << 8)
          | (static_cast<ui32>(bytes[2]) << 16)
          | (static_cast<ui32>(bytes[3]) << 24);
    return true;
}

bool read_f32_le(std::istream& input, f32& value) {
    ui32 bits;
    if (!read_u32_le(input, bits))
        return false;
    std::memcpy(&value, &bits, sizeof(value));
    return std::isfinite(value);
}

bool read_f32_array(std::istream& input, std::vector<f32>& values) {
    for (f32& value : values)
        if (!read_f32_le(input, value))
            return false;
    return true;
}

i32 feature_index(Colour piece_colour, PieceType type, Square square,
                 Colour perspective) {
    const i32 relative_piece = static_cast<i32>(type)
        + (piece_colour == perspective ? 0 : 6);
    const i32 relative_square = perspective == WHITE
        ? static_cast<i32>(square)
        : static_cast<i32>(square) ^ 56;
    return relative_piece * 64 + relative_square;
}

void add_piece(const Network& net, std::vector<f32>& accumulator,
               i32 index) {
    const f32* weights = net.feature_weights.data()
                       + static_cast<size_t>(index) * net.hidden_size;
    for (ui32 neuron = 0; neuron < net.hidden_size; ++neuron)
        accumulator[neuron] += weights[neuron];
}

void change_piece(const Network& net, Accumulator& accumulator, Piece piece,
                  Square square, f32 direction) {
    if (piece == NO_PIECE)
        return;

    const Colour colour = colour_of(piece);
    const PieceType type = type_of(piece);
    for (Colour perspective : { WHITE, BLACK }) {
        const i32 index = feature_index(colour, type, square, perspective);
        const f32* weights = net.feature_weights.data()
                           + static_cast<size_t>(index) * net.hidden_size;
        for (ui32 neuron = 0; neuron < net.hidden_size; ++neuron)
            accumulator.values[perspective][neuron] += direction * weights[neuron];
    }
}

std::vector<f32> build_accumulator(const Position& pos, const Network& net,
                                   Colour perspective) {
    std::vector<f32> accumulator = net.feature_bias;
    for (Colour colour : { WHITE, BLACK }) {
        for (PieceType type : { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING }) {
            Bitboard pieces = pos.pieces(colour, type);
            while (pieces) {
                const Square square = static_cast<Square>(pop_lsb(pieces));
                add_piece(net, accumulator,
                          feature_index(colour, type, square, perspective));
            }
        }
    }
    return accumulator;
}

} // namespace

bool load(const std::string& path, std::string& error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "unable to open NNUE file: " + path;
        return false;
    }

    char magic[sizeof(FILE_MAGIC)];
    ui32 input_count = 0;
    Network loaded;
    if (!input.read(magic, sizeof(magic))
        || !std::equal(std::begin(FILE_MAGIC), std::end(FILE_MAGIC), magic)
        || !read_u32_le(input, input_count)
        || !read_u32_le(input, loaded.hidden_size)
        || !read_f32_le(input, loaded.eval_scale)) {
        error = "invalid or truncated NNUE header";
        return false;
    }
    if (input_count != INPUT_COUNT || loaded.hidden_size != HIDDEN_SIZE
        || loaded.eval_scale <= 0.0f) {
        error = "unsupported NNUE dimensions or evaluation scale";
        return false;
    }

    loaded.feature_weights.resize(static_cast<size_t>(INPUT_COUNT) * loaded.hidden_size);
    loaded.feature_bias.resize(loaded.hidden_size);
    loaded.output_weights.resize(static_cast<size_t>(2) * loaded.hidden_size);
    if (!read_f32_array(input, loaded.feature_weights)
        || !read_f32_array(input, loaded.feature_bias)
        || !read_f32_array(input, loaded.output_weights)
        || !read_f32_le(input, loaded.output_bias)
        || input.peek() != std::char_traits<char>::eof()) {
        error = "invalid, non-finite, or truncated NNUE weights";
        return false;
    }

    loaded.generation = next_generation++;
    network = std::move(loaded);
    error.clear();
    return true;
}

void clear() {
    network = Network{};
    ++next_generation;
}

bool is_loaded() {
    return network.loaded();
}

void refresh(const Position& pos, Accumulator& accumulator) {
    accumulator = Accumulator{};
    if (!network.loaded())
        return;

    for (Colour perspective : { WHITE, BLACK })
        std::copy(network.feature_bias.begin(), network.feature_bias.end(),
                  accumulator.values[perspective]);

    for (Colour colour : { WHITE, BLACK }) {
        for (PieceType type : { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING }) {
            Bitboard pieces = pos.pieces(colour, type);
            while (pieces) {
                const Square square = static_cast<Square>(pop_lsb(pieces));
                change_piece(network, accumulator, make_piece(colour, type),
                             square, 1.0f);
            }
        }
    }
    accumulator.generation = network.generation;
    accumulator.valid = true;
}

void update_for_move(const Position& pos, Move move,
                     const Accumulator& parent, Accumulator& child) {
    if (!network.loaded()) {
        child.valid = false;
        return;
    }
    child = parent;
    if (!parent.valid || parent.generation != network.generation || move.is_null()) {
        child.valid = false;
        return;
    }

    const Colour us = pos.side_to_move();
    const Square from = move.from();
    const Square to = move.to();
    const Piece moving = pos.piece_on(from);
    if (moving == NO_PIECE) {
        child.valid = false;
        return;
    }

    auto remove_capture = [&](Square square) {
        const Piece captured = pos.piece_on(square);
        if (captured != NO_PIECE)
            change_piece(network, child, captured, square, -1.0f);
    };

    change_piece(network, child, moving, from, -1.0f);
    switch (move.flag()) {
    case NORMAL:
        remove_capture(to);
        change_piece(network, child, moving, to, 1.0f);
        break;
    case PROMOTION: {
        remove_capture(to);
        PieceType promoted_type = QUEEN;
        switch (move.promo()) {
        case PROMO_KNIGHT: promoted_type = KNIGHT; break;
        case PROMO_BISHOP: promoted_type = BISHOP; break;
        case PROMO_ROOK: promoted_type = ROOK; break;
        case PROMO_QUEEN: promoted_type = QUEEN; break;
        }
        change_piece(network, child, make_piece(us, promoted_type), to, 1.0f);
        break;
    }
    case EN_PASSANT: {
        const Square captured_square = static_cast<Square>(
            static_cast<i32>(to) - pawn_push_dir(us));
        remove_capture(captured_square);
        change_piece(network, child, moving, to, 1.0f);
        break;
    }
    case CASTLING: {
        change_piece(network, child, moving, to, 1.0f);
        Square rook_from;
        Square rook_to;
        if (to == g1) { rook_from = h1; rook_to = f1; }
        else if (to == c1) { rook_from = a1; rook_to = d1; }
        else if (to == g8) { rook_from = h8; rook_to = f8; }
        else { rook_from = a8; rook_to = d8; }
        const Piece rook = pos.piece_on(rook_from);
        change_piece(network, child, rook, rook_from, -1.0f);
        change_piece(network, child, rook, rook_to, 1.0f);
        break;
    }
    }
    child.valid = true;
}

Score evaluate(const Position& pos, const Accumulator& accumulator) {
    if (!network.loaded())
        return 0;
    if (!accumulator.valid || accumulator.generation != network.generation)
        return evaluate(pos);

    const f32* stm = accumulator.values[pos.side_to_move()];
    const f32* ntm = accumulator.values[flip(pos.side_to_move())];

    f64 output = network.output_bias;
    for (ui32 neuron = 0; neuron < network.hidden_size; ++neuron) {
        const f64 stm_activation = std::clamp<f64>(stm[neuron], 0.0, 1.0);
        const f64 ntm_activation = std::clamp<f64>(ntm[neuron], 0.0, 1.0);
        output += stm_activation * stm_activation * network.output_weights[neuron];
        output += ntm_activation * ntm_activation
                * network.output_weights[network.hidden_size + neuron];
    }

    const f64 centipawns = output * network.eval_scale;
    return static_cast<Score>(std::lround(std::clamp(centipawns, -29000.0, 29000.0)));
}

Score evaluate(const Position& pos) {
    Accumulator accumulator;
    refresh(pos, accumulator);
    return evaluate(pos, accumulator);
}

} // namespace NNUE