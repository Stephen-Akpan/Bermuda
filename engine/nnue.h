#pragma once

#include "position.h"
#include <string>

/* Platform headers for file mapping */
#ifdef _WIN32
#include <Windows.h>
#include <string>
#include <cstdint>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <immintrin.h> /* SIMD intrinsics (AVX2, SSE, etc.) */

/* OS-specific file descriptor and mapping types */
#ifdef _WIN32
using FD = HANDLE;
#define FD_ERR INVALID_HANDLE_VALUE
using map_t = HANDLE;
#else
typedef i32 FD;
#define FD_ERR -1
typedef size_t map_t;
#endif

/* Silence cast-qual warnings in GCC/Clang for packed weight readers */
#ifdef _MSC_VER
#else
#pragma GCC diagnostic ignored "-Wcast-qual"
#endif


/* Capability flags (used on MSVC builds; not required elsewhere) */
#ifdef _MSC_VER
enum : ui8 {
  use_avx2=1, use_sse41=1, use_sse3=1, use_sse2=1, use_sse=1, is_64_bit=1
};
#endif

/* Feature-transformer tiling and utility macros */
#define TILE_HEIGHT (NUM_REG * SIMD_WIDTH / 16)
#define CLAMP(a, b, c) ((a) < (b) ? (b) : (a) > (c) ? (c) : (a))

/* Optional embedded network file for non-MSVC builds */
#if !defined(_MSC_VER)
#define NNUE_EMBEDDED
#define NNUE_EVAL_FILE "network.nnue"
#endif

#define KING(c) ((c) ? B_KING : W_KING)
#define IS_KING(p) (((p) == W_KING) || ((p) == B_KING))

/* Default evaluation file name and version */
static std::string nnue_evalfile = "network.nnue";
inline const char* nnue_file = nnue_evalfile.c_str();
static constexpr ui32 nnue_version = 0x7AF32F16u;

enum NN_Pieces : ui8 {
  BLANK = 0, W_KING, W_QUEEN, W_ROOK, W_BISHOP, W_KNIGHT, W_PAWN, B_KING,
  B_QUEEN, B_ROOK, B_BISHOP, B_KNIGHT, B_PAWN
};

/* Piece-to-index base offsets for HalfKP feature indices */
enum : ui16 {
  ps_w_pawn = 1, ps_b_pawn = 1 * 64 + 1,
  ps_w_knight = 2 * 64 + 1, ps_b_knight = 3 * 64 + 1,
  ps_w_bishop = 4 * 64 + 1, ps_b_bishop = 5 * 64 + 1,
  ps_w_rook = 6 * 64 + 1, ps_b_rook = 7 * 64 + 1,
  ps_w_queen = 8 * 64 + 1, ps_b_queen = 9 * 64 + 1,
  ps_end = 10 * 64 + 1
};

/* Quantization and right-shift parameters used by affine kernels */
enum : ui8 {
  FV_SCALE = 16,
  SHIFT_ = 6
};

/* Dimensions for feature-transformer and dense layers */
enum : ui16 {
  K_HALF_DIMENSIONS = 256,        /* features per side after transform */
  FT_IN_DIMS = 64 * ps_end,         /* total input dims for HalfKP */
  FT_OUT_DIMS = K_HALF_DIMENSIONS * 2 /* both sides concatenated */
};

/* SIMD layout (AVX2) */
enum : ui16 {
  NUM_REG = 16,
  SIMD_WIDTH = 256
};

/* Serialized model offsets (transformer + dense network) */
enum {
  TRANSFMR_START = 3 * 4 + 177,
  NETWORK_START = TRANSFMR_START + 4 + 2 * 256 + 2 * 256 * 64 * 641
};

/*
 * Tracks up to 3 piece changes that occurred from the previous position.
 * Used to update the accumulator incrementally.
 */
struct DirtyPiece {
  i32 dirty_num;
  i32 pc[3];
  i32 from[3];
  i32 to[3];
};

/*
 * Per-color 256-lane accumulation buffer (int16) after HalfKP transform.
 * computed_accumulation == 1 means the buffers are valid for this position.
 */
struct Accumulator {
  alignas(64) int16_t accumulation[2][256];
  i32 computed_accumulation;
};

/*
 * nnue_data
 * ---------
 * Bundle of the accumulator and the last move's dirty-piece info.
 * We typically keep a small ring of these across plies.
 */
struct NNUE_Data {
  Accumulator accu;
  DirtyPiece dirty_piece;
};

/*
 *  * Minimal board description used by the NNUE front-ends.
 * The main engine may keep richer structures; only these fields are read.
 */
struct NNUE_Position {
  i32 player;         /* side to move: 0 = white, 1 = black */
  i32* pieces;        /* piece list: index 0..1 are kings, then others, 0-terminated */
  i32* squares;       /* parallel square list for pieces[] */
  NNUE_Data* nnue[3]; /* current and up to two previous nnue_data entries */
};

/* SIMD aliases and masks */
using vec16_t = __m256i; /* 16-bit packed */
using vec8_t = __m256i;  /* 8-bit packed */
using mask_t = ui32; /* bitmask chunk for activations */
using mask2_t = ui64;

/*
 * Small container for active/changed feature indices per color.
 */
struct IndexList {
  size_t size;
  unsigned values[30];
};

/* Vector helpers (AVX2) */
#define VEC_ADD_16(a, b) _mm256_add_epi16(a, b)
#define VEC_SUB_16(a, b) _mm256_sub_epi16(a, b)
#define VEC_PACKS(a, b)  _mm256_packs_epi16(a, b)
#define VEC_MASK_POS(a) \
    _mm256_movemask_epi8(_mm256_cmpgt_epi8(a, _mm256_setzero_si256()))

/* Network parameter storage (filled by init) */
inline i8 output_weights alignas(64)[1 * 32];
inline i32 hidden1_biases alignas(64)[32];
inline i32 hidden2_biases alignas(64)[32];
inline i32 output_biases[1];
inline i8 hidden1_weights alignas(64)[64 * 512];
inline i8 hidden2_weights alignas(64)[64 * 32];

/* Map piece and color to base index offset for HalfKP */
inline ui32 piece_to_index[2][14] = {
    { 0, 0, ps_w_queen, ps_w_rook, ps_w_bishop, ps_w_knight, ps_w_pawn, 0,
        ps_b_queen, ps_b_rook, ps_b_bishop, ps_b_knight, ps_b_pawn, 0 },
    { 0, 0, ps_b_queen, ps_b_rook, ps_b_bishop, ps_b_knight, ps_b_pawn, 0,
    ps_w_queen, ps_w_rook, ps_w_bishop, ps_w_knight, ps_w_pawn, 0 }
};

/*
 * Temporary buffers used during evaluation passes.
 */
struct NetData {
  alignas(64) i8 input[FT_OUT_DIMS]; /* HalfKP output (both sides) */
  i8 hidden1_out[32];
  int8_t hidden2_out[32];
};

/* Public API */
i32 nnue_evaluate_pos(const NNUE_Position* pos);
i32 nnue_init(const char* eval_file);
i32 nnue_evaluate(i32 player, i32* pieces, i32* squares);

/* File helpers for model loading */
FD open_file(const char* name);
void close_file(FD fd);
size_t file_size(FD fd);
const void* map_file(FD fd, map_t* map);
void unmap_file(const void* data, map_t map);

namespace NNUE {
  inline i32 load(const std::string& evl_file) {
    return nnue_init(evl_file.c_str());
  }
}

