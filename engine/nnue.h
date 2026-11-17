#pragma once

#include "position.h"
#include <string>

namespace NNUE {

constexpr ui32 HIDDEN_SIZE = 64;

struct Accumulator {
	f32 values[2][HIDDEN_SIZE] = {};
	ui64 generation = 0;
	bool valid = false;
};

bool load(const std::string& path, std::string& error);
void clear();
bool is_loaded();
f32 eval_scale();
void refresh(const Position& pos, Accumulator& accumulator);
void update_for_move(const Position& pos, Move move,
					 const Accumulator& parent, Accumulator& child);
f32 predict_white_probability(const Position& pos,
							 const Accumulator& accumulator);
f32 predict_white_probability(const Position& pos);
Score evaluate(const Position& pos, const Accumulator& accumulator);
Score evaluate(const Position& pos);

} // namespace NNUE