#include "uci.h"
#include "movegen.h"
#include "perft.h"
#include "eval.h"
#include "nnue.h"
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>

namespace UCI {

// ============================================================================
// Global Options
// ============================================================================

Options options;

// ============================================================================
// Helpers
// ============================================================================

static std::string token_after(const std::string& line,
                                const std::string& keyword) {
    size_t pos = line.find(keyword);
    if (pos == std::string::npos) return "";

    pos += keyword.size();
    while (pos < line.size() && line[pos] == ' ') pos++;

    std::string result;
    while (pos < line.size() && line[pos] != ' ')
        result += line[pos++];

    return result;
}

static bool has_token(const std::string& line, const std::string& keyword) {
    size_t pos = line.find(keyword);
    if (pos == std::string::npos) return false;

    if (pos > 0 && line[pos - 1] != ' ') return false;
    size_t end = pos + keyword.size();
    if (end < line.size() && line[end] != ' ') return false;

    return true;
}

static i32 parse_int(const std::string& line,
                     const std::string& keyword,
                     i32 default_val = 0) {
    std::string tok = token_after(line, keyword);
    if (tok.empty()) return default_val;
    try { return std::stoi(tok); }
    catch (...) { return default_val; }
}

static Move find_move(const Position& pos, const std::string& uci) {
    MoveList list;
    MoveGen::generate_moves(pos, list);

    for (i32 i = 0; i < list.size(); i++) {
        if (list[i].to_string() == uci) {
            UndoInfo undo = const_cast<Position&>(pos).make_move(list[i]);
            bool legal = !pos.is_attacked_by(
                pos.king_square(flip(pos.side_to_move())),
                pos.side_to_move());
            const_cast<Position&>(pos).unmake_move(list[i], undo);

            if (legal) return list[i];
        }
    }
    return NULL_MOVE;
}

// ============================================================================
// Command: uci
// ============================================================================

static void cmd_uci() {
    std::cout << "id name Bermuda 0.0.1\n";
    std::cout << "id author AKPAN Stephen & DAUDA Mishael\n";
    std::cout << "\n";

    std::cout << "option name Hash type spin"
              << " default " << options.hash_mb
              << " min 1"
              << " max 65536"
              << "\n";

    std::cout << "option name MoveOverhead type spin"
              << " default " << options.move_overhead
              << " min 0"
              << " max 5000"
              << "\n";

    std::cout << "option name Clear Hash type button\n";

    std::cout << "option name EvalFile type string default network.nnue\n";

    std::cout << "\nuciok\n";
}

// ============================================================================
// Command: setoption
// ============================================================================

void cmd_setoption(const std::string& line) {
    size_t name_pos = line.find("name ");
    if (name_pos == std::string::npos) return;
    name_pos += 5;

    size_t value_pos = line.find(" value ");
    std::string name_str;
    std::string value_str;

    if (value_pos != std::string::npos) {
        name_str  = line.substr(name_pos, value_pos - name_pos);
        value_str = line.substr(value_pos + 7);
    } else {
        name_str = line.substr(name_pos);
    }

    while (!name_str.empty()  && name_str.back()  == ' ') name_str.pop_back();
    while (!value_str.empty() && value_str.back() == ' ') value_str.pop_back();

    std::string name_lower = name_str;
    std::transform(name_lower.begin(), name_lower.end(),
                   name_lower.begin(), ::tolower);

    if (name_lower == "hash") {
        try {
            i32 mb = std::stoi(value_str);
            mb = max(1, min(mb, 65536));
            options.hash_mb = mb;
            TT.resize(mb);
            std::cout << "info string Hash set to " << mb << " MB\n";
        } catch (...) {
            std::cerr << "info string Invalid Hash value: " << value_str << "\n";
        }
    }
    else if (name_lower == "moveoverhead") {
        try {
            i32 ms = std::stoi(value_str);
            ms = max(0, min(ms, 5000));
            options.move_overhead = ms;
        } catch (...) {}
    }
    else if (name_lower == "clear hash") {
        TT.clear();
        std::cout << "info string Hash cleared\n";
    }
    else if (name_lower == "evalfile") {
        Eval::init(value_str);
    }
    else {
        std::cerr << "info string Unknown option: " << name_str << "\n";
    }
}

// ============================================================================
// Command: position
// ============================================================================

void cmd_position(SearchInfo& info, const std::string& line) {
    info.rep_table.clear();

    std::string fen;

    if (has_token(line, "startpos")) {
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    } else {
        size_t fen_pos = line.find("fen ");
        if (fen_pos == std::string::npos) {
            std::cerr << "info string Invalid position command\n";
            return;
        }
        fen_pos += 4;

        size_t moves_pos = line.find(" moves ");
        if (moves_pos != std::string::npos)
            fen = line.substr(fen_pos, moves_pos - fen_pos);
        else
            fen = line.substr(fen_pos);
    }

    try {
        info.pos = Position::from_fen(fen);
    } catch (const std::exception& e) {
        std::cerr << "info string FEN parse error: " << e.what() << "\n";
        return;
    }

    info.rep_table.push(info.pos.hash());

    size_t moves_pos = line.find(" moves ");
    if (moves_pos == std::string::npos) return;

    std::istringstream ss(line.substr(moves_pos + 7));
    std::string move_str;

    while (ss >> move_str) {
        Move m = find_move(info.pos, move_str);

        if (m.is_null()) {
            std::cerr << "info string Illegal move: " << move_str << "\n";
            return;
        }

        info.pos.make_move(m);
        info.rep_table.push(info.pos.hash());
    }
}

// ============================================================================
// Command: go
// ============================================================================

void cmd_go(SearchInfo& info, const std::string& line) {
    SearchLimits limits;

    limits.infinite  = has_token(line, "infinite");
    limits.ponder    = has_token(line, "ponder");

    limits.depth     = parse_int(line, "depth",     MAX_PLY);
    limits.movetime  = parse_int(line, "movetime",  0);
    limits.wtime     = parse_int(line, "wtime",     0);
    limits.btime     = parse_int(line, "btime",     0);
    limits.winc      = parse_int(line, "winc",      0);
    limits.binc      = parse_int(line, "binc",      0);
    limits.movestogo = parse_int(line, "movestogo", 0);

    // Subtract move overhead from time limits before handing to the search.
    // Clamp to 1ms minimum — never let overhead push remaining time to zero
    // or negative, which would cause compute_allocated_time to misbehave.
    if (limits.wtime > 0)
        limits.wtime = max(1, limits.wtime - options.move_overhead);
    if (limits.btime > 0)
        limits.btime = max(1, limits.btime - options.move_overhead);

    // For movetime, overhead is already the only buffer — clamp to 0
    // since compute_allocated_time uses it directly as the hard limit.
    if (limits.movetime > 0)
        limits.movetime = max(0, limits.movetime - options.move_overhead);

    info.limits = limits;

    TT.new_search();
    Search::start(info);
}

// ============================================================================
// Command: ucinewgame
// ============================================================================

static void cmd_ucinewgame(SearchInfo& info) {
    info.rep_table.clear();
    info.pos.set_startpos();
    TT.clear();
    TT.new_search();
}

// ============================================================================
// Debug Commands
// ============================================================================

static void cmd_display(const SearchInfo& info) {
    info.pos.print();
}

static void cmd_perft(SearchInfo& info, const std::string& line) {
    i32 depth = parse_int(line, "perft", 1);
    Perft::divide(info.pos, depth);
}

// ============================================================================
// Main Loop
// ============================================================================

void loop() {
    BB::init();
    Zobrist::init();
    TT.resize(options.hash_mb);
    Eval::init("network.nnue");

    SearchInfo info;
    info.pos.set_startpos();

    std::string line;

    std::cout << "Bermuda Chess Engine\n";

    while (std::getline(std::cin, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n'
               || line.back() == ' '))
            line.pop_back();

        if (line.empty()) continue;

        if (line == "uci") {
            cmd_uci();
        }
        else if (line == "isready") {
            std::cout << "readyok\n";
        }
        else if (line == "ucinewgame") {
            cmd_ucinewgame(info);
        }
        else if (line.substr(0, 8) == "position") {
            cmd_position(info, line);
        }
        else if (line.substr(0, 2) == "go") {
            cmd_go(info, line);
        }
        else if (line == "stop") {
            info.stop.store(true);
        }
        else if (line.substr(0, 9) == "setoption") {
            cmd_setoption(line);
        }
        else if (line == "d") {
            cmd_display(info);
        }
        else if (line.substr(0, 5) == "perft") {
            cmd_perft(info, line);
        }
        else if (line == "quit" || line == "exit") {
            break;
        }
        else {
            std::cerr << "info string Unknown command: " << line << "\n";
        }

        std::cout.flush();
    }
}

} // namespace UCI
