#include "uci.h"
#include "movegen.h"
#include "perft.h"
#include "eval.h"
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

// Extract a token after a keyword in a UCI command line.
// Returns empty string if keyword not found.
static std::string token_after(const std::string& line,
                                const std::string& keyword) {
    size_t pos = line.find(keyword);
    if (pos == std::string::npos) return "";

    pos += keyword.size();

    // Skip whitespace
    while (pos < line.size() && line[pos] == ' ') pos++;

    // Read until next space
    std::string result;
    while (pos < line.size() && line[pos] != ' ') {
        result += line[pos++];
    }
    return result;
}

// Returns true if a keyword appears in the line
static bool has_token(const std::string& line, const std::string& keyword) {
    // Find keyword as a whole word (not substring)
    size_t pos = line.find(keyword);
    if (pos == std::string::npos) return false;

    // Check it's not part of a longer word
    if (pos > 0 && line[pos - 1] != ' ') return false;
    size_t end = pos + keyword.size();
    if (end < line.size() && line[end] != ' ') return false;

    return true;
}

// Parse an integer token after a keyword, returning default_val if not found
static int parse_int(const std::string& line,
                     const std::string& keyword,
                     int default_val = 0) {
    std::string tok = token_after(line, keyword);
    if (tok.empty()) return default_val;
    try { return std::stoi(tok); }
    catch (...) { return default_val; }
}

// Find a UCI move string in the position's legal move list
static Move find_move(const Position& pos, const std::string& uci) {
    MoveList list;
    MoveGen::generate_moves(pos, list);

    for (int i = 0; i < list.size(); i++) {
        if (list[i].to_string() == uci) {
            // Verify legality
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
    std::cout << "id name Bermuda\n";
    std::cout << "id author AKPAN Stephen\n";
    std::cout << "\n";

    // Configurable options
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

    std::cout << "\nuciok\n";
}

// ============================================================================
// Command: setoption
// ============================================================================

void cmd_setoption(const std::string& line) {
    // Format: setoption name <id> [value <x>]
    // Extract name
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

    // Trim whitespace
    while (!name_str.empty()  && name_str.back()  == ' ') name_str.pop_back();
    while (!value_str.empty() && value_str.back() == ' ') value_str.pop_back();

    // Normalise to lowercase for case-insensitive comparison
    std::string name_lower = name_str;
    std::transform(name_lower.begin(), name_lower.end(),
                   name_lower.begin(), ::tolower);

    if (name_lower == "hash") {
        try {
            int mb = std::stoi(value_str);
            mb = std::max(1, std::min(mb, 65536));
            options.hash_mb = mb;
            TT.resize(mb);
            std::cout << "info string Hash set to " << mb << " MB\n";
        } catch (...) {
            std::cerr << "info string Invalid Hash value: " << value_str << "\n";
        }
    }
    else if (name_lower == "moveoverhead") {
        try {
            int ms = std::stoi(value_str);
            ms = std::max(0, std::min(ms, 5000));
            options.move_overhead = ms;
        } catch (...) {}
    }
    else if (name_lower == "clear hash") {
        TT.clear();
        std::cout << "info string Hash cleared\n";
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
        // Extract FEN string between "fen" and "moves" (or end of line)
        size_t fen_pos = line.find("fen ");
        if (fen_pos == std::string::npos) {
            std::cerr << "info string Invalid position command\n";
            return;
        }
        fen_pos += 4;

        size_t moves_pos = line.find(" moves ");
        if (moves_pos != std::string::npos) {
            fen = line.substr(fen_pos, moves_pos - fen_pos);
        } else {
            fen = line.substr(fen_pos);
        }
    }

    // Set up the position from FEN
    try {
        info.pos = Position::from_fen(fen);
    } catch (const std::exception& e) {
        std::cerr << "info string FEN parse error: " << e.what() << "\n";
        return;
    }

    // Push the initial position hash
    info.rep_table.push(info.pos.hash());

    // Apply moves if present
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

    // Apply move overhead to time limits
    if (limits.wtime > 0) limits.wtime -= options.move_overhead;
    if (limits.btime > 0) limits.btime -= options.move_overhead;
    if (limits.movetime > 0) limits.movetime -= options.move_overhead;

    // Clamp — don't let overhead push time negative
    limits.wtime    = std::max(limits.wtime,    1);
    limits.btime    = std::max(limits.btime,    1);
    limits.movetime = std::max(limits.movetime, 0);

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

static void cmd_eval(const SearchInfo& info) {
    int score = Eval::evaluate(info.pos);
    std::cout << "info string Static eval: " << score
              << " cp (side to move perspective)\n";
    std::cout << "info string White perspective: "
              << (info.pos.side_to_move() == WHITE ? score : -score)
              << " cp\n";
}

static void cmd_perft(SearchInfo& info, const std::string& line) {
    int depth = parse_int(line, "perft", 1);
    Perft::divide(info.pos, depth);
}

// ============================================================================
// Main Loop
// ============================================================================

void loop() {
    // Initialise engine
    BB::init();
    Zobrist::init();
    TT.resize(options.hash_mb);

    SearchInfo info;
    info.pos.set_startpos();

    std::string line;

    std::cout << "Bermuda Chess Engine\n";

    while (std::getline(std::cin, line)) {
        // Trim trailing whitespace
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n'
               || line.back() == ' '))
            line.pop_back();

        if (line.empty()) continue;

        // Dispatch commands
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
        else if (line == "eval") {
            cmd_eval(info);
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

        // Flush after every response
        std::cout.flush();
    }
}

} // namespace UCI
