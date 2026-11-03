#include <bermuda_tuner/dataset.h>
#include "bitboard.h"
#include "movegen.h"
#include "position.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
namespace Bermuda::Tuner {
namespace {
bool parse_result(const std::string& token, double& result) {
    if (token == "1-0" || token == "1") {
        result = 1.0;
        return true;
    }
    if (token == "1/2-1/2" || token == "0.5") {
        result = 0.5;
        return true;
    }
    if (token == "0-1" || token == "0") {
        result = 0.0;
        return true;
    }
    return false;
}

struct PgnEvent {
    bool is_tag;
    std::string text;
    i32 line;
};

bool tokenize_pgn(const std::string& path, std::vector<PgnEvent>& events,
                  std::string& error) {
    std::ifstream input(path);
    if (!input) {
        error = "unable to open PGN: " + path;
        return false;
    }
    const std::string source((std::istreambuf_iterator<char>(input)),
                             std::istreambuf_iterator<char>());
    if (!input.eof() && input.fail()) {
        error = "error reading PGN: " + path;
        return false;
    }

    std::string word;
    i32 line = 1;
    i32 word_line = 1;
    i32 variation_depth = 0;
    auto flush_word = [&] {
        if (!word.empty()) {
            events.push_back({ false, word, word_line });
            word.clear();
        }
    };

    for (std::size_t i = 0; i < source.size(); ++i) {
        const char ch = source[i];
        if (ch == '\n')
            ++line;

        if (ch == '{') {
            flush_word();
            const i32 comment_line = line;
            bool closed = false;
            while (++i < source.size()) {
                if (source[i] == '}') {
                    closed = true;
                    break;
                }
                if (source[i] == '\n')
                    ++line;
            }
            if (!closed) {
                error = "unterminated PGN comment near line "
                      + std::to_string(comment_line);
                return false;
            }
            continue;
        }
        if (ch == ';') {
            flush_word();
            while (i + 1 < source.size() && source[i + 1] != '\n')
                ++i;
            continue;
        }
        if (variation_depth > 0) {
            if (ch == '(')
                ++variation_depth;
            else if (ch == ')')
                --variation_depth;
            continue;
        }
        if (ch == '(') {
            flush_word();
            variation_depth = 1;
            continue;
        }
        if (ch == ')') {
            error = "unmatched PGN variation close on line "
                  + std::to_string(line);
            return false;
        }
        if (ch == '[') {
            flush_word();
            const i32 tag_line = line;
            std::string tag;
            bool quoted = false;
            bool escape_next = false;
            bool closed = false;
            for (++i; i < source.size(); ++i) {
                if (source[i] == '\n')
                    ++line;
                if (source[i] == '"' && !escape_next)
                    quoted = !quoted;
                if (quoted && source[i] == '\\' && !escape_next)
                    escape_next = true;
                else
                    escape_next = false;
                if (source[i] == ']' && !quoted) {
                    closed = true;
                    break;
                }
                tag += source[i];
            }
            if (!closed) {
                error = "unterminated PGN tag near line "
                      + std::to_string(tag_line);
                return false;
            }
            events.push_back({ true, tag, tag_line });
            continue;
        }
        if (std::isspace(static_cast<unsigned char>(ch))) {
            flush_word();
            continue;
        }
        if (word.empty())
            word_line = line;
        word += ch;
    }
    flush_word();
    if (variation_depth != 0) {
        error = "unterminated PGN variation near end of file";
        return false;
    }
    error.clear();
    return true;
}

bool parse_tag(const std::string& text, std::string& name,
               std::string& value) {
    std::size_t offset = 0;
    while (offset < text.size()
           && std::isspace(static_cast<unsigned char>(text[offset])))
        ++offset;
    const std::size_t name_start = offset;
    while (offset < text.size()
           && !std::isspace(static_cast<unsigned char>(text[offset])))
        ++offset;
    if (offset == name_start)
        return false;
    name = text.substr(name_start, offset - name_start);
    while (offset < text.size()
           && std::isspace(static_cast<unsigned char>(text[offset])))
        ++offset;
    if (offset >= text.size() || text[offset] != '"')
        return false;
    ++offset;
    bool closed = false;
    while (offset < text.size()) {
        const char ch = text[offset++];
        if (ch == '\\' && offset < text.size()) {
            const char escaped = text[offset++];
            if (escaped == '"' || escaped == '\\')
                value += escaped;
            else {
                value += ch;
                value += escaped;
            }
        } else if (ch == '"') {
            closed = true;
            break;
        } else {
            value += ch;
        }
    }
    if (!closed)
        return false;
    while (offset < text.size()
           && std::isspace(static_cast<unsigned char>(text[offset])))
        ++offset;
    return offset == text.size();
}

std::string tag_value(const std::map<std::string, std::string>& tags,
                      const std::string& key) {
    const auto found = tags.find(key);
    return found == tags.end() ? std::string{} : found->second;
}

void assign_metadata_fields(GameMetadata& game) {
    game.event = tag_value(game.tags, "Event");
    game.site = tag_value(game.tags, "Site");
    game.date = tag_value(game.tags, "Date");
    game.round = tag_value(game.tags, "Round");
    game.white = tag_value(game.tags, "White");
    game.black = tag_value(game.tags, "Black");
    game.result = tag_value(game.tags, "Result");
    game.eco = tag_value(game.tags, "ECO");
    game.opening = tag_value(game.tags, "Opening");
    game.time_control = tag_value(game.tags, "TimeControl");
    game.termination = tag_value(game.tags, "Termination");
    game.variant = tag_value(game.tags, "Variant");
}

bool parse_san_move(Position& position, std::string san, Move& selected) {
    while (!san.empty() && (san.back() == '!' || san.back() == '?'
                           || san.back() == '+' || san.back() == '#'))
        san.pop_back();
    std::replace(san.begin(), san.end(), '0', 'O');

    MoveList moves;
    MoveGen::generate_moves(position, moves);
    i32 matches = 0;
    if (san == "O-O" || san == "O-O-O") {
        const ui8 destination_file = san == "O-O" ? 6 : 2;
        for (i32 i = 0; i < moves.size(); ++i) {
            if (moves[i].is_castling() && file(moves[i].to()) == destination_file) {
                const UndoInfo undo = position.make_move(moves[i]);
                const bool legal = !position.is_attacked_by(
                    position.king_square(flip(position.side_to_move())),
                    position.side_to_move());
                position.unmake_move(moves[i], undo);
                if (legal) {
                    selected = moves[i];
                    ++matches;
                }
            }
        }
        return matches == 1;
    }

    PieceType moving_type = PAWN;
    if (!san.empty()) {
        switch (san.front()) {
        case 'K': moving_type = KING; break;
        case 'Q': moving_type = QUEEN; break;
        case 'R': moving_type = ROOK; break;
        case 'B': moving_type = BISHOP; break;
        case 'N': moving_type = KNIGHT; break;
        default: break;
        }
    }
    if (moving_type != PAWN)
        san.erase(san.begin());

    PromoPiece promotion = PROMO_QUEEN;
    bool is_promotion = false;
    const std::size_t promotion_marker = san.find('=');
    if (promotion_marker != std::string::npos) {
        if (promotion_marker + 1 >= san.size())
            return false;
        is_promotion = true;
        switch (san[promotion_marker + 1]) {
        case 'Q': case 'q': promotion = PROMO_QUEEN; break;
        case 'R': case 'r': promotion = PROMO_ROOK; break;
        case 'B': case 'b': promotion = PROMO_BISHOP; break;
        case 'N': case 'n': promotion = PROMO_KNIGHT; break;
        default: return false;
        }
        san.erase(promotion_marker);
    }

    if (san.size() < 2)
        return false;
    const std::size_t target_offset = san.size() - 2;
    const char target_file = san[target_offset];
    const char target_rank = san[target_offset + 1];
    if (target_file < 'a' || target_file > 'h'
        || target_rank < '1' || target_rank > '8')
        return false;
    const Square target = make_square(
        static_cast<ui8>(target_file - 'a'),
        static_cast<ui8>(target_rank - '1'));
    std::string disambiguation = san.substr(0, target_offset);
    const bool is_capture = disambiguation.find('x') != std::string::npos;
    disambiguation.erase(
        std::remove(disambiguation.begin(), disambiguation.end(), 'x'),
        disambiguation.end());

    for (i32 i = 0; i < moves.size(); ++i) {
        const Move move = moves[i];
        if (move.to() != target)
            continue;
        const Piece moving = position.piece_on(move.from());
        if (moving == NO_PIECE || type_of(moving) != moving_type)
            continue;
        if (move.is_promotion() != is_promotion)
            continue;
        if (is_promotion && move.promo() != promotion)
            continue;

        const bool captures = move.is_en_passant()
            || position.piece_on(move.to()) != NO_PIECE;
        if (captures != is_capture)
            continue;

        bool matches_disambiguation = true;
        for (const char part : disambiguation) {
            if (part >= 'a' && part <= 'h')
                matches_disambiguation &= file(move.from()) == part - 'a';
            else if (part >= '1' && part <= '8')
                matches_disambiguation &= rank(move.from()) == part - '1';
            else
                matches_disambiguation = false;
        }
        if (!matches_disambiguation)
            continue;

        const UndoInfo undo = position.make_move(move);
        const bool legal = !position.is_attacked_by(
            position.king_square(flip(position.side_to_move())),
            position.side_to_move());
        position.unmake_move(move, undo);
        if (legal) {
            selected = move;
            ++matches;
        }
    }
    return matches == 1;
}

bool load_pgn_dataset_impl(const std::string& path, Dataset& dataset,
                           ImportStatistics& statistics,
                           const std::vector<std::pair<std::string, std::string>>&
                               metadata_filters,
                           std::string& error) {
    std::vector<PgnEvent> events;
    if (!tokenize_pgn(path, events, error)) {
        statistics.games_rejected = 1;
        return false;
    }

    Position position;
    position.set_startpos();
    std::vector<std::string> game_positions;
    std::map<std::string, std::string> game_tags;
    bool game_open = false;

    auto finish_game = [&](const std::string& move_result, i32 line) {
        if (!game_open)
            return true;
        ++statistics.games_encountered;
        double result = 0.0;
        double tagged_result = 0.0;
        const std::string header_result = tag_value(game_tags, "Result");
        const bool has_move_result = parse_result(move_result, result);
        const bool has_tagged_result = parse_result(header_result, tagged_result);
        if (has_move_result && has_tagged_result && result != tagged_result) {
            ++statistics.games_rejected;
            error = "PGN result tag disagrees with movetext near line "
                  + std::to_string(line);
            return false;
        }
        if (game_positions.empty()) {
            ++statistics.games_skipped;
            ++statistics.skip_reasons["no_positions"];
            game_tags.clear();
            position.set_startpos();
            game_open = false;
            return true;
        }
        if (!has_move_result && !has_tagged_result) {
            game_positions.clear();
            game_tags.clear();
            position.set_startpos();
            ++statistics.games_skipped;
            ++statistics.skip_reasons["unknown_result"];
            game_open = false;
            return true;
        }
        if (!has_move_result)
            result = tagged_result;

        GameMetadata metadata;
        metadata.id = "pgn-" + std::to_string(statistics.games_encountered);
        metadata.tags = game_tags;
        metadata.source = path;
        assign_metadata_fields(metadata);
        if (has_move_result)
            metadata.result = move_result;
        const std::size_t game_index = dataset.game_ids.size();
        dataset.game_ids.push_back(metadata.id);
        dataset.games.push_back(std::move(metadata));
        for (const std::string& fen : game_positions) {
            try {
                dataset.samples.push_back({
                    Position::from_fen(fen), result, game_index
                });
            } catch (const std::exception& exception) {
                ++statistics.games_rejected;
                error = "unable to restore PGN position: "
                      + std::string(exception.what());
                return false;
            }
        }
        ++statistics.games_accepted;
        game_positions.clear();
        game_tags.clear();
        position.set_startpos();
        game_open = false;
        return true;
    };

    for (const PgnEvent& event : events) {
        if (event.is_tag) {
            std::string name;
            std::string value;
            if (!parse_tag(event.text, name, value)) {
                error = "malformed PGN tag on line "
                      + std::to_string(event.line);
                ++statistics.games_rejected;
                return false;
            }
            if (name == "Event" && game_open && !game_positions.empty()) {
                if (!finish_game("", event.line))
                    return false;
            }
            game_open = true;
            game_tags[name] = value;
            if (name == "FEN") {
                if (!game_positions.empty()) {
                    error = "FEN tag appears after moves near line "
                          + std::to_string(event.line);
                    ++statistics.games_rejected;
                    return false;
                }
                try {
                    position = Position::from_fen(value);
                } catch (const std::exception& exception) {
                    error = "invalid PGN FEN on line "
                          + std::to_string(event.line) + ": "
                          + exception.what();
                    ++statistics.games_rejected;
                    return false;
                }
            }
            continue;
        }

        const std::string& token = event.text;
        if (token == "1-0" || token == "0-1" || token == "1/2-1/2"
            || token == "*") {
            if (!finish_game(token, event.line))
                return false;
            continue;
        }
        if (token.empty() || token.front() == '$')
            continue;

        std::string san = token;
        const std::size_t nag = san.find('$');
        if (nag != std::string::npos)
            san.erase(nag);
        std::size_t offset = 0;
        while (offset < san.size()
               && (std::isdigit(static_cast<unsigned char>(san[offset]))
                   || san[offset] == '.'))
            ++offset;
        san.erase(0, offset);
        if (san.empty() || san == "e.p." || san == "ep")
            continue;

        game_open = true;
        game_positions.push_back(position.to_fen());
        Move move;
        if (!parse_san_move(position, san, move)) {
            ++statistics.games_rejected;
            error = "unable to parse legal SAN move '" + token
                  + "' on PGN line " + std::to_string(event.line);
            return false;
        }
        position.make_move(move);
    }
    if (!finish_game("", 0))
        return false;
    if (dataset.samples.empty() || dataset.game_ids.size() < 2) {
        error = "PGN needs at least two completed games with known results";
        return false;
    }
    dataset.source_description = path;
    statistics.positions_generated = dataset.samples.size();
    statistics.games_accepted = dataset.game_ids.size();
    statistics.games_skipped = statistics.games_encountered
                             - statistics.games_accepted
                             - statistics.games_rejected;
    if (!filter_dataset_metadata(dataset, metadata_filters, statistics, error))
        return false;
    if (dataset.game_ids.size() < 2) {
        error = "PGN filters must leave at least two completed games";
        return false;
    }
    error.clear();
    return true;
}


} // namespace
bool load_pgn_dataset(const std::string& path, Dataset& dataset, std::string& error) {
    ImportStatistics statistics;
    return load_pgn_dataset(path, dataset, statistics, {}, error);
}
bool load_pgn_dataset(
    const std::string& path, Dataset& dataset, ImportStatistics& statistics,
    const std::vector<std::pair<std::string, std::string>>& metadata_filters,
    std::string& error) {
    Dataset candidate;
    if (!load_pgn_dataset_impl(path, candidate, statistics, metadata_filters,
                               error)) {
        error = path + ": " + error;
        return false;
    }
    if (!candidate.validate(error))
        return false;
    statistics.positions_generated = candidate.samples.size();
    dataset = std::move(candidate);
    return true;
}
} // namespace Bermuda::Tuner
