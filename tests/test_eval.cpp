#include "catch_amalgamated.hpp"
#include "bitboard.h"
#include "eval.h"
#include "position.h"
#include "zobrist.h"

#include <cstdio>
#include <fstream>

namespace {

void initialize_evaluation_tables() {
    static const bool initialized = [] {
        BB::init();
        Zobrist::init();
        return true;
    }();
    (void)initialized;
}

Position position_from_fen(const std::string& fen) {
    initialize_evaluation_tables();
    return Position::from_fen(fen);
}

} // namespace

TEST_CASE("HCE evaluates the starting position as balanced", "[eval]") {
    Position pos = position_from_fen(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    REQUIRE(Eval::evaluate(pos) == 10);
}

TEST_CASE("HCE reports material advantage from the side to move", "[eval]") {
    Position whiteToMove = position_from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1");
    Position blackToMove = position_from_fen("4k3/8/8/8/8/8/8/R3K3 b - - 0 1");

    REQUIRE(Eval::evaluate(whiteToMove) > 400);
    REQUIRE(Eval::evaluate(whiteToMove) - Eval::evaluate(blackToMove) > 900);
}

TEST_CASE("HCE recognizes a passed pawn", "[eval]") {
    Position withPasser = position_from_fen("4k3/8/8/4P3/8/8/8/4K3 w - - 0 1");
    Position passedWithBlackPawn = position_from_fen("4k3/p7/8/4P3/8/8/8/4K3 w - - 0 1");
    Position blockedWithBlackPawn = position_from_fen("4k3/3p4/8/4P3/8/8/8/4K3 w - - 0 1");

    REQUIRE(Eval::evaluate(passedWithBlackPawn) > Eval::evaluate(blockedWithBlackPawn));
    REQUIRE(Eval::evaluate(withPasser) > 0);
}

TEST_CASE("HCE tapers middlegame and endgame values in the correct direction", "[eval]") {
    Position middlegame = position_from_fen(
        "rnbqkbnr/pppp1ppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1");
    Position endgame = position_from_fen(
        "4k3/8/8/8/4P3/8/8/4K3 w - - 0 1");

    Eval::Parameters mgMaterial;
    mgMaterial.piece_square = 0.0;
    mgMaterial.pawn_structure = 0.0;
    mgMaterial.passed_pawns = 0.0;
    mgMaterial.mobility = 0.0;
    mgMaterial.activity = 0.0;
    mgMaterial.threats = 0.0;
    mgMaterial.king_safety = 0.0;
    mgMaterial.tempo = 0.0;
    mgMaterial.pawn_eg = 0.0;

    Eval::Parameters egMaterial = mgMaterial;
    egMaterial.pawn_mg = 0.0;
    egMaterial.pawn_eg = 1.0;

    CHECK(Eval::evaluate(middlegame, mgMaterial)
          > Eval::evaluate(middlegame, egMaterial));
    CHECK(Eval::evaluate(endgame, mgMaterial)
          < Eval::evaluate(endgame, egMaterial));
}

TEST_CASE("HCE discounts opposite-colored bishop endgames", "[eval]") {
    Position sameColoredBishops = position_from_fen(
        "4k3/8/5b2/4P3/8/8/8/2B1K3 w - - 0 1");
    Position oppositeColoredBishops = position_from_fen(
        "2b1k3/8/8/4P3/8/8/8/2B1K3 w - - 0 1");

    Eval::Parameters materialOnly;
    materialOnly.piece_square = 0.0;
    materialOnly.pawn_structure = 0.0;
    materialOnly.passed_pawns = 0.0;
    materialOnly.mobility = 0.0;
    materialOnly.activity = 0.0;
    materialOnly.threats = 0.0;
    materialOnly.king_safety = 0.0;
    materialOnly.tempo = 0.0;

    CHECK(Eval::evaluate(oppositeColoredBishops, materialOnly)
          < Eval::evaluate(sameColoredBishops, materialOnly));
}

TEST_CASE("HCE evaluates the full promotion path of a passed pawn", "[eval]") {
    Position clearPath = position_from_fen(
        "r6k/8/8/3P4/8/8/8/4K3 w - - 0 1");
    Position attackedPath = position_from_fen(
        "3r3k/8/8/3P4/8/8/8/4K3 w - - 0 1");

    Eval::Parameters passerOnly;
    passerOnly.material = 0.0;
    passerOnly.piece_square = 0.0;
    passerOnly.pawn_structure = 0.0;
    passerOnly.mobility = 0.0;
    passerOnly.activity = 0.0;
    passerOnly.threats = 0.0;
    passerOnly.king_safety = 0.0;
    passerOnly.tempo = 0.0;

    CHECK(Eval::evaluate(clearPath, passerOnly)
          > Eval::evaluate(attackedPath, passerOnly));
}

TEST_CASE("HCE scores king-ring pressure through the king safety feature", "[eval]") {
    Position pressuredKing = position_from_fen(
        "6k1/8/4N3/5N2/8/8/8/R2QK3 w - - 0 1");
    Eval::Parameters withKingSafety;
    withKingSafety.material = 0.0;
    withKingSafety.piece_square = 0.0;
    withKingSafety.pawn_structure = 0.0;
    withKingSafety.passed_pawns = 0.0;
    withKingSafety.mobility = 0.0;
    withKingSafety.activity = 0.0;
    withKingSafety.threats = 0.0;
    withKingSafety.tempo = 0.0;

    Eval::Parameters withoutKingSafety = withKingSafety;
    withoutKingSafety.king_safety = 0.0;

    CHECK(Eval::evaluate(pressuredKing, withKingSafety)
          > Eval::evaluate(pressuredKing, withoutKingSafety));
}

TEST_CASE("HCE recognizes insufficient mating material", "[eval]") {
    Position kingsOnly = position_from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    Position kingAndKnight = position_from_fen("4k3/8/8/8/8/8/8/1N2K3 w - - 0 1");

    REQUIRE(Eval::evaluate(kingsOnly) == 0);
    REQUIRE(Eval::evaluate(kingAndKnight) == 0);
}

TEST_CASE("HCE applies parameter weights to feature groups", "[eval]") {
    Position rookAdvantage = position_from_fen(
        "4k3/8/8/8/8/8/8/R3K3 w - - 0 1");
    Eval::Parameters withoutMaterial;
    withoutMaterial.material = 0.0;

    REQUIRE(Eval::evaluate(rookAdvantage, withoutMaterial)
            < Eval::evaluate(rookAdvantage));
}

TEST_CASE("HCE tunes individual material values by game phase", "[eval]") {
    Position rookAdvantage = position_from_fen(
        "4k3/8/8/8/8/8/8/R3K3 w - - 0 1");
    Eval::Parameters lowerRookValue;
    lowerRookValue.rook_mg = 0.8;
    lowerRookValue.rook_eg = 0.8;

    REQUIRE(Eval::evaluate(rookAdvantage, lowerRookValue)
            < Eval::evaluate(rookAdvantage));
}

TEST_CASE("Loading tuner parameters does not change active evaluation weights", "[eval]") {
    const std::string path = "bermuda-eval-parameters-load-test.txt";
    {
        std::ofstream output(path);
        REQUIRE(output.good());
        output << "material 0.5\n"
               << "piece_square 1\n"
               << "pawn_structure 1\n"
               << "passed_pawns 1\n"
               << "mobility 1\n"
               << "activity 1\n"
               << "threats 1\n"
               << "king_safety 1\n"
               << "tempo 1\n"
               << "knight_mg 0.9\n";
    }

    Eval::reset_parameters();
    const Eval::Parameters active_before = Eval::get_parameters();
    Eval::Parameters loaded;
    std::string error;
    const bool success = Eval::load_parameters(path, loaded, error);
    std::remove(path.c_str());

    REQUIRE(success);
    CHECK(loaded.material == 0.5);
    CHECK(loaded.knight_mg == 0.9);
    CHECK(Eval::get_parameters().material == active_before.material);
    CHECK(Eval::get_parameters().knight_mg == active_before.knight_mg);
}

TEST_CASE("HCE values an attacked undefended piece above a pawn-defended one", "[eval]") {
    Position hangingKnight = position_from_fen(
        "7k/8/8/3pn3/8/8/8/K3R3 w - - 0 1");
    Position pawnDefendedKnight = position_from_fen(
        "7k/8/3p4/4n3/8/8/8/K3R3 w - - 0 1");

    REQUIRE(Eval::evaluate(hangingKnight) > Eval::evaluate(pawnDefendedKnight));
}

TEST_CASE("HCE rewards a safe pawn attack on an enemy piece", "[eval]") {
    Position pawnAttacksKnight = position_from_fen(
        "7k/8/8/3n4/2P5/8/8/K7 w - - 0 1");
    Position pawnDoesNotAttackKnight = position_from_fen(
        "7k/8/8/3n4/3P4/8/8/K7 w - - 0 1");

    Eval::Parameters threatsOnly;
    threatsOnly.material = 0.0;
    threatsOnly.piece_square = 0.0;
    threatsOnly.pawn_structure = 0.0;
    threatsOnly.passed_pawns = 0.0;
    threatsOnly.mobility = 0.0;
    threatsOnly.activity = 0.0;
    threatsOnly.king_safety = 0.0;
    threatsOnly.tempo = 0.0;

    CHECK(Eval::evaluate(pawnAttacksKnight, threatsOnly)
          > Eval::evaluate(pawnDoesNotAttackKnight, threatsOnly));
}

TEST_CASE("HCE values bishop-pair material imbalance", "[eval]") {
    Position bishopPair = position_from_fen(
        "7k/8/8/8/8/8/8/1BB1K3 w - - 0 1");
    Position bishopAndKnight = position_from_fen(
        "7k/8/8/8/8/8/8/1BN1K3 w - - 0 1");

    Eval::Parameters imbalanceOnly;
    imbalanceOnly.material = 1.0;
    imbalanceOnly.piece_square = 0.0;
    imbalanceOnly.pawn_structure = 0.0;
    imbalanceOnly.passed_pawns = 0.0;
    imbalanceOnly.mobility = 0.0;
    imbalanceOnly.activity = 0.0;
    imbalanceOnly.threats = 0.0;
    imbalanceOnly.king_safety = 0.0;
    imbalanceOnly.tempo = 0.0;

    CHECK(Eval::evaluate(bishopPair, imbalanceOnly)
          > Eval::evaluate(bishopAndKnight, imbalanceOnly));
}

TEST_CASE("HCE gives a minor piece credit for reaching a safe outpost", "[eval]") {
    Position canReachOutpost = position_from_fen(
        "7k/8/8/8/8/8/P1N5/4K3 w - - 0 1");
    Position noReachableOutpost = position_from_fen(
        "7k/8/8/8/8/8/P7/2N1K3 w - - 0 1");

    Eval::Parameters activityOnly;
    activityOnly.material = 0.0;
    activityOnly.piece_square = 0.0;
    activityOnly.pawn_structure = 0.0;
    activityOnly.passed_pawns = 0.0;
    activityOnly.mobility = 0.0;
    activityOnly.threats = 0.0;
    activityOnly.king_safety = 0.0;
    activityOnly.tempo = 0.0;

    CHECK(Eval::evaluate(canReachOutpost, activityOnly)
          > Eval::evaluate(noReachableOutpost, activityOnly));
}

TEST_CASE("HCE space evaluation is symmetric by color", "[eval]") {
    Position whiteSpace = position_from_fen(
        "rnbqkbnr/ppp2ppp/8/3pp3/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 1");
    Position blackSpace = position_from_fen(
        "rnbqkbnr/ppp2ppp/8/3pp3/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 1");

    CHECK(Eval::evaluate(whiteSpace) - 10
          == -(Eval::evaluate(blackSpace) - 10));
}

TEST_CASE("HCE recognizes a clear pawn-only promotion race", "[eval]") {
    Position clearRace = position_from_fen(
        "7k/8/1P6/8/8/8/8/K7 w - - 0 1");
    Position blockedRace = position_from_fen(
        "1k6/8/1P6/8/8/8/8/K7 w - - 0 1");

    Eval::Parameters passerOnly;
    passerOnly.material = 0.0;
    passerOnly.piece_square = 0.0;
    passerOnly.pawn_structure = 0.0;
    passerOnly.mobility = 0.0;
    passerOnly.activity = 0.0;
    passerOnly.threats = 0.0;
    passerOnly.king_safety = 0.0;
    passerOnly.tempo = 0.0;

    CHECK(Eval::evaluate(clearRace, passerOnly)
          > Eval::evaluate(blockedRace, passerOnly));
}

TEST_CASE("HCE penalizes a king on a pawnless flank", "[eval]") {
    Position pawnlessFlank = position_from_fen(
        "3k4/3p4/8/8/8/8/P7/R5K1 w - - 0 1");
    Eval::Parameters kingSafetyOnly;
    kingSafetyOnly.material = 0.0;
    kingSafetyOnly.piece_square = 0.0;
    kingSafetyOnly.pawn_structure = 0.0;
    kingSafetyOnly.passed_pawns = 0.0;
    kingSafetyOnly.mobility = 0.0;
    kingSafetyOnly.activity = 0.0;
    kingSafetyOnly.threats = 0.0;
    kingSafetyOnly.tempo = 0.0;

    Eval::Parameters withoutKingSafety = kingSafetyOnly;
    withoutKingSafety.king_safety = 0.0;

    CHECK(Eval::evaluate(pawnlessFlank, kingSafetyOnly)
          < Eval::evaluate(pawnlessFlank, withoutKingSafety));
}
