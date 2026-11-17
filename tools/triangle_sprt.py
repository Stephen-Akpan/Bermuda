#!/usr/bin/env python3
"""Calculate a Fishtest-style pentanomial SPRT from TriangleTest pair PGNs."""

import argparse
import base64
import binascii
import json
import math
import re
import sys
from pathlib import Path


TAG_PATTERN = re.compile(r'^\[(\w+)\s+"((?:\\.|[^"\\])*)"\]\s*$')
RESULTS = {"1-0", "0-1", "1/2-1/2"}
EPSILON = 1e-3


def _unescape_tag(value):
    return re.sub(r'\\(["\\])', r'\1', value)


def read_games(path):
    return [tags for tags, _ in _read_pgn_records(path)]


def _read_pgn_records(path):
    games = []
    current = {}
    in_headers = False
    current_lines = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        match = TAG_PATTERN.match(line.strip())
        if match:
            if current and not in_headers:
                games.append((current, "\n".join(current_lines).rstrip() + "\n"))
                current = {}
                current_lines = []
            in_headers = True
            tag = match.group(1)
            if tag in current:
                raise ValueError(f"{path}:{line_number}: duplicate PGN tag {tag}")
            current[tag] = _unescape_tag(match.group(2))
            current_lines.append(line)
        elif not line.strip():
            if in_headers:
                in_headers = False
            current_lines.append(line)
        elif in_headers:
            raise ValueError(f"{path}:{line_number}: invalid PGN tag line")
        else:
            current_lines.append(line)

    if current:
        games.append((current, "\n".join(current_lines).rstrip() + "\n"))
    return games


def _candidate_score(game, path, game_number):
    white = game.get("White")
    black = game.get("Black")
    result = game.get("Result")
    if {white, black} != {"Baseline", "Candidate"}:
        raise ValueError(
            f"{path}: game {game_number} must contain Baseline and Candidate"
        )
    if result not in RESULTS:
        raise ValueError(
            f"{path}: game {game_number} has missing or unfinished Result"
        )

    if result == "1/2-1/2":
        return 0.5
    candidate_is_white = white == "Candidate"
    candidate_won = (result == "1-0") == candidate_is_white
    return 1.0 if candidate_won else 0.0


def _validate_pair(games, label):
    if len(games) != 2:
        raise ValueError(f"{label}: expected exactly two games, found {len(games)}")

    for game_number, game in enumerate(games, start=1):
        _candidate_score(game, label, game_number)

    first_fen = games[0].get("FEN")
    second_fen = games[1].get("FEN")
    if first_fen != second_fen:
        raise ValueError(f"{label}: paired games do not have matching FEN tags")

    first_candidate_white = games[0]["White"] == "Candidate"
    second_candidate_white = games[1]["White"] == "Candidate"
    if first_candidate_white == second_candidate_white:
        raise ValueError(f"{label}: Candidate did not switch colors between games")

    points = sum(
        _candidate_score(game, label, game_number)
        for game_number, game in enumerate(games, start=1)
    )
    return int(points * 2)


def read_pair(path):
    return _validate_pair(read_games(path), path)


def _move_tokens(movetext):
    text = re.sub(r"\[[^\]]*\]", " ", movetext)
    text = re.sub(r"\{[^}]*\}|;[^\r\n]*", " ", text)
    while re.search(r"\([^()]*\)", text):
        text = re.sub(r"\([^()]*\)", " ", text)
    text = re.sub(r"\$\d+", " ", text)
    text = re.sub(r"\d+\.(?:\.\.)?", " ", text)
    return [
        token.rstrip("!?+#")
        for token in text.split()
        if token not in {"1-0", "0-1", "1/2-1/2", "*", "e.p."}
    ]


def _validate_opening_prefix(records, opening_plies, label):
    if opening_plies is None:
        return
    if opening_plies < 1:
        raise ValueError("opening plies must be positive")
    first_moves = _move_tokens(records[0][1])
    second_moves = _move_tokens(records[1][1])
    if len(first_moves) < opening_plies or len(second_moves) < opening_plies:
        raise ValueError(f"{label}: paired games are shorter than the opening prefix")
    if first_moves[:opening_plies] != second_moves[:opening_plies]:
        raise ValueError(f"{label}: paired games do not share the configured opening prefix")


def _count_movetext_plies(movetext):
    text = re.sub(r"\{[^}]*\}|;[^\r\n]*|\([^()]*\)", " ", movetext)
    text = re.sub(r"\$\d+", " ", text)
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"\d+\.(?:\.\.)?", " ", text)
    return sum(token not in {"1-0", "0-1", "1/2-1/2", "*"} for token in text.split())


def _termination_category(termination):
    if not termination:
        return "unavailable"
    normalized = termination.lower()
    if "time" in normalized and "forfeit" in normalized:
        return "time_forfeit"
    if "illegal" in normalized:
        return "illegal_move"
    if any(word in normalized for word in ("crash", "disconnect", "abandon")):
        return "crash_or_disconnect"
    if any(word in normalized for word in ("adjudicat", "adjudication")):
        return "adjudication"
    if any(word in normalized for word in ("normal", "checkmate", "resign", "stalemate", "draw")):
        return "normal_or_adjudicated"
    return "other"


def analyze_fixed_matches(matches_dir, completed_pairs, metadata=None):
    """Analyze a fixed paired sample without applying SPRT early stopping."""
    matches_dir = Path(matches_dir)
    if completed_pairs < 1:
        raise ValueError("a fixed match report requires at least one completed pair")

    candidate_wins = candidate_draws = candidate_losses = 0
    candidate_white_points = candidate_white_games = 0.0
    candidate_black_points = candidate_black_games = 0.0
    pair_scores = []
    ply_counts = []
    duration_seconds = []
    termination_counts = {}
    pair_bins = [0, 0, 0, 0, 0]
    opening_plies = None
    if metadata:
        opening_plies = metadata.get("experiment", {}).get("openingPlies")

    for pair_number in range(1, completed_pairs + 1):
        path = matches_dir / f"pair-{pair_number:05d}.pgn"
        records = _read_pgn_records(path)
        pair_score = _validate_pair([tags for tags, _ in records], path)
        _validate_opening_prefix(records, opening_plies, path)
        pair_bins[pair_score] += 1
        pair_points = 0.0

        for tags, movetext in records:
            score = _candidate_score(tags, path, 1)
            pair_points += score
            if tags["White"] == "Candidate":
                candidate_white_points += score
                candidate_white_games += 1
            else:
                candidate_black_points += score
                candidate_black_games += 1

            if score == 1.0:
                candidate_wins += 1
            elif score == 0.5:
                candidate_draws += 1
            else:
                candidate_losses += 1

            ply_count = tags.get("PlyCount")
            try:
                ply_counts.append(int(ply_count) if ply_count is not None else _count_movetext_plies(movetext))
            except ValueError:
                pass
            duration = tags.get("GameDuration")
            if duration:
                fields = duration.split(":")
                try:
                    seconds = sum(float(field) * (60 ** position) for position, field in enumerate(reversed(fields)))
                    duration_seconds.append(seconds)
                except ValueError:
                    pass
            category = _termination_category(tags.get("Termination"))
            termination_counts[category] = termination_counts.get(category, 0) + 1

        pair_scores.append(pair_points / 2.0)

    completed_games = completed_pairs * 2
    score = (candidate_wins + 0.5 * candidate_draws) / completed_games
    if 0.0 < score < 1.0:
        elo_difference = 400.0 * math.log10(score / (1.0 - score))
    else:
        elo_difference = None

    pair_mean = sum(pair_scores) / len(pair_scores)
    pair_variance = sum((value - pair_mean) ** 2 for value in pair_scores) / max(1, len(pair_scores) - 1)
    score_standard_error = math.sqrt(pair_variance / len(pair_scores))
    score_lower = max(0.0, score - 1.96 * score_standard_error)
    score_upper = min(1.0, score + 1.96 * score_standard_error)

    def to_elo(value):
        if value <= 0.0 or value >= 1.0:
            return None
        return 400.0 * math.log10(value / (1.0 - value))

    elo_lower = to_elo(score_lower)
    elo_upper = to_elo(score_upper)
    elo_uncertainty = (
        max(elo_difference - elo_lower, elo_upper - elo_difference)
        if elo_difference is not None and elo_lower is not None and elo_upper is not None
        else None
    )
    statistically_convincing = bool(elo_lower is not None and elo_lower > 0.0)
    llr_zero_vs_five = calculate_llr(pair_bins, elo0=0.0, elo1=5.0)

    report = {
        "completed_games": completed_games,
        "completed_pairs": completed_pairs,
        "candidate": "Bermuda MLP + GNN (epoch 7, \u03bb=1.0)",
        "baseline": "Bermuda MLP",
        "candidate_wins": candidate_wins,
        "candidate_draws": candidate_draws,
        "candidate_losses": candidate_losses,
        "score": score,
        "score_percent": 100.0 * score,
        "elo_difference": elo_difference,
        "elo_uncertainty_95": elo_uncertainty,
        "score_standard_error_paired": score_standard_error,
        "score_ci95_paired": [score_lower, score_upper],
        "elo_ci95_paired": [elo_lower, elo_upper],
        "statistically_convincing_improvement": statistically_convincing,
        "statistical_method": "fixed-sample paired-pair-score standard error, normal 95% interval, Elo logit transform",
        "sprt_llr_h0_0_vs_h1_5_at_fixed_endpoint": llr_zero_vs_five,
        "sprt_stopping_applied": False,
        "white_score_percent_candidate": (
            100.0 * candidate_white_points / candidate_white_games
            if candidate_white_games else None
        ),
        "black_score_percent_candidate": (
            100.0 * candidate_black_points / candidate_black_games
            if candidate_black_games else None
        ),
        "average_game_length_plies": sum(ply_counts) / len(ply_counts) if ply_counts else None,
        "average_game_duration_seconds": (
            sum(duration_seconds) / len(duration_seconds) if duration_seconds else None
        ),
        "termination_counts": termination_counts,
        "abnormal_terminations": (
            None if termination_counts.get("unavailable", 0) > 0 else sum(
                count for category, count in termination_counts.items()
                if category != "normal_or_adjudicated"
            )
        ),
        "average_nodes_per_move": None,
        "average_nodes_per_game": None,
        "average_evaluation_time": None,
        "average_nodes_per_second": None,
        "unavailable_metrics": [
            "nodes per move/game, evaluation time, and nodes per second are not present in the retained PGNs"
        ],
        "pentanomial_counts": pair_bins,
    }
    if metadata:
        report["experiment"] = metadata
        if metadata.get("performanceBenchmark"):
            report["performance_benchmark"] = metadata["performanceBenchmark"]
    return report


def write_fixed_report(report, json_path, markdown_path):
    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    elo_text = "not estimable at 0% or 100% score"
    if report["elo_difference"] is not None and report["elo_uncertainty_95"] is not None:
        elo_text = f"{report['elo_difference']:.2f} +/- {report['elo_uncertainty_95']:.2f} Elo (approx. 95% paired CI)"
    elif report["elo_difference"] is not None:
        elo_text = (
            f"{report['elo_difference']:.2f} Elo; paired 95% interval reaches a score boundary, "
            "so finite Elo uncertainty is unavailable"
        )
    rows = [
        "# Bermuda MLP + GNN Epoch 7 Match Report",
        "",
        f"- Candidate: {report['candidate']}",
        f"- Baseline: {report['baseline']}",
        f"- Games/pairs: {report['completed_games']}/{report['completed_pairs']}",
        f"- Candidate W-D-L: {report['candidate_wins']}-{report['candidate_draws']}-{report['candidate_losses']}",
        f"- Score: {report['score_percent']:.2f}%",
        f"- Elo estimate: {elo_text}",
        f"- Statistically convincing improvement (95% paired CI above 0): {report['statistically_convincing_improvement']}",
        f"- Candidate score as White: {report['white_score_percent_candidate']}",
        f"- Candidate score as Black: {report['black_score_percent_candidate']}",
        f"- Average game length (plies): {report['average_game_length_plies']}",
        f"- Average game duration (seconds): {report['average_game_duration_seconds']}",
        f"- Terminations: {json.dumps(report['termination_counts'], sort_keys=True)}",
        "- Nodes/evaluation-time/NPS metrics: unavailable from existing PGN outputs",
        "- SPRT stopping: not applied; all requested pairs are analyzed",
        "",
    ]
    experiment = report.get("experiment") or {}
    configuration = experiment.get("experiment", experiment)
    if configuration.get("experimentName"):
        rows.extend([
            "## Reproducibility",
            "",
            f"- Experiment: {configuration.get('experimentName')}",
            f"- Run ID: {configuration.get('runId')}",
            f"- Recorded: {experiment.get('recordedAt')}",
            f"- Time control / games / seed: {configuration.get('timeControl')} / "
            f"{configuration.get('games')} / {configuration.get('seed')}",
            f"- Hash / concurrency / threads: {configuration.get('hashMb')} MB / "
            f"{configuration.get('concurrency')} / {configuration.get('engineThreads')}",
            f"- Openings SHA-256: {configuration.get('openingsSha256')}",
            f"- Baseline NNUE SHA-256: {configuration.get('nnueSha256')}",
            f"- Graph checkpoint: {configuration.get('graphCheckpointPath')} "
            f"(epoch {configuration.get('graphCheckpointEpoch')}, "
            f"SHA-256 {configuration.get('graphCheckpointSha256')})",
            f"- Graph export SHA-256: {configuration.get('graphExportSha256')}",
            f"- Residual multiplier / runtime scale: {configuration.get('graphResidualMultiplier')} / "
            f"{configuration.get('graphOptionScale')}",
            "",
        ])
    benchmark = report.get("performance_benchmark")
    if benchmark:
        rows.extend([
            "## Engine Microbenchmark",
            "",
            f"- Graph encoding: {benchmark.get('graphEncodingUsPerOp')} us/op",
            f"- Graph inference: {benchmark.get('graphInferenceUsPerOp')} us/op",
            f"- Combined static evaluation: {benchmark.get('combinedEvaluationUsPerOp')} us/op",
            f"- Full-search benchmark: depth {benchmark.get('searchDepth')}, "
            f"{benchmark.get('searchNodesPerSecond')} nodes/s",
            "",
        ])
    markdown_path.write_text("\n".join(rows), encoding="utf-8")


def split_pgn(path, output_dir, first_pair, expected_pairs, opening_plies=None):
    if first_pair < 1 or expected_pairs < 1:
        raise ValueError("first-pair and expected-pairs must be positive")
    games = _read_pgn_records(path)
    if len(games) != expected_pairs * 2:
        raise ValueError(
            f"{path}: expected {expected_pairs * 2} games, found {len(games)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    try:
        for offset in range(expected_pairs):
            records = games[offset * 2 : offset * 2 + 2]
            _validate_pair([tags for tags, _ in records], path)
            _validate_opening_prefix(records, opening_plies, path)
            pair_number = first_pair + offset
            target = output_dir / f"pair-{pair_number:05d}.pgn"
            temporary = target.with_suffix(".pgn.tmp")
            text = "\n".join(record.rstrip() for _, record in records) + "\n"
            temporary.write_text(text, encoding="utf-8")
            pending.append((temporary, target))
        for temporary, target in pending:
            temporary.replace(target)
    except Exception:
        for temporary, _ in pending:
            temporary.unlink(missing_ok=True)
        raise


def sprt_boundaries(alpha=0.05, beta=0.05):
    if not math.isfinite(alpha) or not math.isfinite(beta):
        raise ValueError("alpha and beta must be finite")
    if not 0 < alpha < 1 or not 0 < beta < 1:
        raise ValueError("alpha and beta must be strictly between 0 and 1")
    lower = math.log(beta / (1 - alpha))
    upper = math.log((1 - beta) / alpha)
    return lower, upper


def _validate_sprt_parameters(counts, alpha, beta, elo0, elo1):
    if len(counts) != 5 or any(not isinstance(value, int) or value < 0 for value in counts):
        raise ValueError("pentanomial counts must be five non-negative integers")
    if not math.isfinite(elo0) or not math.isfinite(elo1) or elo0 >= elo1:
        raise ValueError("elo0 and elo1 must be finite, with elo0 less than elo1")
    sprt_boundaries(alpha, beta)


def _regularized_pdf(counts):
    regularized = [value if value > 0 else EPSILON for value in counts]
    total = sum(regularized)
    scores = [index / 4 for index in range(5)]
    return total, [(score, count / total) for score, count in zip(scores, regularized)]


def _mle_expected(pdf, target):
    shifted = [(score - target, probability) for score, probability in pdf]
    minimum = min(value for value, _ in shifted)
    maximum = max(value for value, _ in shifted)
    if minimum >= 0 or maximum <= 0:
        raise ValueError("SPRT target score is outside the pentanomial support")

    lower = -1 / maximum + 1e-9
    upper = -1 / minimum - 1e-9

    def secular(value):
        return sum(
            probability * score / (1 + value * score)
            for score, probability in shifted
        )

    for _ in range(100):
        midpoint = (lower + upper) / 2
        if midpoint == lower or midpoint == upper:
            break
        if secular(midpoint) > 0:
            lower = midpoint
        else:
            upper = midpoint
    root = (lower + upper) / 2
    return [
        (score, probability / (1 + root * (score - target)))
        for score, probability in pdf
    ]


def calculate_llr(counts, elo0=0.0, elo1=5.0):
    _validate_sprt_parameters(counts, 0.05, 0.05, elo0, elo1)
    if sum(counts) == 0:
        return None

    total, pdf = _regularized_pdf(counts)
    score0 = 1 / (1 + 10 ** (-elo0 / 400))
    score1 = 1 / (1 + 10 ** (-elo1 / 400))
    pdf0 = _mle_expected(pdf, score0)
    pdf1 = _mle_expected(pdf, score1)
    return total * sum(
        probability * (math.log(estimate1) - math.log(estimate0))
        for (_, probability), (_, estimate0), (_, estimate1) in zip(pdf, pdf0, pdf1)
    )


def update_sprt(counts, overshoot, alpha=0.05, beta=0.05, elo0=0.0, elo1=5.0):
    _validate_sprt_parameters(counts, alpha, beta, elo0, elo1)
    lower, upper = sprt_boundaries(alpha, beta)
    llr = calculate_llr(counts, elo0=elo0, elo1=elo1)
    if llr is None:
        return {
            "llr": None,
            "lower_boundary": lower,
            "upper_boundary": upper,
            "adjusted_lower_boundary": lower,
            "adjusted_upper_boundary": upper,
            "status": "CONTINUE",
        }

    sample_count = sum(counts)
    if sample_count == overshoot["last_update"]:
        pass
    elif sample_count == overshoot["last_update"] + 1:
        if llr < overshoot["ref0"]:
            delta = llr - overshoot["ref0"]
            overshoot["m0"] += delta
            overshoot["sq0"] += delta**2
            overshoot["ref0"] = llr
        if llr > overshoot["ref1"]:
            delta = llr - overshoot["ref1"]
            overshoot["m1"] += delta
            overshoot["sq1"] += delta**2
            overshoot["ref1"] = llr
    else:
        overshoot["ref0"] = llr
        overshoot["ref1"] = llr
    overshoot["last_update"] = sample_count

    lower_correction = (
        -overshoot["sq0"] / overshoot["m0"] / 2
        if overshoot["m0"] != 0
        else 0
    )
    upper_correction = (
        overshoot["sq1"] / overshoot["m1"] / 2
        if overshoot["m1"] != 0
        else 0
    )
    adjusted_lower = lower + lower_correction
    adjusted_upper = upper - upper_correction
    status = "CONTINUE"
    if llr < adjusted_lower:
        status = "FAIL"
    elif llr > adjusted_upper:
        status = "PASS"
    return {
        "llr": llr,
        "lower_boundary": lower,
        "upper_boundary": upper,
        "adjusted_lower_boundary": adjusted_lower,
        "adjusted_upper_boundary": adjusted_upper,
        "status": status,
    }


def _new_overshoot_state():
    return {
        "last_update": 0,
        "ref0": 0.0,
        "m0": 0.0,
        "sq0": 0.0,
        "ref1": 0.0,
        "m1": 0.0,
        "sq1": 0.0,
    }


def analyze_matches(
    matches_dir,
    completed_pairs,
    max_pairs,
    alpha=0.05,
    beta=0.05,
    elo0=0.0,
    elo1=5.0,
    previous_result=None,
):
    if completed_pairs < 0 or max_pairs < 1 or completed_pairs > max_pairs:
        raise ValueError("completed-pairs must be between zero and max-pairs")

    if previous_result is None:
        counts = [0, 0, 0, 0, 0]
        overshoot = _new_overshoot_state()
        wins = draws = losses = 0
        first_pair = 1
        result = update_sprt(
            counts, overshoot, alpha=alpha, beta=beta, elo0=elo0, elo1=elo1
        )
    else:
        expected_previous_pairs = completed_pairs - 1
        if (
            previous_result.get("completed_pairs") != expected_previous_pairs
            or previous_result.get("max_pairs") != max_pairs
            or previous_result.get("status") != "CONTINUE"
            or previous_result.get("alpha") != alpha
            or previous_result.get("beta") != beta
            or previous_result.get("elo0") != elo0
            or previous_result.get("elo1") != elo1
        ):
            raise ValueError("previous SPRT state does not match this incremental update")
        counts = list(previous_result["pentanomial_counts"])
        wins = previous_result["candidate_wins"]
        draws = previous_result["candidate_draws"]
        losses = previous_result["candidate_losses"]
        overshoot = dict(previous_result["overshoot_state"])
        first_pair = completed_pairs
        result = {
            "llr": previous_result["llr"],
            "lower_boundary": previous_result["lower_boundary"],
            "upper_boundary": previous_result["upper_boundary"],
            "adjusted_lower_boundary": previous_result[
                "adjusted_lower_boundary"
            ],
            "adjusted_upper_boundary": previous_result[
                "adjusted_upper_boundary"
            ],
            "status": previous_result["status"],
        }

    for pair_number in range(first_pair, completed_pairs + 1):
        path = matches_dir / f"pair-{pair_number:05d}.pgn"
        if not path.is_file():
            raise ValueError(f"completed pair file is missing: {path}")
        pair_score = read_pair(path)
        counts[pair_score] += 1
        result = update_sprt(
            counts, overshoot, alpha=alpha, beta=beta, elo0=elo0, elo1=elo1
        )
        pair_games = read_games(path)
        for game_number, game in enumerate(pair_games, start=1):
            score = _candidate_score(game, path, game_number)
            if score == 1:
                wins += 1
            elif score == 0.5:
                draws += 1
            else:
                losses += 1
        if result["status"] != "CONTINUE" and pair_number < completed_pairs:
            raise ValueError(
                f"completed pairs continue after the SPRT stopped at pair {pair_number}"
            )

    if completed_pairs == max_pairs and result["status"] == "CONTINUE":
        result["status"] = "INCONCLUSIVE"
    result.update(
        {
            "completed_pairs": completed_pairs,
            "completed_games": completed_pairs * 2,
            "max_pairs": max_pairs,
            "max_games": max_pairs * 2,
            "pentanomial_counts": counts,
            "candidate_wins": wins,
            "candidate_draws": draws,
            "candidate_losses": losses,
            "candidate_score": wins + draws / 2,
            "alpha": alpha,
            "beta": beta,
            "elo0": elo0,
            "elo1": elo1,
            "elo_model": "logistic",
            "overshoot_state": overshoot,
        }
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches-dir", type=Path)
    parser.add_argument("--completed-pairs", type=int)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--elo0", type=float, default=0.0)
    parser.add_argument("--elo1", type=float, default=5.0)
    parser.add_argument("--previous-result-base64")
    parser.add_argument("--split-pgn", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--first-pair", type=int)
    parser.add_argument("--expected-pairs", type=int)
    parser.add_argument("--opening-plies", type=int)
    parser.add_argument("--fixed-report", action="store_true")
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--report-markdown", type=Path)
    parser.add_argument("--metadata-json", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.split_pgn:
            if not args.output_dir or args.first_pair is None or args.expected_pairs is None:
                raise ValueError(
                    "--split-pgn requires --output-dir, --first-pair, and --expected-pairs"
                )
            split_pgn(
                args.split_pgn,
                args.output_dir,
                args.first_pair,
                args.expected_pairs,
                opening_plies=args.opening_plies,
            )
            return 0
        if args.fixed_report:
            if args.matches_dir is None or args.completed_pairs is None:
                raise ValueError("fixed report requires --matches-dir and --completed-pairs")
            if args.report_json is None or args.report_markdown is None:
                raise ValueError("fixed report requires --report-json and --report-markdown")
            metadata = None
            if args.metadata_json is not None:
                metadata = json.loads(args.metadata_json.read_text(encoding="utf-8-sig"))
            report = analyze_fixed_matches(args.matches_dir, args.completed_pairs, metadata)
            write_fixed_report(report, args.report_json, args.report_markdown)
            print(json.dumps(report, sort_keys=True))
            return 0
        if (
            args.matches_dir is None
            or args.completed_pairs is None
            or args.max_pairs is None
        ):
            raise ValueError(
                "SPRT analysis requires --matches-dir, --completed-pairs, and --max-pairs"
            )
        result = analyze_matches(
            args.matches_dir,
            args.completed_pairs,
            args.max_pairs,
            alpha=args.alpha,
            beta=args.beta,
            elo0=args.elo0,
            elo1=args.elo1,
            previous_result=(
                json.loads(
                    base64.b64decode(
                        args.previous_result_base64, validate=True
                    ).decode("utf-8")
                )
                if args.previous_result_base64
                else None
            ),
        )
    except (OSError, ValueError, binascii.Error) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
