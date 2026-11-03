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


def split_pgn(path, output_dir, first_pair, expected_pairs):
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
            )
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
