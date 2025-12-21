"""Simulate Countdown letters rounds using shuffled vowel and consonant decks.

The script builds consonant and vowel decks from weighted letter counts and
reshuffles any deck whose clumpiness exceeds the configured threshold. If no
deck clears the threshold within the allotted attempts, the least-clumpy deck
seen so far is chosen instead.

Clumpiness captures how tightly duplicates and nasty letters bunch together; a
clumpy deck limits letter synergy and makes for dull rounds. The TV show spaces
its decks manually before filming, and this script approximates that behaviour
without over-correcting.

Once decks are locked in, each game simulates 10 letters rounds by drawing from
the decks according to the vowel/consonant frequency mix, finding the best
constructible words for every draw, and repeating for the requested `--games`
count.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from distributions import BONUS_CONSONANT_SETS, BONUS_VOWEL_SETS, CONSONANT_COUNTS, VOWEL_COUNTS, VOWEL_FREQUENCIES


LetterDeck = List[str]
AnagramIndex = dict[int, dict[str, List[str]]]
WordOrder = dict[str, int]

NASTY_LETTERS = {"J", "K", "Q", "V", "W", "X", "Y", "Z"}


@dataclass(frozen=True)
class SimulationConfig:
    max_clumpiness: int
    shuffle_attempts: int
    duplicate_safe_gap: int
    nasty_safe_gap: int
    debug_log: Path | None


@dataclass(frozen=True)
class ClumpinessBreakdown:
    total: int
    duplicate_penalty: int
    nasty_penalty: int
    duplicate_details: List[str]
    nasty_details: List[str]


def main() -> None:
    args = parse_args()  # Fetch args
    rng = random.Random(args.seed)  # Determine RNG (if no seed is provided, system time is used)

    # Bundle simulation arguments
    config = SimulationConfig(
        max_clumpiness=args.max_clumpiness,
        shuffle_attempts=max(2, args.shuffle_attempts),
        duplicate_safe_gap=args.duplicate_safe_gap,
        nasty_safe_gap=args.nasty_safe_gap,
        debug_log=args.debug_log,
    )

    anagram_index, word_order = load_dictionary(args.words)  # Load list of valid Countdown words

    # Simulate the requested number of games
    all_rows = []
    for game_idx in range(args.games):
        all_rows.extend(simulate_game(rng, anagram_index, word_order, config=config))
        if (game_idx + 1) % 10000 == 0:
            print(f"Simulated {(game_idx + 1) * 10} rounds")

    # Write results to CSV
    write_results(args.output, all_rows)
    print(f"Appended {len(all_rows)} rounds to {args.output}")


def parse_args() -> argparse.Namespace:
    """Resolve values for command-line args."""

    parser = argparse.ArgumentParser(description="Simulate Countdown letters rounds.")
    repo_root = Path(__file__).resolve().parent.parent
    dictionary_default = repo_root / "dictionary" / "words.txt"
    simulation_dir = repo_root / "simulations"
    legacy_default = repo_root / "words.txt"
    word_list_path = dictionary_default if dictionary_default.exists() else legacy_default

    parser.add_argument("--words", type=Path, default=(word_list_path), help="Path to the dictionary file (text file or a CSV with a 'Word' column)")
    parser.add_argument("--games", type=int, default=1, help="Number of games to simulate (10 rounds per game)")
    parser.add_argument("--output", type=Path, default=simulation_dir / "simulated_rounds.csv", help="What CSV to append the output to")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility")
    parser.add_argument("--max-clumpiness", type=int, default=26, help="Maximum clumpiness score for a deck to be accepted")
    parser.add_argument("--duplicate-safe-gap", type=int, default=5, help="Minimum gap between duplicate letters to avoid clumpiness penalties")
    parser.add_argument("--nasty-safe-gap", type=int, default=6, help="Minimum gap between nasty letters to avoid clumpiness penalties")
    parser.add_argument("--shuffle-attempts", type=int, default=25, help="Maximum number of random shuffles to attempt when generating a deck")
    parser.add_argument("--debug-log", type=Path, nargs="?", const=simulation_dir / "deck_debug.log", default=None, help="Enable debug logging for decks)")

    return parser.parse_args()


def load_dictionary(path: Path) -> Tuple[AnagramIndex, WordOrder]:
    """Load words into an anagram index plus a stable word ordering.

    Supports a plain newline-delimited word list or a CSV containing a `Word`
    column; the anagram index is keyed by sorted letters for fast lookups.
    """

    def store_word(raw: str) -> None:
        raw = raw.strip()
        if not raw:
            return

        for variant in raw.split("/"):
            word = variant.strip().upper()
            if not word or word in order:
                continue

            order[word] = len(order)
            key = "".join(sorted(word))
            index[len(word)][key].append(word)

    index: AnagramIndex = defaultdict(lambda: defaultdict(list))
    order: WordOrder = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames and "Word" in reader.fieldnames:
            for row in reader:
                store_word(row.get("Word") or "")
        else:
            handle.seek(0)
            for raw in handle:
                store_word(raw)

    return {length: dict(keys) for length, keys in index.items()}, order


def weighted_order(counts: dict[str, int], rng: random.Random) -> LetterDeck:
    """Generate a deck where each draw is weighted by the remaining letter
    counts."""

    remaining = dict(counts)
    deck: LetterDeck = []
    while remaining:
        letters = list(remaining)
        weights = [remaining[ch] for ch in letters]
        choice = rng.choices(letters, weights=weights, k=1)[0]
        deck.append(choice)
        remaining[choice] -= 1
        if remaining[choice] == 0:
            del remaining[choice]

    return deck


def clumpiness_score(deck: Sequence[str], duplicate_safe_gap: int, nasty_safe_gap: int) -> int:
    """Score how tightly duplicates and nasty letters are bunched together in
    the supplied deck."""

    penalty: int = 0
    last_seen: dict[str, int] = {}
    last_nasty: int | None = None

    for idx, letter in enumerate(deck):
        if letter in last_seen:
            gap = idx - last_seen[letter]
            if gap < duplicate_safe_gap:
                penalty += duplicate_safe_gap - gap
        last_seen[letter] = idx

        if letter in NASTY_LETTERS:
            if last_nasty is not None:
                gap = idx - last_nasty
                if gap < nasty_safe_gap:
                    penalty += 2 * (nasty_safe_gap - gap)
            last_nasty = idx

    return penalty


def clumpiness_breakdown(deck: Sequence[str], duplicate_safe_gap: int, nasty_safe_gap: int) -> ClumpinessBreakdown:
    """Return a detailed clumpiness breakdown for a deck.

    Details the individual occurrences of duplicate and nasty letter clumping.
    Each penalty records the indices, gap, and points added.
    """

    duplicate_penalty: int = 0
    nasty_penalty: int = 0
    duplicate_details: List[str] = []
    nasty_details: List[str] = []

    last_seen: dict[str, int] = {}
    last_nasty_idx: int | None = None
    last_nasty_letter: str | None = None

    for idx, letter in enumerate(deck):
        prev_idx = last_seen.get(letter)
        if prev_idx is not None:
            gap = idx - prev_idx
            if gap < duplicate_safe_gap:
                penalty = duplicate_safe_gap - gap
                duplicate_penalty += penalty
                duplicate_details.append(f"{letter}@{prev_idx}->{idx} gap {gap} (+{penalty})")

        last_seen[letter] = idx

        if letter in NASTY_LETTERS:
            if last_nasty_idx is not None:
                gap = idx - last_nasty_idx
                if gap < nasty_safe_gap:
                    penalty = 2 * (nasty_safe_gap - gap)
                    nasty_penalty += penalty
                    previous_letter = last_nasty_letter or deck[last_nasty_idx]
                    nasty_details.append(f"{previous_letter}@{last_nasty_idx}->{letter}@{idx} gap {gap} (+{penalty})")

            last_nasty_idx = idx
            last_nasty_letter = letter

    return ClumpinessBreakdown(
        total=duplicate_penalty + nasty_penalty,
        duplicate_penalty=duplicate_penalty,
        nasty_penalty=nasty_penalty,
        duplicate_details=duplicate_details,
        nasty_details=nasty_details,
    )


def scaled_clumpiness_target(counts: dict[str, int], base_target: int) -> int:
    """Scale a base clumpiness threshold according to deck diversity.

    The scaling factor grows as the deck size rises and the number of unique
    letters shrinks, relaxing the threshold for low-diversity decks (e.g.
    vowels) and tightening it slightly for higher-diversity decks. Negative base
    targets still disable clumpiness checks.
    """
    if base_target < 0:
        return base_target

    total_letters = sum(counts.values())
    unique_letters = max(1, len(counts))
    scale = math.sqrt(total_letters / unique_letters)
    return math.ceil(base_target * scale)


def smoothed_order(counts: dict[str, int], rng: random.Random, config: SimulationConfig, max_clumpiness: int, return_meta: bool = False) -> LetterDeck:
    """Shuffle until a deck meets the clumpiness target or attempts are
    exhausted.

    The provided `max_clumpiness` is expected to be scaled for this deck;
    callers can pass a negative value to disable the threshold check entirely.
    When `return_meta` is True, include the score and attempt count for logging.
    """
    target = max_clumpiness if max_clumpiness >= 0 else None

    best_deck: LetterDeck | None = None
    best_score: int | None = None

    for attempt in range(1, config.shuffle_attempts + 1):
        deck = weighted_order(counts, rng)
        score = clumpiness_score(deck, duplicate_safe_gap=config.duplicate_safe_gap, nasty_safe_gap=config.nasty_safe_gap)
        if best_score is None or score < best_score:
            best_score = score
            best_deck = deck
        if target is not None and score <= target:
            return (deck, score, attempt) if return_meta else deck

    assert best_deck is not None
    assert best_score is not None

    return (best_deck, best_score, config.shuffle_attempts) if return_meta else best_deck


def append_debug_log(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line.rstrip("\n") + "\n")


def deck_debug_lines(deck_label: str, deck: LetterDeck, target: int, score: int, attempts: int, config: SimulationConfig) -> List[str]:
    """Format a deck's clumpiness summary and breakdown for logging."""

    breakdown = clumpiness_breakdown(deck, config.duplicate_safe_gap, config.nasty_safe_gap)
    if breakdown.total != score:
        summary = f"{deck_label} target {target} -> score {score} (recalculated {breakdown.total}) in {attempts} attempt(s)"
    else:
        summary = f"{deck_label} target {target} -> score {score} in {attempts} attempt(s)"

    deck_line = f"{deck_label} deck ({len(deck)}): {''.join(deck)}"
    duplicate_lines: List[str] = []
    if breakdown.duplicate_details:
        duplicate_lines.append("  Duplicate penalties:")
        duplicate_lines.extend(f"    - {detail}" for detail in breakdown.duplicate_details)
    else:
        duplicate_lines.append(f"  Duplicate penalties: none (gaps >= {config.duplicate_safe_gap})")

    nasty_lines: List[str] = []
    if breakdown.nasty_details:
        nasty_lines.append("  Nasty penalties:")
        nasty_lines.extend(f"    - {detail}" for detail in breakdown.nasty_details)
    else:
        nasty_lines.append(f"  Nasty penalties: none (gaps >= {config.nasty_safe_gap})")

    summary_line = f"{summary} (duplicates {breakdown.duplicate_penalty}, nasty {breakdown.nasty_penalty})"

    return [summary_line, deck_line, *duplicate_lines, *nasty_lines]


def choose_mix(rng: random.Random) -> Tuple[int, int]:
    """Sample a vowel/consonant mix according to the configured frequency
    weights."""

    mixes: list[Tuple[int, int]] = []
    weights: list[float] = []
    for label, weight in VOWEL_FREQUENCIES.items():
        try:
            vowels = int(label.split("-")[0])
        except (ValueError, IndexError):
            raise ValueError(f"Invalid vowel frequency label: {label!r}")

        consonants = 9 - vowels
        mixes.append((vowels, consonants))
        weights.append(weight)

    return rng.choices(mixes, weights=weights, k=1)[0]


def find_best_words(selection: Sequence[str], anagram_index: AnagramIndex, word_order: WordOrder) -> Tuple[List[str], int]:
    """Return the longest constructible word(s) and their length for the
    selection."""

    if not anagram_index:
        return [], 0

    letters = sorted(selection)
    max_length = min(len(letters), max(anagram_index))

    for target_length in range(max_length, 0, -1):
        buckets = anagram_index.get(target_length)
        if not buckets:
            continue

        seen_keys: set[str] = set()
        matches: List[str] = []

        for combo in combinations(range(len(letters)), target_length):
            key = "".join(letters[i] for i in combo)

            # Skip duplicate keys produced by repeated letters to avoid redundant lookups
            if key in seen_keys:
                continue

            seen_keys.add(key)
            words = buckets.get(key)

            if words:
                matches.extend(words)

        if matches:
            matches.sort(key=word_order.get)
            return matches, target_length

    return [], 0


def simulate_game(rng: random.Random, anagram_index: AnagramIndex, word_order: WordOrder, config: SimulationConfig) -> List[dict[str, str | int]]:
    """Simulate the standard 10 letters rounds of a Countdown game and return
    the round results."""

    # Apply one random bonus set to each deck for this game
    consonant_bonus = rng.choice(BONUS_CONSONANT_SETS)
    consonant_counts = dict(CONSONANT_COUNTS)
    for letter, extra in consonant_bonus.items():
        consonant_counts[letter] = consonant_counts.get(letter, 0) + extra

    vowel_bonus = rng.choice(BONUS_VOWEL_SETS)
    vowel_counts = dict(VOWEL_COUNTS)
    for letter, extra in vowel_bonus.items():
        vowel_counts[letter] = vowel_counts.get(letter, 0) + extra

    vowel_target = scaled_clumpiness_target(vowel_counts, config.max_clumpiness)
    consonant_target = scaled_clumpiness_target(consonant_counts, config.max_clumpiness)

    # Call smoothed_order with return_meta flag if debug flag is on
    if config.debug_log:
        vowel_deck, vowel_score, vowel_attempts = smoothed_order(vowel_counts, rng, config, vowel_target, return_meta=True)
        consonant_deck, consonant_score, consonant_attempts = smoothed_order(consonant_counts, rng, config, consonant_target, return_meta=True)

        # Add deck info to debug log
        lines: List[str] = []
        lines.extend(deck_debug_lines("Vowel", vowel_deck, vowel_target, vowel_score, vowel_attempts, config))
        lines.append("")
        lines.extend(deck_debug_lines("Consonant", consonant_deck, consonant_target, consonant_score, consonant_attempts, config))
        lines.append("")
        append_debug_log(config.debug_log, lines)
    else:
        vowel_deck = smoothed_order(vowel_counts, rng, config, max_clumpiness=vowel_target)
        consonant_deck = smoothed_order(consonant_counts, rng, config, max_clumpiness=consonant_target)

    vowel_idx = consonant_idx = 0
    rounds: List[dict[str, str | int]] = []

    # Simulate 10 rounds
    for _ in range(10):
        vowels_needed, consonants_needed = choose_mix(rng)
        if vowel_idx + vowels_needed > len(vowel_deck) or consonant_idx + consonants_needed > len(consonant_deck):
            raise RuntimeError("Not enough letters left in a deck for this round.")

        # Randomize the order letters are taken whilst keeping deck order within each type
        draw_pattern = ["V"] * vowels_needed + ["C"] * consonants_needed
        rng.shuffle(draw_pattern)

        draw: List[str] = []
        for slot in draw_pattern:
            if slot == "V":
                draw.append(vowel_deck[vowel_idx])
                vowel_idx += 1
            else:
                draw.append(consonant_deck[consonant_idx])
                consonant_idx += 1

        best_words, best_length = find_best_words(draw, anagram_index, word_order)
        rounds.append({"Selection": "".join(draw), "Max": "/".join(best_words), "Length": best_length})

    return rounds


def write_results(path: Path, rows: Iterable[dict[str, str | int]]) -> None:
    """Append rows of results to a CSV, writing the header only when the file is
    empty or new."""

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Selection", "Max", "Length"])
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
