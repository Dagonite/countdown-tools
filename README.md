# Countdown tools

Utilities for analysing Countdown words:

- `words.txt` - Base list of words of length 9 or less.

- `all_conundrums.txt` - List of every conundrum.

- `standard_conundrums.txt` - Conundrums I've deemed likely to appear in a regular Countdown episode.

- `tricky_conundrums.txt` - Conumdrums I've deemed unlikely to appear in a regular Countdown episode. This could be because they're too difficulty, hard to pronounce, or are inappropriate.

- `distributions.py` - The letter distributions and vowel/consonant frequencies in letters rounds.

- `simulate_rounds.py` - Script which generates a decks of consonants and vowels then simulates 10 letters rounds. Decks are de-clumped to avoid nasty clusters whilst keeping strong word coverage. Simulated games are appended to an CSV.

- `calculate_usefulness.py` - Script which analyses all of the simulated rounds to ascertain which words are the most useful to know. Words which have a 0% chance of showing up are excluded.

- `generate_stems.py` - Script which goes through words by usefulness and finds all of the words which can be stemmed by adding one letter. Only computed for words of length 6 through 9.

- `generate_unstemmable.py` - Script which goes through words by usefulness and finds all of the words which don't have a direct stem. Only computed for words of length 6 through 9.
