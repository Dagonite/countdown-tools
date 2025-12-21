"""Defines the letter counts which make up the consonant and vowel decks used on
the game show Countdown. Because of certain letter counts and the limited
vowel/consonant distributions allowed, there's roughly 3,300 words which have a
0% chance of ever showing up in letters rounds. Some of the 9-letter words can
still show up as conundrums. All of these words can be found in
`impossible.csv`.

The rough popularity of the three vowel/consonant distributions used on the
show are:

3-vowels, 6-consonants (picked in 28% of rounds)
4-vowels, 5-consonants (picked in 60% of rounds)
5-vowels, 4-consonants (picked in 12% of rounds)
"""

VOWEL_COUNTS = {
    "A": 15,
    "E": 21,
    "I": 13,
    "O": 13,
    "U": 7,
}

BONUS_VOWEL_SETS = [{}]

CONSONANT_COUNTS = {
    "B": 2,
    "C": 2,
    "D": 6,
    "F": 2,
    "G": 4,
    "H": 2,
    "J": 1,
    "K": 1,
    "L": 5,
    "M": 4,
    "N": 7,
    "P": 4,
    "Q": 1,
    "R": 9,
    "S": 9,
    "T": 9,
    "V": 1,
    "W": 1,
    "X": 1,
    "Y": 1,
    "Z": 1,
}

BONUS_CONSONANT_SETS = [
    {
        "B": 1,
        "L": 1,
        "V": 1,
        "W": 1,
    },
    {
        "C": 1,
        "G": 1,
        "V": 1,
        "W": 1,
    },
    {
        "C": 2,
        "F": 1,
        "L": 1,
    },
]

VOWEL_FREQUENCIES = {
    "3-vowels": 0.28,
    "4-vowels": 0.60,
    "5-vowels": 0.12,
}
