"""
Defines the letter counts which make up the consonant and vowel decks for the 10
letters rounds in a game of Countdown. There are bonus letter sets which can be
added to each deck to provide more varied games and increase the scope of
possible words that can appear.

There is also a variable for determining the frequency of each of the three
selection types:

3-vowels, 6-consonants (28% of rounds)
4-vowels, 5-consonants (59% of rounds)
5-vowels, 4-consonants (13% of rounds)
"""

VOWEL_COUNTS = {
    "A": 15,
    "E": 21,
    "I": 13,
    "O": 13,
    "U": 7,
}

CONSONANT_COUNTS = {
    "B": 2,
    "C": 3,
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

BONUS_VOWEL_SETS = [
    {
        "I": 1,
        "O": 1,
        "U": 1,
    },
    {
        "A": 1,
        "E": 1,
        "O": 1,
    },
    {
        "I": 1,
        "U": 2,
    },
    {
        "A": 2,
        "E": 1,
    },
]

BONUS_CONSONANT_SETS = [
    {
        "H": 1,
        "S": 1,
        "T": 1,
        "V": 1,
    },
    {
        "R": 1,
        "N": 1,
        "V": 1,
        "W": 1,
    },
    {
        "C": 2,
        "K": 1,
        "L": 1,
    },
    {
        "F": 2,
        "W": 1,
        "Y": 1,
    },
    {
        "B": 2,
        "D": 1,
        "M": 1,
    },
]

VOWEL_FREQUENCIES = {
    "3-vowels": 0.28,
    "4-vowels": 0.59,
    "5-vowels": 0.13,
}
