punctuation = ["!", "?", "…", ",", ".", "'", "-"]
pu_symbols = punctuation + ["SP", "UNK"]
pad = "_"

# ARPAbet phonemes (lowercased as in the reviewed code)
phonemes = [
    "aa",
    "ae",
    "ah",
    "ao",
    "aw",
    "ay",
    "b",
    "ch",
    "d",
    "dh",
    "eh",
    "er",
    "ey",
    "f",
    "g",
    "hh",
    "ih",
    "iy",
    "jh",
    "k",
    "l",
    "m",
    "n",
    "ng",
    "ow",
    "oy",
    "p",
    "r",
    "s",
    "sh",
    "t",
    "th",
    "uh",
    "uw",
    "v",
    "w",
    "y",
    "z",
    "zh",
]

symbols = [pad] + pu_symbols + phonemes

if __name__ == "__main__":
    print(f"Total symbols: {len(symbols)}")
    print("Symbols:", symbols)
