punctuation = [
    "!",
    "?",
    "…",
    ",",
    ".",
    "'",
    "-",
    "！",
    "？",
    "，",
    "。",
    "…",
    "；",
    "：",
    "「",
    "」",
    "『",
    "』",
    "（",
    "）",
    "【",
    "】",
]
pu_symbols = punctuation + ["SP", "UNK"]
pad = "_"

# Mandarin phonemes based on pinyin
initials = ["b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s"]
finals = [
    "a",
    "o",
    "e",
    "i",
    "u",
    "v",
    "ai",
    "ei",
    "ui",
    "ao",
    "ou",
    "iu",
    "ie",
    "ve",
    "er",
    "an",
    "en",
    "in",
    "un",
    "vn",
    "ang",
    "eng",
    "ing",
    "ong",
]

phonemes = initials + finals

symbols = [pad] + pu_symbols + phonemes

if __name__ == "__main__":
    print(f"Total symbols: {len(symbols)}")
    print("Symbols:", symbols)
