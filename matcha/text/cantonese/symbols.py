punctuation = ["!", "?", "…", ",", ".", "'", "-"]
pu_symbols = punctuation + ["SP", "UNK"]
pad = "_"
ONSETS = "b d g gw z p t k kw c m n ng f h s l w j"
NUCLEUSES = "aa a i yu u oe e eo o m n ng"
CODAS = "p t k m n ng i u"

symbols = list(set(ONSETS.split() + NUCLEUSES.split() + CODAS.split()))
# sort symbols to ensure consistent order
symbols.sort()
symbols = [pad] + pu_symbols + symbols

if __name__ == "__main__":
    print(len(symbols))
