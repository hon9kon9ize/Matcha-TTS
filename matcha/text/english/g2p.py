import pickle
import os
import re
from matcha.text.english.symbols import punctuation
from g2p_en import G2p
from transformers import DebertaV2Tokenizer
from pydips import BertModel

current_file_path = os.path.dirname(__file__)
CMU_DICT_PATH = os.path.join(current_file_path, "cmudict.rep")
CACHE_PATH = os.path.join(current_file_path, "cmudict_cache.pickle")
_g2p = G2p()
LOCAL_PATH = "./bert/deberta-v3-large"
tokenizer = DebertaV2Tokenizer.from_pretrained(LOCAL_PATH)
ws_model = BertModel()

arpa = {
    "AH0",
    "S",
    "AH1",
    "EY2",
    "AE2",
    "EH0",
    "OW2",
    "UH0",
    "NG",
    "B",
    "G",
    "AY0",
    "M",
    "AA0",
    "F",
    "AO0",
    "ER2",
    "UH1",
    "IY1",
    "AH2",
    "DH",
    "IY0",
    "EY1",
    "IH0",
    "K",
    "N",
    "W",
    "IY2",
    "T",
    "AA1",
    "ER1",
    "EH2",
    "OY0",
    "UH2",
    "UW1",
    "Z",
    "AW2",
    "AW1",
    "V",
    "UW2",
    "AA2",
    "ER",
    "AW0",
    "UW0",
    "R",
    "OW1",
    "EH1",
    "ZH",
    "AE0",
    "IH2",
    "IH",
    "Y",
    "JH",
    "P",
    "AY1",
    "EY0",
    "OY2",
    "TH",
    "HH",
    "D",
    "ER0",
    "CH",
    "AO1",
    "AE1",
    "AO2",
    "OY1",
    "AY2",
    "IH1",
    "OW0",
    "L",
    "SH",
}


def post_replace_ph(ph):
    from matcha.text.english.symbols import symbols

    rep_map = {
        "：": ",",
        "；": ",",
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "\n": ".",
        "·": ",",
        "、": ",",
        "…": "...",
        "···": "...",
        "・・・": "...",
        "v": "V",
    }
    if ph in rep_map.keys():
        ph = rep_map[ph]
    if ph in symbols:
        return ph
    if ph not in symbols:
        ph = "UNK"
    return ph


def read_dict():
    g2p_dict = {}
    start_line = 49
    with open(CMU_DICT_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= start_line:
                line = line.strip()
                word_split = line.split("  ")
                word = word_split[0]

                syllable_split = word_split[1].split(" - ")
                g2p_dict[word] = []
                for syllable in syllable_split:
                    phone_split = syllable.split(" ")
                    g2p_dict[word].append(phone_split)

            line_index = line_index + 1
            line = f.readline()

    return g2p_dict


def cache_dict(g2p_dict, file_path):
    with open(file_path, "wb") as pickle_file:
        pickle.dump(g2p_dict, pickle_file)


def get_dict():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as pickle_file:
            g2p_dict = pickle.load(pickle_file)
    else:
        g2p_dict = read_dict()
        cache_dict(g2p_dict, CACHE_PATH)

    return g2p_dict


eng_dict = get_dict()


def refine_ph(phn):
    tone = 0
    if re.search(r"\d$", phn):
        tone = int(phn[-1]) + 1
        phn = phn[:-1]
    else:
        tone = 3
    return phn.lower(), tone


def refine_syllables(syllables):
    tones = []
    phonemes = []
    for phn_list in syllables:
        for i in range(len(phn_list)):
            phn = phn_list[i]
            phn, tone = refine_ph(phn)
            phonemes.append(phn)
            tones.append(tone)
    return phonemes, tones


def distribute_phone(n_phone, n_word):
    phones_per_word = [0] * n_word
    for task in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word


def get_syllable_positions(phones):
    """Assign syllable positions to phonemes: 0=padding, 1=onset, 2=nucleus, 3=coda"""
    # English vowels (nuclei)
    vowels = {"aa", "ae", "ah", "ao", "aw", "ay", "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh", "uw"}

    positions = []
    for phone in phones:
        if phone == "_":  # Padding
            positions.append(0)
        elif phone in vowels:  # Vowel = nucleus
            positions.append(2)
        else:  # Consonant - need to determine if onset or coda
            # For simplicity, treat all consonants as onset (1)
            # A more sophisticated approach would track syllable boundaries
            positions.append(1)

    return positions


def text_to_words(text):
    tokens = tokenizer.tokenize(text)
    words = []
    for idx, t in enumerate(tokens):
        if t.startswith("▁"):
            words.append([t[1:]])
        else:
            if t in punctuation:
                if idx == len(tokens) - 1:
                    words.append([f"{t}"])
                else:
                    if not tokens[idx + 1].startswith("▁") and tokens[idx + 1] not in punctuation:
                        if idx == 0:
                            words.append([])
                        words[-1].append(f"{t}")
                    else:
                        words.append([f"{t}"])
            else:
                if idx == 0:
                    words.append([])
                words[-1].append(f"{t}")
    return words


def g2p(text, phoneme=None, skip_pos=False):
    phones = []
    tones = []
    phone_len = []
    words = text_to_words(text)

    for word in words:
        temp_phones, temp_tones = [], []
        if len(word) > 1:
            if "'" in word:
                word = ["".join(word)]
        for w in word:
            if w in punctuation:
                temp_phones.append(w)
                temp_tones.append(0)
                continue
            if w.upper() in eng_dict:
                phns, tns = refine_syllables(eng_dict[w.upper()])
                temp_phones += [post_replace_ph(i) for i in phns]
                temp_tones += tns
            else:
                phone_list = list(filter(lambda p: p != " ", _g2p(w)))
                phns = []
                tns = []
                for ph in phone_list:
                    if ph in arpa:
                        ph, tn = refine_ph(ph)
                        phns.append(ph)
                        tns.append(tn)
                    else:
                        phns.append(ph)
                        tns.append(0)
                temp_phones += [post_replace_ph(i) for i in phns]
                temp_tones += tns
        phones += temp_phones
        tones += temp_tones
        phone_len.append(len(temp_phones))

    word2ph = []
    for token, pl in zip(words, phone_len):
        word_len = len(token)
        aaa = distribute_phone(pl, word_len)
        word2ph += aaa

    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]
    word2ph = [1] + phone_len + [1]
    assert len(phones) == len(tones), text
    assert len(phones) == sum(word2ph), text

    if not skip_pos:
        # Use word-based segmentation with boundary labels for consistency
        # Convert word2ph to boundary labels: 1=Begin, 2=Middle, 3=End
        word_pos = []
        for i, count in enumerate(word2ph):
            if i == 0 or i == len(word2ph) - 1:  # Padding
                word_pos.extend([0] * count)
            else:
                if count == 1:
                    word_pos.append(1)  # Single phone word: Begin
                elif count == 2:
                    word_pos.extend([1, 3])  # Two phone word: Begin, End
                else:
                    word_pos.extend([1] + [2] * (count - 2) + [3])  # Multi phone word: Begin, Middle..., End
        # Generate syllable positions for phonemes
        syllable_pos = get_syllable_positions(phones)
    else:
        word_pos = [0] * len(phones)
        syllable_pos = [0] * len(phones)

    return phones, tones, word2ph, word_pos, syllable_pos


if __name__ == "__main__":
    text = "In this paper, we propose 1 DSPGAN, a GAN-based universal vocoder."
    phones, tones, word2ph, word_pos, syllable_pos = g2p(text)
    print("Text:", text)
    print("Phones:", phones)
    print("Tones:", tones)
    print("Word2ph:", word2ph)
