import re
from matcha.text.english.g2p import g2p
from matcha.text.cantonese.numbers import normalize_numbers  # Reuse English numbers

rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "．": ".",
    "…": "...",
    "···": "...",
    "・・・": "...",
    "·": ",",
    "・": ",",
    "、": ",",
    "$": ".",
    "“": "'",
    "”": "'",
    '"': "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "—": "-",
    "−": "-",
    "～": "-",
    "~": "-",
    "「": "'",
    "」": "'",
}


def replace_punctuation(text):
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)
    return replaced_text


def text_normalize(text):
    text = normalize_numbers(text)
    text = replace_punctuation(text)
    text = re.sub(r"([,;.\?\!])([\w])", r"\1 \2", text)
    return text


def clean_text(text, phoneme=None, skip_pos=False):
    norm_text = text_normalize(text)
    phones, tones, word2ph, word_pos, syllable_pos = g2p(norm_text, phoneme, skip_pos)
    return norm_text, phones, tones, word_pos, syllable_pos


if __name__ == "__main__":
    text = "Hello world! This is a test."
    norm_text, phones, tones, word_pos, syllable_pos = clean_text(text)
    print("Original:", text)
    print("Normalized:", norm_text)
    print("Phones:", phones)
