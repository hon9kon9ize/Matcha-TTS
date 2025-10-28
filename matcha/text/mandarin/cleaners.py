import re
from matcha.text.mandarin.symbols import punctuation
from matcha.text.mandarin.g2p import g2p

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
    "⋯": "…",
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
    "～": "-",
    "~": "-",
    "「": "'",
    "」": "'",
}


def replace_punctuation(text):
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)
    # Keep only Chinese characters and punctuation
    replaced_text = re.sub(r"[^\u4e00-\u9fff" + re.escape("".join(punctuation)) + r"\s]", "", replaced_text)
    return replaced_text


def text_normalize(text):
    text = text.strip()
    text = replace_punctuation(text)
    return text


def clean_text(text, pinyin=None, skip_pos=False):
    norm_text = text_normalize(text)
    phones, tones, word2ph, word_pos, syllable_pos = g2p(norm_text, pinyin, skip_pos)
    return norm_text, phones, tones, word_pos, syllable_pos


if __name__ == "__main__":
    text = "你好世界！这是一个测试。"
    norm_text, phones, tones, word_pos, syllable_pos = clean_text(text)
    print("Original:", text)
    print("Normalized:", norm_text)
    print("Phones:", phones)
