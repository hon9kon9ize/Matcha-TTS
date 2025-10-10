import re
import cn2an
from matcha.text.cantonese.symbols import punctuation
from matcha.text.cantonese.g2p import g2p
import unicodedata


rep_map = {
    "：": ",",
    "︰": ",",
    "；": ",",
    "，": ",",
    "﹐": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "﹖": "?",
    "﹗": "!",
    "\n": ".",
    "·": ",",
    "、": ",",
    "丶": ",",
    "...": "…",
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
    "_": "-",
}

replacement_chars = {
    "ㄧ": "一",
    "—": "一",
    "更": "更",
    "不": "不",
    "料": "料",
    "聯": "聯",
    "行": "行",
    "利": "利",
    "謢": "護",
    "岀": "出",
    "鎭": "鎮",
    "戯": "戲",
    "旣": "既",
    "立": "立",
    "來": "來",
    "年": "年",
    "㗇": "蝦",
    "臺": "台",
    "檯": "枱",
    "櫈": "凳",
}


def normalizer(x):
    x = cn2an.transform(x, "an2cn")

    return x


def replace_punctuation(text):
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)
    replaced_text = "".join(
        c for c in replaced_text if unicodedata.name(c, "").startswith("CJK UNIFIED IDEOGRAPH") or c in punctuation
    )
    # replace multiple punctuations with single one
    replaced_text = re.sub(r"([{}])\1+".format(re.escape("".join(punctuation))), r"\1", replaced_text)

    return replaced_text


def replace_chars(text):
    for k, v in replacement_chars.items():
        text = text.replace(k, v)
    return text


def text_normalize(text):
    text = text.strip()
    text = normalizer(text)
    text = replace_punctuation(text)
    text = replace_chars(text)
    return text


def clean_text(text, jyutping=None, skip_pos=False):
    norm_text = text_normalize(text)
    phones, tones, word2ph, word_pos, syllable_pos = g2p(norm_text, jyutping, skip_pos)

    return norm_text, phones, tones, word_pos, syllable_pos


if __name__ == "__main__":
    pass
