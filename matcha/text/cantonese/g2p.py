from matcha.text.cantonese.symbols import punctuation
import re
import unicodedata
from pydips import BertModel
from typing import Optional
import pycantonese
import ToJyutping

ws_model = BertModel()


def word2jyutping(word):
    jyutpings = [
        pycantonese.characters_to_jyutping(w)[0][1]
        for w in word
        if unicodedata.name(w, "").startswith("CJK UNIFIED IDEOGRAPH")
    ]

    if None in jyutpings:
        raise ValueError(f"Failed to convert {word} to jyutping: {jyutpings}")

    return " ".join(jyutpings)


def jyutping_to_onsets_nucleuses_codas_tones(jyutping_syllables):
    onsets_nucleuses_codas = []
    tones = []
    word2ph = []
    syllable_pos = []

    try:
        for syllable in jyutping_syllables:
            if syllable in punctuation:
                onsets_nucleuses_codas.append(syllable)
                tones.append(0)
                word2ph.append(1)  # Add 1 for punctuation
                syllable_pos.append(0)  # Punctuation has no syllable position
            else:
                onset, nucleus, coda, tone = parse_jyutping(syllable)
                num_phones = 0

                if onset != "":
                    onsets_nucleuses_codas.append("^" + onset)
                    tones.append(int(tone))
                    syllable_pos.append(1)
                    num_phones += 1
                if nucleus != "":
                    onsets_nucleuses_codas.append(nucleus)
                    tones.append(int(tone))
                    syllable_pos.append(2)
                    num_phones += 1
                if coda != "":
                    onsets_nucleuses_codas.append(coda + "$")
                    tones.append(int(tone))
                    syllable_pos.append(3)
                    num_phones += 1
                word2ph.append(num_phones)
    except Exception as e:
        raise ValueError(f"Failed to parse jyutping: {jyutping_syllables}") from e

    assert len(onsets_nucleuses_codas) == len(tones)
    return onsets_nucleuses_codas, tones, word2ph, syllable_pos


def get_jyutping(text):
    jyutping_array = []
    punct_pattern = re.compile(r"^[{}]+$".format(re.escape("".join(punctuation))))
    syllables = ToJyutping.get_jyutping_list(text)

    for word, syllable in syllables:
        if punct_pattern.match(word):
            puncts = re.split(r"([{}])".format(re.escape("".join(punctuation))), word)
            for punct in puncts:
                if len(punct) > 0:
                    jyutping_array.append(punct)
        else:
            # match multiple jyutping eg: liu4 ge3, or single jyutping eg: liu4
            if not re.search(r"^([a-z]+[1-6]+[ ]?)+$", syllable):
                raise ValueError(f"Failed to convert {word} to jyutping: {syllable}")

            jyutping_array.append(syllable)

    return jyutping_array


def parse_jyutping(jyutping: str):
    x = pycantonese.parse_jyutping(jyutping)

    if not x or len(x) == 0:
        raise ValueError(f"Failed to parse jyutping: {jyutping}")
    x = x[0]  # Take the first parsed result

    return x.onset, x.nucleus, x.coda, x.tone


def g2p(
    text: str,
    jyutping: Optional[str] = None,
    skip_pos: bool = False,
):
    if jyutping is None:
        jyutping = get_jyutping(text)
    elif isinstance(jyutping, str):
        jyutping = jyutping.split(" ")

    if len(jyutping) != len(text):
        print(
            "Warning: Jyutping length does not match text length. Using default conversion.",
            text,
            jyutping,
        )
        jyutping = get_jyutping(text)

    phones, tones, word2ph, syllable_pos = jyutping_to_onsets_nucleuses_codas_tones(jyutping)
    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]

    if not skip_pos:
        ws = ws_model.cut(text, mode="coarse")

        assert sum([len(x) for x in ws]) == len(text), "BERT output length mismatch with text length."

        ws_labels = []
        word_pos = []

        for w in ws:
            if len(w) == 0:
                continue
            elif len(w) == 1:
                ws_labels.append(1)  # Begin
            elif len(w) == 2:
                ws_labels.extend([1, 3])  # End
            elif len(w) > 2:
                ws_labels.extend([1] + [2] * (len(w) - 2) + [3])  # Begin, Middle, End

        for i, ws_label in enumerate(ws_labels):
            num_phones = word2ph[i]
            word_pos.extend([ws_label] * num_phones)
        word_pos = [0] + word_pos + [0]
        syllable_pos = [0] + syllable_pos + [0]

        assert (
            len(phones) == len(tones) == len(word_pos) == len(syllable_pos)
        ), "Phones, tones, word positions, and syllable positions must have the same length."
    else:
        # Return default zero lists when skip_pos is True
        word_pos = [0] * len(phones)
        syllable_pos = [0] * len(phones)

    return phones, tones, word2ph, word_pos, syllable_pos


if __name__ == "__main__":
    from matcha.text.cantonese.cleaners import text_normalize

    text = "佢邊係想辭工吖，跳下草裙舞想加人工之嘛。"
    text = text_normalize(text)

    print("normalized text", text)

    phones, tones, word2ph, word_pos, syllable_pos = g2p(text)

    print(phones, tones, word2ph, word_pos, syllable_pos)
