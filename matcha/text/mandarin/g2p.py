from typing import Optional, List
from matcha.text.mandarin.symbols import punctuation
import pypinyin
from pypinyin import Style
from pydips import BertModel

ws_model = BertModel()


def text_to_pinyin(text: str) -> List[tuple]:
    """Convert Chinese text to pinyin initials and finals."""
    initials_list = pypinyin.pinyin(text, style=Style.INITIALS, heteronym=False)
    finals_list = pypinyin.pinyin(text, style=Style.FINALS_TONE3, heteronym=False)
    initials_flat = [item[0] for item in initials_list]
    finals_flat = [item[0] for item in finals_list]
    return list(zip(initials_flat, finals_flat))


def pinyin_to_phonemes(pinyin_syllables: List[tuple]) -> tuple:
    """Convert pinyin syllables to phonemes."""
    phonemes = []
    tones = []
    word2ph = []
    syllable_pos = []

    for initial, final in pinyin_syllables:
        if initial in punctuation:
            # Assuming final is also punctuation
            phonemes.append(initial)
            tones.append(0)
            word2ph.append(1)
            syllable_pos.append(0)
        else:
            tone = 0
            if final and final[-1].isdigit():
                tone = int(final[-1])
                final = final[:-1]

            if initial:
                phonemes.append(initial)
                tones.append(tone)
                syllable_pos.append(1)  # Initial
            if final:
                phonemes.append(final)
                tones.append(tone)
                syllable_pos.append(2)  # Final

            word2ph.append(len([x for x in [initial, final] if x]))

    return phonemes, tones, word2ph, syllable_pos


def g2p(
    text: str,
    pinyin_input: Optional[str] = None,
    skip_pos: bool = False,
):
    """Grapheme to phoneme conversion for Mandarin."""
    if pinyin_input is None:
        pinyin_syllables = text_to_pinyin(text)
    else:
        # For now, assume pinyin_input is not used or handle differently
        raise NotImplementedError("pinyin_input not supported yet")

    phones, tones, word2ph, syllable_pos_out = pinyin_to_phonemes(pinyin_syllables)

    # Add padding
    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]

    if not skip_pos:
        # Use BERT-based word segmentation for proper word boundaries
        ws = ws_model.cut(text, mode="coarse")

        assert sum([len(x) for x in ws]) == len(text), "BERT output length mismatch with text length."

        ws_labels = []
        word_pos_temp = []

        for w in ws:
            if len(w) == 0:
                continue
            elif len(w) == 1:
                ws_labels.append(1)  # Begin
            elif len(w) == 2:
                ws_labels.extend([1, 3])  # Begin, End
            elif len(w) > 2:
                ws_labels.extend([1] + [2] * (len(w) - 2) + [3])  # Begin, Middle..., End

        # Extend word boundary labels to phonemes
        for i, ws_label in enumerate(ws_labels):
            num_phones = word2ph[i]
            word_pos_temp.extend([ws_label] * num_phones)

        word_pos = word_pos_temp
        word_pos = [0] + word_pos + [0]
        syllable_pos_out = [0] + syllable_pos_out + [0]
    else:
        # Return default zero lists when skip_pos is True
        word_pos = [0] * len(phones)
        syllable_pos_out = [0] * len(phones)

    return phones, tones, word2ph, word_pos, syllable_pos_out


if __name__ == "__main__":
    text = "你好世界！"
    phones, tones, word2ph, word_pos, syllable_pos = g2p(text)
    print("Text:", text)
    print("Phones:", phones)
    print("Tones:", tones)
    print("Word2ph:", word2ph)
    print("Word pos:", word_pos)
    print("Syllable pos:", syllable_pos)
