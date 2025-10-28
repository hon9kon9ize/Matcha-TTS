from typing import Optional, List
from matcha.text.mandarin.symbols import punctuation
import pypinyin
from pypinyin import Style


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
    word_pos: Optional[List[int]] = None,
    syllable_pos: Optional[List[int]] = None,
):
    """Grapheme to phoneme conversion for Mandarin."""
    if pinyin_input is None:
        pinyin_syllables = text_to_pinyin(text)
    else:
        # If pinyin_input provided, assume it's space-separated pinyin
        pinyin_list = pinyin_input.split()
        # For simplicity, assume no initials-finals split, but since we need tuples, perhaps parse
        # For now, assume pinyin_input is not used or handle differently
        raise NotImplementedError("pinyin_input not supported yet")

    phones, tones, word2ph, syllable_pos_out = pinyin_to_phonemes(pinyin_syllables)

    # Add padding
    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]

    if not skip_pos:
        if word_pos is None:
            # Simple word position assignment
            word_pos = []
            for i, count in enumerate(word2ph):
                word_pos.extend([i + 1] * count)
        word_pos = [0] + word_pos + [0]
        syllable_pos_out = [0] + syllable_pos_out + [0]

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
