from matcha.text.cantonese import g2p, text_normalize


def clean_text(text, jyutping=None, skip_pos=False):
    norm_text = text_normalize(text)
    phones, tones, word2ph, word_pos, syllable_pos = g2p(norm_text, jyutping, skip_pos)

    return norm_text, phones, tones, word_pos, syllable_pos


if __name__ == "__main__":
    pass
