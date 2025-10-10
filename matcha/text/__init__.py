"""from https://github.com/keithito/tacotron"""

from matcha.text.cantonese.symbols import symbols
from matcha.text.cantonese.cleaners import clean_text

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}  # pylint: disable=unnecessary-comprehension


class UnknownCleanerException(Exception):
    pass


def text_to_sequence(text, phone=None, skip_pos=False):
    _, phones, tones, word_pos, syllable_pos = clean_text(text, phone, skip_pos)
    phone_token_ids = cleaned_text_to_sequence(phones)

    return phone_token_ids, tones, word_pos, syllable_pos


def cleaned_text_to_sequence(cleaned_text):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = [_symbol_to_id[symbol] for symbol in cleaned_text]
    return sequence


def sequence_to_text(sequence):
    """Converts a sequence of IDs back to a string"""
    result = ""
    for symbol_id in sequence:
        s = _id_to_symbol[symbol_id]
        result += s
    return result
