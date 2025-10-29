# Linguistic Features: word_pos and syllable_pos

This document explains the linguistic features used in the Matcha-TTS multilingual system for consistent text-to-speech processing across English, Mandarin Chinese, and Cantonese.

## Overview

The system uses two key linguistic features to provide structural information about text:
- **word_pos**: Word boundary positions
- **syllable_pos**: Syllable structure positions

Both features use consistent label schemes across all languages to ensure proper multilingual training.

## word_pos: Word Boundary Positions

Word positions indicate where words begin, continue, and end within a sequence of phonemes. Word segmentation is performed using **pydips (BERT-based model)** for accurate linguistic word boundary detection.

### Labels
- `0`: Padding/sequence boundary
- `1`: Word begin (start of a word)
- `2`: Word middle (continuation of a word)
- `3`: Word end (end of a word)

### Examples

#### English
```
Text: "Hello world"
Phones: ['_', 'hh', 'ah', 'l', 'ow', 'w', 'er', 'l', 'd', '_']
Word pos: [0, 1, 2, 2, 3, 1, 2, 2, 3, 0]

Analysis:
- "Hello": positions 1-4 → [1, 2, 2, 3] (begin, middle, middle, end)
- "world": positions 5-8 → [1, 2, 2, 3] (begin, middle, middle, end)
```

#### Mandarin Chinese
```
Text: "你好世界"
Phones: ['_', 'n', 'i', 'h', 'ao', 'sh', 'i', 'j', 'ie', '_']
Word pos: [0, 1, 1, 1, 1, 1, 1, 3, 3, 0]

Analysis:
- "你好": positions 1-6 → [1, 1, 1, 1, 1, 1] (single word unit)
- "世界": positions 7-8 → [3, 3] (single word unit)
```

#### Cantonese
```
Text: "早晨"
Phones: ['_', '^z', 'o', 'u$', '^s', 'a', 'n$', '_']
Word pos: [0, 1, 1, 1, 3, 3, 3, 0]

Analysis:
- "早": positions 1-3 → [1, 1, 1] (single word unit)
- "晨": positions 4-6 → [3, 3, 3] (single word unit)
```

## syllable_pos: Syllable Structure Positions

Syllable positions indicate the structural components within syllables (onset, nucleus, coda).

### Labels
- `0`: Padding/sequence boundary
- `1`: Onset/Initial (consonant sounds before the vowel)
- `2`: Nucleus (vowel sound, the core of the syllable)
- `3`: Coda/Final (consonant sounds after the vowel)

### Examples

#### English
```
Text: "Hello world"
Phones: ['_', 'hh', 'ah', 'l', 'ow', 'w', 'er', 'l', 'd', '_']
Syllable pos: [0, 1, 2, 1, 2, 1, 2, 1, 1, 0]

Analysis:
- 'hh' (h): onset
- 'ah' (ɑ): nucleus
- 'l' (l): onset (simplified classification)
- 'ow' (oʊ): nucleus
- 'w' (w): onset
- 'er' (ɜr): nucleus
- 'l' (l): onset
- 'd' (d): onset
```

#### Mandarin Chinese
```
Text: "你好世界"
Phones: ['_', 'n', 'i', 'h', 'ao', 'sh', 'i', 'j', 'ie', '_']
Syllable pos: [0, 1, 2, 1, 2, 1, 2, 1, 2, 0]

Analysis:
- "n" (initial) + "i" (final) → onset + nucleus
- "h" (initial) + "ao" (final) → onset + nucleus
- "sh" (initial) + "i" (final) → onset + nucleus
- "j" (initial) + "ie" (final) → onset + nucleus
```

#### Cantonese
```
Text: "早晨"
Phones: ['_', '^z', 'o', 'u$', '^s', 'a', 'n$', '_']
Syllable pos: [0, 1, 2, 3, 1, 2, 3, 0]

Analysis:
- "早" (zou): ^z (onset) + o (nucleus) + u$ (coda)
- "晨" (san): ^s (onset) + a (nucleus) + n$ (coda)
```

## Implementation Notes

### Language-Specific Considerations

- **English**: Uses simplified syllable classification (vowels = nucleus, consonants = onset)
- **Mandarin**: Uses traditional initial-final syllable structure
- **Cantonese**: Uses full onset-nucleus-coda syllable structure

### Consistency Across Languages

Both `word_pos` and `syllable_pos` use the same label scheme (`0, 1, 2, 3`) across all languages, ensuring:
- Consistent model input dimensions
- Proper multilingual training
- No language-specific feature conflicts

### Usage in TTS

These features help the model understand:
- Word boundaries for prosody and phrasing
- Syllable structure for natural pronunciation
- Cross-lingual consistency for multilingual synthesis

## Technical Details

- Features are generated during text preprocessing
- **Word segmentation**: Uses pydips (BERT-based model) for accurate linguistic word boundaries
- Applied at the phoneme level for fine-grained control
- Integrated into the Matcha-TTS data pipeline
- Essential for high-quality multilingual speech synthesis</content>
<parameter name="filePath">/notebooks/bert-vits2/frankenstein-matcha/LINGUISTIC_FEATURES.md