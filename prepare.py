#!/usr/bin/env python3
"""
Prepare script to preprocess Hugging Face dataset by adding phoneme, tone, word_pos, syllable_pos, and speaker embedding fields.
This script processes text and audio data for multilingual TTS training.
"""

import argparse
import librosa
import numpy as np
from datasets import load_dataset
from matcha.text.english.cleaners import clean_text as clean_text_en
from matcha.text.cantonese.cleaners import clean_text as clean_text_yue
from matcha.text.mandarin.cleaners import clean_text as clean_text_zh
from matcha.text.symbols import symbol_to_id
from matcha.feature_extractions.spkemb_speechbrain import SpeechBrainSpkEmbExtractor

speaker_extractor = SpeechBrainSpkEmbExtractor(device="cpu")


def process_row(row):
    """Process a single row to add phoneme, positional, and speaker embedding information."""
    text = row["text"]
    phone = row.get("phone", None)
    lang = row.get("lang", "en")  # Default to English if no language specified

    # Select appropriate cleaner based on language
    if lang == "en":
        clean_func = clean_text_en
    elif lang == "zh":
        clean_func = clean_text_zh
    elif lang == "yue":
        clean_func = clean_text_yue
    else:
        raise ValueError(f"Unsupported language: {lang}")

    # Clean text and extract phonemes, tones, and positions
    _, phones, tones, word_pos, syllable_pos = clean_func(text, phone)

    # Convert phoneme symbols to IDs
    phone_ids = [symbol_to_id.get(p, symbol_to_id.get("UNK", 0)) for p in phones]

    # Shift phoneme tones by 1 to avoid conflict with blank tone 0
    if tones:
        pad_start = tones[0]
        pad_end = tones[-1]
        phoneme_tones = tones[1:-1]
        phoneme_tones = [t + 1 for t in phoneme_tones]
        shifted_tones = [pad_start] + phoneme_tones + [pad_end]
    else:
        shifted_tones = tones

    # Extract speaker embedding if audio is available
    spk_emb = None
    if "audio" not in row:
        raise ValueError("Audio data is required for TTS training but not found in dataset row")
    if "audio" in row:
        try:
            # Extract audio data
            audio_data = np.array(row["audio"]["array"])
            sr = row["audio"]["sampling_rate"]

            # Resample to 16kHz for speaker embedding extraction
            if sr != 16000:
                audio_16k = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
            else:
                audio_16k = audio_data

            # Extract speaker embedding
            spk_emb = speaker_extractor.forward(audio_16k, 16000)

        except (KeyError, ValueError, RuntimeError) as e:
            print(f"Warning: Could not extract speaker embedding for sample: {e}")
            spk_emb = None

    # Add processed fields to the row
    row["phone"] = phones  # Phoneme text (jyutping, pinyin, or ARPABET)
    row["phone_token_ids"] = phone_ids  # Phoneme token IDs
    row["tones"] = shifted_tones
    row["word_pos"] = word_pos
    row["syllable_pos"] = syllable_pos
    row["spk_emb"] = spk_emb  # Speaker embedding

    return row


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset for multilingual TTS")
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path or Hugging Face dataset identifier to load"
    )
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the processed dataset")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to process (default: train)")
    parser.add_argument(
        "--push_to_hub", type=str, default=None, help="Hugging Face repository name to push the processed dataset to"
    )

    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset_path}, split: {args.split}")
    try:
        ds = load_dataset(args.dataset_path, split=args.split)
    except ValueError as e:
        if "save_to_disk" in str(e):
            from datasets import load_from_disk

            ds = load_from_disk(args.dataset_path)
        else:
            raise

    print(f"Processing {len(ds)} samples...")
    ds = ds.map(process_row, desc="Processing text data")

    if args.push_to_hub:
        print(f"Pushing processed dataset to Hugging Face Hub: {args.push_to_hub}")
        ds.push_to_hub(args.push_to_hub)
    else:
        print(f"Saving processed dataset to {args.output_path}")
        ds.save_to_disk(args.output_path)

    print("Preprocessing complete!")


if __name__ == "__main__":
    main()
