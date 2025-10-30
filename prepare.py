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
    try:
        _, phones, tones, word_pos, syllable_pos = clean_func(text, phone)
    except Exception as e:
        print(f"Failed to clean text for row: {e}")
        return None

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

    # Add processed fields to the row
    row["phone"] = phones  # Phoneme text (jyutping, pinyin, or ARPABET)
    row["phone_token_ids"] = phone_ids  # Phoneme token IDs
    row["tones"] = shifted_tones
    row["word_pos"] = word_pos
    row["syllable_pos"] = syllable_pos

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
    ds = ds.map(process_row, num_proc=16, desc="Processing text data")

    if args.push_to_hub:
        print(f"Pushing processed dataset to Hugging Face Hub: {args.push_to_hub}")
        ds.push_to_hub(args.push_to_hub)
    else:
        print(f"Saving processed dataset to {args.output_path}")
        ds.save_to_disk(args.output_path)

    print("Preprocessing complete!")


if __name__ == "__main__":
    main()
