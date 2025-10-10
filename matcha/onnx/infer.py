import argparse
import os
import warnings
from pathlib import Path
from time import perf_counter

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

from matcha.cli import plot_spectrogram_to_numpy
from matcha.utils.utils import intersperse
from matcha.text import sequence_to_text, text_to_sequence
from matcha.data.text_mel_datamodule import load_spk_embedding, get_spk_embedding


def process_text(text: str, jyutping: str, prompt_audio: str, add_blank=True, speaker_session=None):
    phone_token_ids, tones, word_pos, syllable_pos = text_to_sequence(text, jyutping)
    if add_blank:
        phone_token_ids = intersperse(phone_token_ids, 0)
        tones = intersperse(tones, 0)
        word_pos = intersperse(word_pos, 0)
        syllable_pos = intersperse(syllable_pos, 0)
    x = torch.tensor(phone_token_ids, dtype=torch.long)
    tones = torch.tensor(tones, dtype=torch.long)
    word_pos = torch.tensor(word_pos, dtype=torch.long)
    syllable_pos = torch.tensor(syllable_pos, dtype=torch.long)
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long)
    prompt_speech = librosa.load(prompt_audio, sr=16000)[0]
    spk_emb = get_spk_embedding(
        prompt_speech,
        speaker_session,
    )
    spk_emb = torch.tensor(spk_emb, dtype=torch.float)[None]
    return {
        "x": x,
        "x_lengths": x_lengths,
        "tones": tones,
        "word_pos": word_pos,
        "syllable_pos": syllable_pos,
        "spk_emb": spk_emb,
    }


def validate_args(args):
    assert (
        args.text or args.file
    ), "Either text or file must be provided Matcha-T(ea)TTS need sometext to whisk the waveforms."
    assert args.temperature >= 0, "Sampling temperature cannot be negative"
    assert args.speaking_rate >= 0, "Speaking rate must be greater than 0"
    return args


def write_wavs(model, inputs, output_dir, external_vocoder=None):
    if external_vocoder is None:
        print("The provided model has the vocoder embedded in the graph.\nGenerating waveform directly")
        t0 = perf_counter()
        wavs, wav_lengths = model.run(None, inputs)
        infer_secs = perf_counter() - t0
        mel_infer_secs = vocoder_infer_secs = None
    else:
        print("[🍵] Generating mel using Matcha")
        mel_t0 = perf_counter()
        mels, mel_lengths = model.run(None, inputs)
        mel_infer_secs = perf_counter() - mel_t0
        print("Generating waveform from mel using external vocoder")
        vocoder_inputs = {external_vocoder.get_inputs()[0].name: mels}
        vocoder_t0 = perf_counter()
        wavs = external_vocoder.run(None, vocoder_inputs)[0]
        vocoder_infer_secs = perf_counter() - vocoder_t0
        wavs = wavs.squeeze(1)
        wav_lengths = mel_lengths * 256
        infer_secs = mel_infer_secs + vocoder_infer_secs

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, (wav, wav_length) in enumerate(zip(wavs, wav_lengths)):
        output_filename = output_dir.joinpath(f"output_{i + 1}.wav")
        audio = wav[:wav_length]
        print(f"Writing audio to {output_filename}")
        sf.write(output_filename, audio, 22050, "PCM_24")

    wav_secs = wav_lengths.sum() / 22050
    print(f"Inference seconds: {infer_secs}")
    print(f"Generated wav seconds: {wav_secs}")
    rtf = infer_secs / wav_secs
    if mel_infer_secs is not None:
        mel_rtf = mel_infer_secs / wav_secs
        print(f"Matcha RTF: {mel_rtf}")
    if vocoder_infer_secs is not None:
        vocoder_rtf = vocoder_infer_secs / wav_secs
        print(f"Vocoder RTF: {vocoder_rtf}")
    print(f"Overall RTF: {rtf}")


def write_mels(model, inputs, output_dir):
    t0 = perf_counter()
    mels, mel_lengths = model.run(None, inputs)
    infer_secs = perf_counter() - t0

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, mel in enumerate(mels):
        output_stem = output_dir.joinpath(f"output_{i + 1}")
        plot_spectrogram_to_numpy(mel.squeeze(), output_stem.with_suffix(".png"))
        np.save(output_stem.with_suffix(".numpy"), mel)

    wav_secs = (mel_lengths * 256).sum() / 22050
    print(f"Inference seconds: {infer_secs}")
    print(f"Generated wav seconds: {wav_secs}")
    rtf = infer_secs / wav_secs
    print(f"RTF: {rtf}")


def main():
    parser = argparse.ArgumentParser(
        description=" 🍵 Matcha-TTS: A fast TTS architecture with conditional flow matching"
    )
    parser.add_argument(
        "model",
        type=str,
        help="ONNX model to use",
    )
    parser.add_argument("--vocoder", type=str, default=None, help="Vocoder to use (defaults to None)")
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize")
    parser.add_argument("--file", type=str, default=None, help="Text file to synthesize")
    parser.add_argument("--spk", type=int, default=None, help="Speaker ID")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.667,
        help="Variance of the x0 noise (default: 0.667)",
    )
    parser.add_argument(
        "--speaking-rate",
        type=float,
        default=1.0,
        help="change the speaking rate, a higher value means slower speaking rate (default: 1.0)",
    )
    parser.add_argument("--gpu", action="store_true", help="Use CPU for inference (default: use GPU if available)")
    parser.add_argument("--prompt-audio", type=str, required=True, help="Prompt audio file for speaker embedding")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory (default: ./output)")

    args = parser.parse_args()
    args = validate_args(args)

    speaker_embedding_onnx_session = load_spk_embedding("pretrained_models/campplus.onnx")

    if args.gpu:
        providers = ["GPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    model = ort.InferenceSession(args.model, providers=providers)

    model_inputs = model.get_inputs()
    model_outputs = list(model.get_outputs())

    if args.text:
        text_lines = args.text.splitlines()
    else:
        with open(args.file, encoding="utf-8") as file:
            text_lines = file.read().splitlines()

    processed_lines = [
        process_text(line, "", args.prompt_audio, True, speaker_embedding_onnx_session) for line in text_lines
    ]
    x = [line["x"] for line in processed_lines]
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
    tones = [line["tones"] for line in processed_lines]
    tones = torch.nn.utils.rnn.pad_sequence(tones, batch_first=True)
    word_pos = [line["word_pos"] for line in processed_lines]
    word_pos = torch.nn.utils.rnn.pad_sequence(word_pos, batch_first=True)
    syllable_pos = [line["syllable_pos"] for line in processed_lines]
    syllable_pos = torch.nn.utils.rnn.pad_sequence(syllable_pos, batch_first=True)
    x_lengths = torch.tensor([line["x_lengths"].item() for line in processed_lines], dtype=torch.long)
    spk_emb = processed_lines[0]["spk_emb"].repeat(len(processed_lines), 1)  # repeat for batch

    inputs = {
        "x": x.numpy(),
        "x_lengths": x_lengths.numpy(),
        "scales": np.array([args.temperature, args.speaking_rate], dtype=np.float32),
        "tone": tones.numpy(),
        "word_pos": word_pos.numpy(),
        "syllable_pos": syllable_pos.numpy(),
        "spk_emb": spk_emb.numpy(),
    }

    has_vocoder_embedded = model_outputs[0].name == "wav"
    if has_vocoder_embedded:
        write_wavs(model, inputs, args.output_dir)
    elif args.vocoder:
        external_vocoder = ort.InferenceSession(args.vocoder, providers=providers)
        write_wavs(model, inputs, args.output_dir, external_vocoder=external_vocoder)
    else:
        warn = "[!] A vocoder is not embedded in the graph nor an external vocoder is provided. The mel output will be written as numpy arrays to `*.npy` files in the output directory"
        warnings.warn(warn, UserWarning)
        write_mels(model, inputs, args.output_dir)


if __name__ == "__main__":
    main()
