import numpy as np
from datasets import Dataset
from matcha.data.text_mel_datamodule import TextMelDataset

# Create dummy audio data (1 second at 22050 Hz)
dummy_audio = np.random.randn(22050).astype(np.float32)

# Create dummy dataset
data = [
    {
        "text": "Hello world",
        "lang": "en",
        "audio": {"array": dummy_audio, "path": "dummy1.wav", "sampling_rate": 22050},
    },
    {"text": "你好世界", "lang": "zh", "audio": {"array": dummy_audio, "path": "dummy2.wav", "sampling_rate": 22050}},
    {"text": "早晨", "lang": "yue", "audio": {"array": dummy_audio, "path": "dummy3.wav", "sampling_rate": 22050}},
]

ds = Dataset.from_list(data)

# Create dataset instance
dataset = TextMelDataset(
    ds,
    n_spks=1,
    add_blank=True,
    n_fft=1024,
    n_mels=80,
    sample_rate=22050,
    hop_length=256,
    win_length=1024,
    f_min=0.0,
    f_max=8000,
    data_parameters=None,
    seed=42,
    load_durations=False,
    tmp_dir="tmp",
    skip_pos=False,
    skip_spk_emb=True,  # Skip speaker embedding for testing
)

print("Testing dataloader with dummy multilingual data...")

for i in range(len(dataset)):
    datapoint = dataset[i]
    print(f"\nSample {i+1}:")
    print(f"Text: {datapoint['x_text']}")
    print(f"Lang: {datapoint['lang']}")
    print(f"Phones shape: {datapoint['x'].shape}")
    print(f"Phones (first 10): {datapoint['x'][:10]}")
    print(f"Tones (first 10): {datapoint['tone'][:10]}")
    print(f"Mel shape: {datapoint['y'].shape}")

print("\nTest completed successfully!")
